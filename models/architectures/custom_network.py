"""
Własna sieć CNN do klasyfikacji arytmii EKG.
Architektura: wieloskalowa CNN z mechanizmem uwagi, do porównania z ECGFounderem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Mechanizm uwagi na kanałach (uproszczony Squeeze-and-Excitation).
    Pozwala sieci skupić się na ważniejszych odprowadzeniach EKG.
    """
    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, max(n_channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(n_channels // reduction, 1), n_channels),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (batch, channels, length)
        w = x.mean(dim=-1)          # global average pooling -> (batch, channels)
        w = self.fc(w)              # (batch, channels)
        return x * w.unsqueeze(-1)  # skalowanie kanałów


class MultiScaleBlock(nn.Module):
    """
    Blok wieloskalowej konwolucji — przetwarza sygnał EKG
    jednocześnie małymi (lokalne cechy) i dużymi (globalne wzorce) kernelami.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        mid = out_channels // 3

        # 3 równoległe konwolucje z różnymi kernelami
        self.branch_small  = self._conv(in_channels, mid, kernel=3,  stride=stride)  # lokalne
        self.branch_medium = self._conv(in_channels, mid, kernel=7,  stride=stride)  # średnie
        self.branch_large  = self._conv(in_channels, mid, kernel=15, stride=stride)  # globalne

        # wyrównanie kanałów po złączeniu gałęzi
        total = mid * 3
        self.merge = nn.Sequential(
            nn.Conv1d(total, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

        # shortcut connection (residual)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if (in_channels != out_channels or stride != 1) else nn.Identity()

        self.attention = ChannelAttention(out_channels)

    @staticmethod
    def _conv(in_ch, out_ch, kernel, stride):
        pad = kernel // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        s = self.branch_small(x)
        m = self.branch_medium(x)
        l = self.branch_large(x)

        # przycinamy do tego samego rozmiaru (mogą się różnić o 1 przy stride)
        min_len = min(s.shape[-1], m.shape[-1], l.shape[-1])
        out = torch.cat([s[..., :min_len], m[..., :min_len], l[..., :min_len]], dim=1)
        out = self.merge(out)
        out = self.attention(out)

        # residual
        sc = self.shortcut(x)
        if sc.shape[-1] != out.shape[-1]:
            sc = sc[..., :out.shape[-1]]
        return F.gelu(out + sc)


class CustomECGNet(nn.Module):
    """
    Własna sieć CNN do klasyfikacji arytmii EKG.

    Architektura:
        Wejście (batch, in_channels, 5000)
        -> stem conv
        -> 4 etapy MultiScaleBlock z rosnącą liczbą filtrów
        -> global average pooling
        -> classifier z dropout

    Parametry:
        in_channels: liczba odprowadzeń EKG (np. 3 dla [0,3,5] lub 12 dla wszystkich)
        n_classes:   liczba klas do klasyfikacji
        base_filters: bazowa liczba filtrów (domyślnie 32)
        dropout:     współczynnik dropout (domyślnie 0.4)
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_classes:   int = 8,
        base_filters: int = 32,
        dropout: float = 0.4
    ):
        super().__init__()
        f = base_filters  # skrót

        # stem — pierwsza konwolucja, szybkie zmniejszenie długości
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, f, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(f),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 5000 -> 1250
        )

        # 4 etapy — rosnąca głębokość, malejąca długość sygnału
        self.stage1 = self._make_stage(f,      f*2,  n_blocks=2, stride=2)   # 1250 -> 625
        self.stage2 = self._make_stage(f*2,    f*4,  n_blocks=3, stride=2)   # 625 -> 313
        self.stage3 = self._make_stage(f*4,    f*8,  n_blocks=3, stride=2)   # 313 -> 157
        self.stage4 = self._make_stage(f*8,    f*16, n_blocks=2, stride=2)   # 157 -> 79

        # global average pooling + klasyfikator
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f * 16, f * 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(f * 8, n_classes)
        )

        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [MultiScaleBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(MultiScaleBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, signal=None, x=None):
        if signal is not None:
            x = signal
        if x is None:
            raise ValueError("Podaj 'signal' lub 'x'")

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.classifier(x)


def build_custom_ecg_net(n_classes: int, in_channels: int = 3) -> CustomECGNet:
    """
    Fabryka modelu — tworzy CustomECGNet z domyślnymi parametrami.
    Użycie w setup_classificator_architecture.py zamiast Net1D gdy use_ecg_weights=False.
    """
    return CustomECGNet(
        in_channels=in_channels,
        n_classes=n_classes,
        base_filters=32,
        dropout=0.4
    )
