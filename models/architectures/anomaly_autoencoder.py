import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class UpsampleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(self.upsample(x))))


class BottleneckEncoder(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, latent_dim=64):
        super().__init__()
        f = base_filters
        self.conv_layers = nn.Sequential(
            ConvBlock1D(in_channels, f, kernel=15, stride=2, padding=7),
            ConvBlock1D(f, f*2, kernel=7, stride=2, padding=3),
            ConvBlock1D(f*2, f*4, kernel=7, stride=2, padding=3),
            ConvBlock1D(f*4, f*8, kernel=5, stride=2, padding=2),
            ConvBlock1D(f*8, f*8, kernel=5, stride=2, padding=2),
            ConvBlock1D(f*8, f*8, kernel=5, stride=2, padding=2),
        )
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(f*8, latent_dim), nn.LayerNorm(latent_dim), nn.GELU()
        )

    def forward(self, x):
        return self.to_latent(self.conv_layers(x))


class BottleneckDecoder(nn.Module):
    def __init__(self, latent_dim=64, base_filters=256, out_channels=1, target_length=5000):
        super().__init__()
        f = base_filters
        self.target_length = target_length
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, f * 8), nn.Unflatten(1, (f, 8))
        )
        self.ups = nn.ModuleList([
            UpsampleConv1D(f, f//2, kernel=5),      # 16
            UpsampleConv1D(f//2, f//4, kernel=5),   # 32
            UpsampleConv1D(f//4, f//8, kernel=7),   # 64
            UpsampleConv1D(f//8, f//16, kernel=7),  # 128
            UpsampleConv1D(f//16, f//32, kernel=7), # 256
            UpsampleConv1D(f//32, f//32, kernel=7), # 512
        ])
        self.final = nn.Conv1d(f//32, out_channels, kernel_size=7, padding=3)

    def forward(self, z):
        x = self.from_latent(z)
        for up in self.ups:
            x = up(x)
        if x.shape[-1] != self.target_length:
            x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        return self.final(x)


class VAEEncoder(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, latent_dim=64):
        super().__init__()
        f = base_filters
        self.conv_layers = nn.Sequential(
            ConvBlock1D(in_channels, f, kernel=15, stride=2, padding=7),
            ConvBlock1D(f, f*2, kernel=7, stride=2, padding=3),
            ConvBlock1D(f*2, f*4, kernel=7, stride=2, padding=3),
            ConvBlock1D(f*4, f*8, kernel=5, stride=2, padding=2),
            ConvBlock1D(f*8, f*8, kernel=5, stride=2, padding=2),
            ConvBlock1D(f*8, f*8, kernel=5, stride=2, padding=2),
        )
        self.to_latent = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.fc_mu = nn.Linear(f*8, latent_dim)
        self.fc_logvar = nn.Linear(f*8, latent_dim)
        nn.init.normal_(self.fc_logvar.weight, 0.0, 0.001)
        nn.init.constant_(self.fc_logvar.bias, 0.0)

    def forward(self, x):
        h = self.to_latent(self.conv_layers(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class RhythmEncoder(nn.Module):
    def __init__(self, ecg_length=5000, n_peaks=30):
        super().__init__()
        self.n_peaks = n_peaks
        self.ecg_length = ecg_length
        self.peak_detector = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, padding=15),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1), nn.Sigmoid(),
        )
        self.rhythm_fc = nn.Sequential(
            nn.Linear(n_peaks, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 16)
        )

    def forward(self, ecg):
        B = ecg.size(0)
        peak_prob = self.peak_detector(ecg)
        _, peak_idx = torch.topk(peak_prob.squeeze(1), k=self.n_peaks, dim=-1)
        peak_idx = torch.sort(peak_idx, dim=-1).values.float()
        rr = (peak_idx[:, 1:] - peak_idx[:, :-1]) / self.ecg_length
        if rr.size(1) < self.n_peaks:
            rr = torch.cat([rr, torch.zeros(B, self.n_peaks - rr.size(1), device=rr.device)], dim=1)
        return self.rhythm_fc(rr), rr


class SingleSignalAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, latent_channels=128):
        super().__init__()
        # backward compat: latent_channels ignored, uses bottleneck
        self.encoder = BottleneckEncoder(in_channels, base_filters, latent_dim=64)
        self.decoder = BottleneckDecoder(64, base_filters * 8, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode='linear', align_corners=False)
        return out

    def encode(self, x):
        return self.encoder(x)


class MultimodalAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, latent_channels=128):
        super().__init__()
        self.ae_ppg = SingleSignalAutoencoder(in_channels, base_filters, latent_channels)
        self.ae_ecg = SingleSignalAutoencoder(in_channels, base_filters, latent_channels)
        self.rhythm_enc = RhythmEncoder()
        self.rhythm_decoder = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 40))

    def forward(self, ppg, ecg):
        return self.ae_ppg(ppg), self.ae_ecg(ecg)

    def forward_with_rhythm(self, ppg, ecg):
        recon_ppg = self.ae_ppg(ppg)
        recon_ecg = self.ae_ecg(ecg)
        rhythm_z = self.rhythm_enc(ecg)[0]
        rhythm_recon = self.rhythm_decoder(rhythm_z)
        return recon_ppg, recon_ecg, rhythm_z, rhythm_recon

    def reconstruction_error(self, ppg, ecg, mode='sum'):
        with torch.no_grad():
            if mode == 'rhythm':
                rhythm_z = self.rhythm_enc(ecg)[0]
                rhythm_recon = self.rhythm_decoder(rhythm_z)
                return rhythm_recon.std(dim=1)
            recon_ppg, recon_ecg = self.forward(ppg, ecg)
            mse_ppg = F.mse_loss(recon_ppg, ppg, reduction='none').mean(dim=[1, 2])
            mse_ecg = F.mse_loss(recon_ecg, ecg, reduction='none').mean(dim=[1, 2])
            if mode == 'sum':
                return mse_ppg + mse_ecg
            elif mode == 'max':
                return torch.maximum(mse_ppg, mse_ecg)
            elif mode == 'ppg':
                return mse_ppg
            elif mode == 'ecg':
                return mse_ecg
            else:
                raise ValueError(f"Unknown mode: {mode}")


class MultimodalVAE(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, latent_channels=128, kl_weight=0.001):
        super().__init__()
        self.kl_weight = kl_weight
        self.enc_ppg = VAEEncoder(in_channels, base_filters, latent_dim=64)
        self.enc_ecg = VAEEncoder(in_channels, base_filters, latent_dim=64)
        self.dec_ppg = BottleneckDecoder(64, base_filters * 8, in_channels)
        self.dec_ecg = BottleneckDecoder(64, base_filters * 8, in_channels)
        self.rhythm_enc = RhythmEncoder()
        self.rhythm_decoder = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 40))

    def forward(self, ppg, ecg):
        mu_ppg, logvar_ppg = self.enc_ppg(ppg)
        mu_ecg, logvar_ecg = self.enc_ecg(ecg)
        z_ppg = self.enc_ppg.reparameterize(mu_ppg, logvar_ppg)
        z_ecg = self.enc_ecg.reparameterize(mu_ecg, logvar_ecg)
        return self.dec_ppg(z_ppg), self.dec_ecg(z_ecg), mu_ppg, logvar_ppg, mu_ecg, logvar_ecg

    def forward_with_rhythm(self, ppg, ecg):
        recon_ppg, recon_ecg, mu_ppg, logvar_ppg, mu_ecg, logvar_ecg = self.forward(ppg, ecg)
        rhythm_z, rr = self.rhythm_enc(ecg)
        rhythm_recon = self.rhythm_decoder(rhythm_z)
        return recon_ppg, recon_ecg, rhythm_z, rhythm_recon, mu_ppg, logvar_ppg, mu_ecg, logvar_ecg

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def loss_function(self, recon_ppg, ppg, recon_ecg, ecg, mu_ppg, logvar_ppg, mu_ecg, logvar_ecg):
        recon = F.mse_loss(recon_ppg, ppg, reduction='sum') + F.mse_loss(recon_ecg, ecg, reduction='sum')
        kl = self.kl_divergence(mu_ppg, logvar_ppg).sum() + self.kl_divergence(mu_ecg, logvar_ecg).sum()
        return recon + self.kl_weight * kl, recon, kl

    def reconstruction_error(self, ppg, ecg, mode='sum', use_rhythm=False, use_kl=False, n_samples=5):
        with torch.no_grad():
            if mode == 'rhythm':
                rhythm_z = self.rhythm_enc(ecg)[0]
                rhythm_recon = self.rhythm_decoder(rhythm_z)
                return rhythm_recon.std(dim=1)

            mu_ppg, logvar_ppg = self.enc_ppg(ppg)
            mu_ecg, logvar_ecg = self.enc_ecg(ecg)

            mse_ppg = mse_ecg = 0
            for _ in range(n_samples):
                z_ppg = self.enc_ppg.reparameterize(mu_ppg, logvar_ppg)
                z_ecg = self.enc_ecg.reparameterize(mu_ecg, logvar_ecg)
                mse_ppg += F.mse_loss(self.dec_ppg(z_ppg), ppg, reduction='none').mean(dim=[1, 2])
                mse_ecg += F.mse_loss(self.dec_ecg(z_ecg), ecg, reduction='none').mean(dim=[1, 2])
            mse_ppg /= n_samples
            mse_ecg /= n_samples

            if mode == 'sum':
                error = mse_ppg + mse_ecg
            elif mode == 'max':
                error = torch.maximum(mse_ppg, mse_ecg)
            elif mode == 'ppg':
                error = mse_ppg
            elif mode == 'ecg':
                error = mse_ecg
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if use_kl:
                error = error + 0.1 * (self.kl_divergence(mu_ppg, logvar_ppg) +
                                         self.kl_divergence(mu_ecg, logvar_ecg))
            if use_rhythm:
                error = error + 0.5 * self.rhythm_enc(ecg)[1].std(dim=1)
            return error


def build_multimodal_autoencoder():
    return MultimodalAutoencoder(base_filters=32, latent_channels=128)


def build_multimodal_vae(kl_weight=0.001):
    return MultimodalVAE(base_filters=32, latent_channels=128, kl_weight=kl_weight)