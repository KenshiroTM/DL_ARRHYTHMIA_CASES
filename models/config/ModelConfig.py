from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

from config import weights_folder

class WeightMethod(Enum):
    INVERSE = "inverse"
    SQRT = "sqrt"          # mniej agresywna niż inverse
    EFFECTIVE = "effective" # z papieru Cui et al.
    NONE = None

class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerType(Enum):
    COSINE = "cosine"
    STEP = "step"
    PLATEAU = "plateau"
    NONE = None


@dataclass
class ModelConfig:
    experiment_name: str

    signal_cols: list = None # kolumny z sygnałami EKG
    feature_cols: list = None # kolumny z cechami ręcznie wpisanymi
    label_col: str = "scp_codes" # kolumna z etykietami klas

    amp: bool = True # mixed precision
    num_workers: int = 4 # ile workerów ładuje dane
    persistent_workers: bool = True # nie zamyka workerów między workerami
    pin_memory: bool = True # szybszy transfer z cpu do gpu
    compile_model: bool = True # optymalizacja grafu

    epochs: int = 20 # główne parametry
    batch_size: int = 64 # ile próbek naraz dodać, stabilniejsze gradienty
    lr: float = 1e-4 # krok wagi, jak za duży to za mocno skacze za mały to stoi w miejscu
    weight_decay: float = 1e-5 # regularyzacja, zapobiega overfittingowi
    optimizer: OptimizerType = OptimizerType.ADAMW # typ optymizera
    scheduler: SchedulerType = SchedulerType.COSINE # typ schedulera

    patience: int = 5 # ile czekać bez poprawy przed breakiem
    min_delta: float = 1e-4 # minimalna różnica aby uznać za poprawę

    # DLA SGD
    momentum: float = 0.9 # bezwładność gradientu

    # scheduler parametry
    step_size: int = 10 # co ile epok obniżyć lr
    gamma: float = 0.1 # przez ile pomnożyć lr przy obniżaniu
    sched_patience: int = 3 # ile epok spadku zanim obniżyć (plateau)

    augment: bool = False # augmentacja, zwiększa różnorodność
    noise_std: float = 0.01  # szum gaussowski dodawany do sygnału
    scale_low: float = 0.9  # skalowanie amplitudy  max (-10% domyślne)
    scale_high: float = 1.1 # sikalowanie amplitudy wysokiej (+10 domyślnie)
    time_shift_pct: float = 0.1  # przesunięcie w czasie

    class_weight_method: WeightMethod = None # jak ważyć klasy

    focal_gamma: float = 2.0           # skupianie lossa na trudnych próbkach
    focal_alpha: float = None          # dodatkowe ważenie per klasa w focal

    rhythm_weight: float = 0.05 # autoenkoder

    # nie nadpisywać
    save_path: Path = None # gdzie zapisać wagi
    model_type: str = None # tag do wyników plików

    def __post_init__(self):
        if self.signal_cols is None:
            self.signal_cols = ["signal"]
        if self.save_path is None:
            self.save_path = Path(weights_folder / self.experiment_name).with_suffix(".pth")
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def copy(self, **changes):
        """Kopiuje config i nadpisuje podane pola."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(changes)
        return ModelConfig(**current)