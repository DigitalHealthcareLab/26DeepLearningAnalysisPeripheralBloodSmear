import os
import yaml
import torch
from dataclasses import dataclass, asdict, field


@dataclass
class PBSTEncConfig:
    patch_feat_dim: int = 768
    type_vocab_size: int = 10
    type_dropout: float = 0.1
    use_type_embed: bool = True
    use_patch_feat: bool = True

    d_model: int = 768
    n_layers: int = 10
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.15
    attn_dropout: float = 0.2
    drop_path_rate: float = 0.1
    qkv_bias: bool = True

    pool_dropout: float = 0.1

    num_classes: int | None = 1
    task: str = "binary"            # 'binary' | 'multiclass'


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    num_workers: int = 10

    lr: float = 2e-5
    weight_decay: float = 0.1
    start_factor: float = 1e-3
    warmup_epochs: int = 10
    min_lr: float = 1e-7

    patience: int = 6


@dataclass
class DataConfig:
    root_path: str = ""
    anno_path: str = ""
    id_column: str = "specimen_id"
    label_column: str = "label"
    label_dtype = torch.float32
    total_nuc_threshold: float = 100.0
    remove_type: list = field(default_factory=lambda: [7, 8])


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    seed: int = 42
    deterministic: bool = True


@dataclass
class LoggingConfig:
    log_dir: str = "./logs/"


@dataclass
class ExperimentConfig:
    encoder: PBSTEncConfig = field(default_factory=PBSTEncConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    experiment_name: str = "default_experiment"

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_config(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save(os.path.join(save_dir, "config.yaml"))

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            encoder=PBSTEncConfig(**config_dict.get("encoder", {})),
            train=TrainConfig(**config_dict.get("train", {})),
            data=DataConfig(**config_dict.get("data", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            runtime=RuntimeConfig(**config_dict.get("runtime", {})),
            experiment_name=config_dict.get("experiment_name", "default_experiment"),
        )

    def override_from_dict(self, overrides):
        for key, value in overrides.items():
            keys = key.split(".")

            if len(keys) == 1:
                setattr(self, keys[0], value)
            elif len(keys) == 2:
                section, attr = keys
                section_obj = getattr(self, section)
                current_value = getattr(section_obj, attr)
                if current_value is not None:
                    target_type = type(current_value)
                    converted_value = target_type(value)
                    setattr(section_obj, attr, converted_value)
                else:
                    setattr(section_obj, attr, value)
