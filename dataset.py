import os

import pandas as pd
import torch
from torch.utils.data import Dataset

def load_anno_df(anno_path, config):
    df = pd.read_csv(anno_path, dtype={config.id_column: str})

    threshold = config.total_nuc_threshold
    tn = pd.to_numeric(df["total_nuc"], errors="coerce")
    df = df[tn >= threshold].reset_index(drop=True)
    return df


class PBSDSDataset(Dataset):
    def __init__(
        self,
        config,
        anno_df,
    ):
        super().__init__()
        self.config = config
        self.root_path = config.root_path
        self.id_column = config.id_column
        self.label_column = config.label_column
        self.remove_type = config.remove_type

        self.samples = []
        for _, row in anno_df.iterrows():
            sid = str(row[self.id_column])
            label_val = row[self.label_column]
            self.samples.append((sid, label_val))

    def __len__(self):
        return len(self.samples)

    def _load_feats(
        self, sample_id
    ):
        pt_path = os.path.join(self.root_path, sample_id, "feats.pt")
        data = torch.load(pt_path, map_location="cpu", weights_only=True)

        patch_features, type_ids = data

        patch_features = patch_features.to(torch.float32)
        type_ids = type_ids.to(torch.long)
        keep_mask = torch.ones_like(type_ids, dtype=torch.bool)

        if self.remove_type is not None:
            rt = torch.as_tensor(self.remove_type, dtype=torch.long).view(-1)
            keep_mask &= ~torch.isin(type_ids, rt)
        patch_features = patch_features[keep_mask]
        type_ids = type_ids[keep_mask]

        return patch_features, type_ids


    def __getitem__(self, idx):
        sample_id, label_val = self.samples[idx]
        patch_features, type_ids = self._load_feats(sample_id)

        if self.config.label_dtype == torch.long:
            label_tensor = torch.tensor(int(label_val), dtype=torch.long)
        else:
            label_tensor = torch.tensor(float(label_val), dtype=torch.float32)

        return {
            "patch_features": patch_features,
            "type_ids": type_ids,
            "label": label_tensor,
        }
