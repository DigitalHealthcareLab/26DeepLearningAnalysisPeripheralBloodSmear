import torch

@torch.no_grad()
def pad_and_stack_batch(
    samples,
    pad_token_id,
):
    B = len(samples)
    feat_dim = samples[0]["patch_features"].shape[1]

    lengths = [int(s["patch_features"].shape[0]) for s in samples]
    N_max = max(lengths)

    batch_patch_features = torch.zeros((B, N_max, feat_dim), dtype=torch.float32)
    batch_type_ids = torch.full((B, N_max), fill_value=pad_token_id, dtype=torch.long)
    batch_padding_mask = torch.ones((B, N_max), dtype=torch.bool)

    for i, s in enumerate(samples):
        n = s["patch_features"].shape[0]
        batch_patch_features[i, :n] = s["patch_features"].to(dtype=torch.float32)
        batch_type_ids[i, :n] = s["type_ids"].to(dtype=torch.long)
        batch_padding_mask[i, :n] = False

    return (
        batch_patch_features,
        batch_type_ids,
        batch_padding_mask,
    )


def collate_fn(
    batch,
    pad_token_id,
):
    patch_features, type_ids, padding_mask = pad_and_stack_batch(
        batch, pad_token_id=pad_token_id,
    )

    labels_list = [s["label"] for s in batch]
    labels = torch.stack(labels_list, dim=0)

    return {
        "patch_features": patch_features,
        "type_ids": type_ids,
        "padding_mask": padding_mask,
        "labels": labels,
    }
