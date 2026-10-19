# Peripheral Blood Smear Transformer

A transformer model that combines the cells of a peripheral blood smear into a slide-level prediction. Cell features and predicted cell types are embedded as tokens, processed by a type-aware Transformer encoder, aggregated by attention pooling, and classified by a task-specific head.

## Structure

- `cell_type_classifier/`: cell-type classifier
- `main.py`, `test.py`, `models/`, ...: slide-level Transformer
- `demographic_model/demographic_model.py`: logistic regression with age and sex
- `comparison.py`: paired bootstrap comparison of AUROCs

The cell encoder was trained with the DiffAE code (https://github.com/phizaz/diffae).

## Installation

```bash
pip install -r requirements.txt   # Python 3.12
```
## Usage
### Cell-type classifier

Train/test input: `torch.save({"feats": features (N, 768), "labels": annotated cell types (N,)})`

Cell types: 0 neutrophil, 1 lymphocyte, 2 monocyte, 3 eosinophil, 4 basophil, 5 immature cell, 6 nRBC, 7 platelet, 8 unidentifiable

```bash
python cell_type_classifier/train.py --train_feats train.pt --output_dir out
python cell_type_classifier/test.py --ckpt out/model.pt --test_feats test.pt --output results.json
```

Inference predicts the cell type of every cell in each slide and produces the input of the slide-level Transformer.
- Input: `torch.save(features (N, 768))`
- Output: `torch.save((features (N, 768), type_ids (N,)))`

```bash
python cell_type_classifier/infer.py --ckpt out/model.pt --input_dir feats --output_dir feats
```

### Slide-level Transformer

Input:
- `feats.pt`: `torch.save((features (N, 768), type_ids (N,)))`
- `{train,valid,test,test_h}.csv`: columns `specimen_id`, `label`, `total_nuc`

```bash
python main.py --root_path feats --anno_path anno --experiment_name crp
python main.py --root_path feats --anno_path anno --experiment_name wbc --task multiclass --num_classes 3
python test.py --log_dir logs/crp/{timestamp}
```

- Platelet-included model: `--remove_type 8`
- Cell-type only model: `--config_file` with `encoder: {use_patch_feat: false}`

### Demographic model

Requires additional columns `age` and `sex` (`M` for male).

```bash
python demographic_model/demographic_model.py --anno_path anno --experiment_name crp --task binary --output_dir out
```

### Paired comparison

Input: `predictions.npz` from `test.py` or `demographic_model.py`, with `{split}_probs`, `{split}_labels`, `{split}_ids` (split = `test`, `test_h`). Δ = AUROC(a) − AUROC(b).

```bash
python comparison.py --pred_a <base_dir>/predictions.npz --pred_b <demo_dir>/predictions.npz --task binary --output comparison.json
```
