# STEM-GNN

This repository contains pretraining and finetuning scripts for STEM-GNN, plus multiple evaluation scripts.

**Quick Start**
1. Go to the repo root (`/home/mca25001/STEM-GNN`).
2. Create and activate the environment.

```bash
conda env create -f environment.yml
conda activate STEM-GNN
```

**Project Layout**
- `STEM-GNN/pretrain.py`: pretrain entry
- `STEM-GNN/finetune.py`: generic finetune entry
- `STEM-GNN/scripts/`: evaluation/variant scripts
- `STEM-GNN/data/`: datasets (organized per script expectations)
- `STEM-GNN/ckpts/`: checkpoints output
- `config/`: config templates (`pretrain.yaml`, `finetune.yaml`)

**Pretrain**
Run from the repo root:

```bash
python STEM-GNN/pretrain.py --use_params
```

Common example:
```bash
python STEM-GNN/pretrain.py \
  --use_params \
  --gpu 0 \
  --pretrain_dataset all \
  --pretrain_epochs 50
```

Output path:
- Pretrained checkpoints are saved to `STEM-GNN/ckpts/pretrain_model/`

**Finetune**
Generic finetune (node task):

```bash
python STEM-GNN/finetune.py --use_params --finetune_dataset cora --gpu 0
```

If you need to load a specific pretrained checkpoint:
```bash
python STEM-GNN/finetune.py \
  --use_params \
  --finetune_dataset cora \
  --pretrain_model_epoch 25
```

Or point directly to the checkpoint folder:
```bash
python STEM-GNN/finetune.py \
  --use_params \
  --finetune_dataset cora \
  --pretrain_path STEM-GNN/ckpts/pretrain_model/your_run
```

**Scripts (in `STEM-GNN/scripts/`)**
All scripts below support **node** tasks only:

```bash
# Degree Shift OOD
python STEM-GNN/scripts/degree_shift_ood.py --use_params --finetune_dataset cora

# Homophily Shift OOD
python STEM-GNN/scripts/homophily_shift_ood.py --use_params --finetune_dataset cora

# Missing Feature
python STEM-GNN/scripts/missing_feature.py --use_params --finetune_dataset cora

# Random Edge Drop
python STEM-GNN/scripts/random_edge_drop.py --use_params --finetune_dataset cora

# Tri-objective finetune
python STEM-GNN/scripts/tri_objective.py --use_params --finetune_dataset cora
```

**Notes**
- `--use_params` loads defaults from `config/finetune.yaml` or `config/pretrain.yaml`.
- Put data under `STEM-GNN/data`, and checkpoints under `STEM-GNN/ckpts`.
