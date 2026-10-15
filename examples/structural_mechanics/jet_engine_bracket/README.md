# Jet Engine Bracket: Training, Out-of-Distribution Detection and Transfer Learning

This example trains a [GeoTransolver](https://arxiv.org/abs/2512.20399) surrogate for
the linear-elastic response of jet engine brackets, and shows two things a user of a
trained surrogate needs next:

1. **Train** (`train.py`): a GeoTransolver predicts the displacement and von Mises
   stress at every mesh node of a bracket under four load cases, trained on the
   2,138 generated brackets of DeepJEB.
2. **Detect out-of-distribution designs** (`ood.py`): before trusting a prediction
   for a new design, check whether the design is like the training designs. Three
   detectors are compared on the human-designed SimJEB brackets: the PhysicsNeMo
   geometry guardrail, the PhysicsNeMo embedded OOD guard, and the Mahalanobis
   distance of the model's own intermediate features.
3. **Transfer** (`finetune.py`): adapt the DeepJEB model to SimJEB with low-rank
   adapters (LoRA, `physicsnemo.experimental.peft`), training 0.6 M of its 5.1 M
   parameters.

## Datasets

| Dataset | Designs | Mesh | Source and license |
| --- | --- | --- | --- |
| DeepJEB | 2,138 generated brackets | 10-node tetrahedra | [Hugging Face](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB), ODC-By 1.0 |
| SimJEB | 381 human-designed brackets | 4-node tetrahedra | [Harvard Dataverse](https://doi.org/10.7910/DVN/XFUWJG), CC0 1.0 |

Both datasets share the bolt interfaces, the material and the four load cases of
the GE jet engine bracket problem: vertical, horizontal and diagonal forces and a
torsional moment at the clevis, with the four bolt holes clamped. Their
OptiStruct results give the displacement and the von Mises stress at every node.

Splits (`splits/`): DeepJEB uses the dataset authors' "latent" split, 1,933
training and 205 test designs, with a seeded 10% of the training designs held out
for validation (1,740 / 193 / 205). SimJEB uses its official 77 test designs and
59 validation designs drawn from the rest (241 / 59 / 77); four designs are
excluded for defects of their mesh or deck, listed with the reason in
`splits/simjeb.json`.

## Setup

```bash
pip install -r requirements.txt

DATA=<data folder>
# 1. Download the parts of both datasets that are used (DeepJEB 77 GB, SimJEB 7.1 GB)
python download_data.py --output $DATA

# 2. Convert them to one DomainMesh per design and load case (bracket_io.py)
python convert_deepjeb.py --input $DATA/deepjeb/extracted \
    --output $DATA/deepjeb/converted --workers 16
python convert_simjeb.py --input $DATA/simjeb/extracted \
    --output $DATA/simjeb/converted --workers 16
```

Each sample holds every mesh node with its region (interior, free surface, clamped
bolt interfaces, loaded clevis interface), the displacement and the von Mises
stress, the boundary triangles, and the load case. The fields are divided by
per-load-case scales of the DeepJEB training designs (`stats/deepjeb_scales.json`,
recomputed by `compute_scales.py`).

## Recipe 1: train on DeepJEB

The model (`conf/model/geotransolver.yaml`) is the stock
`physicsnemo.models.geotransolver.GeoTransolver` with 8 GALE blocks, width 192,
256 physics slices and local features. Every mesh node is a query point; it sees
its coordinates and the one-hot of its region. The geometry context is every node
voxel-downsampled at 2 mm, so the model sees where the part is solid, and the
global context is the one-hot of the load case.

```bash
torchrun --nproc_per_node 4 train.py data.root=$DATA/deepjeb/converted
# DeepJEB test, then the model as is on SimJEB test
python evaluate.py data.root=$DATA/deepjeb/converted
python evaluate.py data.root=$DATA/simjeb/converted data.split=splits/simjeb.json \
    data.subset=test output=outputs/deepjeb/evaluation/simjeb_test
```

Training runs 50 epochs (every design under every load case, one sample per GPU
and step) with Muon for the weight matrices and AdamW for the rest, a two-epoch
warm-up, a cosine decay, bfloat16 autocast and activation checkpointing. A rerun
resumes from the latest checkpoint. For a quick test:

`python train.py data.root=... data.max_designs=8 training.epochs=2`.

## Recipe 2: out-of-distribution detection

```bash
# optional: the error of every SimJEB design, to rank the scores against
python evaluate.py data.root=$DATA/simjeb/converted data.split=splits/simjeb.json \
    data.subset=all output=outputs/deepjeb/evaluation/simjeb_all
python ood.py reference.root=$DATA/deepjeb/converted \
    query.root=$DATA/simjeb/converted \
    query.errors=outputs/deepjeb/evaluation/simjeb_all_cases.csv
```

All three detectors are fitted on the DeepJEB training designs and score each
design by its distance from them:

- **Geometry guardrail** (`GeometryGuardrail`): a Gaussian density on 22 global
  shape descriptors of the surface mesh, that is, a Mahalanobis distance in
  descriptor space. A design is flagged when its score exceeds the 99th
  percentile of the training designs (`WARN`, or `REJECT` above the 99.9th).
- **Embedded guard** (`OODGuard`): the distance between the L2-normalized pooled
  latent of the geometry tokenizer and its 10 nearest training latents, flagged
  above 1.5 times the 99th percentile of the training designs.
- **Latent Mahalanobis distance** (this example): the Mahalanobis distance of the
  node mean of one GALE block's output (by default the fourth of eight), with a
  Ledoit-Wolf shrinkage covariance, flagged above the 99th percentile of the
  DeepJEB validation designs. The same distance of the pooled tokenizer latent is
  reported for comparison.

`outputs/ood/summary.json` gives, per detector, the share of DeepJEB test designs
flagged (false alarms), the share of SimJEB designs flagged, the AUC of SimJEB
against DeepJEB test, and the Spearman correlation of the score with the stress
error of each SimJEB design; `scores.csv` has every score and verdict.

## Recipe 3: transfer to SimJEB with LoRA

```bash
python finetune.py data.root=$DATA/simjeb/converted
python evaluate.py adapter=outputs/simjeb_lora/adapter.lora \
    data.root=$DATA/simjeb/converted data.split=splits/simjeb.json \
    output=outputs/simjeb_lora/simjeb_test
```

The pre-trained model is frozen and rank-16 adapters on every linear layer are
trained with AdamW for 30 epochs on the SimJEB training designs, with the
per-sample relative squared error as the loss. `outputs/simjeb_lora/summary.json`
compares the pre-trained and the adapted model on the SimJEB validation and test
designs.

## Results

Single runs of the recipes as configured, on 4 GPUs. Errors are the mean
per-sample relative L2 errors over every node.

| Model | DeepJEB test stress | SimJEB test stress | SimJEB test displacement |
| --- | --- | --- | --- |
| DeepJEB model | TBD | TBD | TBD |
| + LoRA on SimJEB | | TBD | TBD |

| Detector | DeepJEB test flagged | SimJEB flagged | AUC | Spearman with error |
| --- | --- | --- | --- | --- |
| Geometry guardrail | TBD | TBD | TBD | TBD |
| Embedded guard | TBD | TBD | TBD | TBD |
| Latent Mahalanobis, block 4 | TBD | TBD | TBD | TBD |

## Files

| File | Purpose |
| --- | --- |
| `download_data.py` | download and extract the datasets |
| `nastran.py`, `convert_deepjeb.py`, `convert_simjeb.py`, `bracket_io.py` | read the decks and results, write the samples |
| `compute_scales.py`, `stats/` | per-load-case normalization scales |
| `datapipe.py` | samples as model inputs and normalized targets |
| `train.py`, `evaluate.py`, `ood.py`, `finetune.py` | the recipes |
| `conf/` | Hydra configurations |
| `splits/` | design ids of the splits |

## References

- C. Adams et al., GeoTransolver, [arXiv:2512.20399](https://arxiv.org/abs/2512.20399).
- S. Hong et al., DeepJEB: 3D deep learning-based synthetic jet engine bracket
  dataset, J. Mech. Des. 147(4):041703, 2025.
- E. Whalen et al., SimJEB: Simulated jet engine bracket dataset, Computer
  Graphics Forum 40(5), 2021.
- E. J. Hu et al., LoRA: Low-rank adaptation of large language models, ICLR 2022.
- K. Lee et al., A simple unified framework for detecting out-of-distribution
  samples and adversarial attacks, NeurIPS 2018.
- O. Ledoit and M. Wolf, A well-conditioned estimator for large-dimensional
  covariance matrices, J. Multivar. Anal. 88(2), 2004.
