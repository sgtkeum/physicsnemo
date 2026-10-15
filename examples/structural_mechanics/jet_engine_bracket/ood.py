# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Recipe 2: is a new design like the designs the model was trained on?

Three detectors score every design against the DeepJEB training designs:

- **geometry guardrail** (:class:`~physicsnemo.experimental.guardrails.geometry.GeometryGuardrail`):
  a Gaussian density on 22 global shape descriptors of the surface mesh, which is the Mahalanobis
  distance in that descriptor space; its own verdict (``WARN`` or ``REJECT`` above the 99th or
  99.9th percentile of the training designs) flags a design;
- **embedded guard** (:class:`~physicsnemo.experimental.guardrails.embedded.OODGuard`): the mean
  distance between the L2-normalized pooled latent of the model's geometry tokenizer and its
  :math:`k` nearest training latents, flagged above its own threshold;
- **latent Mahalanobis distance**: the Mahalanobis distance of the model's own features, the
  node mean of the output of one GALE block (and, for comparison, of the pooled tokenizer
  latent), with a Ledoit-Wolf shrinkage covariance of the training designs and a threshold at a
  high percentile of held-out validation designs.

Reported: per design the scores and verdicts (``scores.csv``), and per detector the share of
flagged DeepJEB test designs (false alarms) and SimJEB designs, the AUC of SimJEB against
DeepJEB test, and, given the per-sample errors of ``evaluate.py`` on the screened designs, the
Spearman correlation of each score with the stress error (``summary.json``, ``scores.png``).

    python ood.py reference.root=<data>/deepjeb query.root=<data>/simjeb \
        query.errors=outputs/deepjeb/evaluation/simjeb_all_cases.csv
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import hydra
import numpy as np
import torch
from bracket_io import load_split, sample_path
from datapipe import load_scales, model_inputs, sample_inputs
from evaluate import load_model
from nastran import LOADS
from omegaconf import DictConfig
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from utils import resolve

from physicsnemo.experimental.guardrails.embedded import OODGuard
from physicsnemo.experimental.guardrails.geometry import (
    GeometryGuardrail,
    extract_features,
)
from physicsnemo.mesh import DomainMesh


class LatentProbe:
    r"""Forward hooks that keep, per forward pass, the pooled geometry-tokenizer latent and the node mean of one block output."""

    def __init__(self, model: torch.nn.Module, block: int) -> None:
        self.latents: dict[str, torch.Tensor] = {}
        tokenizer = model.context_builder.geometry_tokenizer
        self.handles = [
            tokenizer.register_forward_hook(self._tokens),
            model.blocks[block].register_forward_hook(self._block),
        ]

    def _tokens(self, module, inputs, output) -> None:
        self.latents["encoder"] = (
            output.detach().mean(dim=(1, 2)).float()
        )  # (B, H, S, D) -> (B, D)

    def _block(self, module, inputs, output) -> None:
        streams = output if isinstance(output, (list, tuple)) else [output]
        self.latents["block"] = torch.cat(
            [s.detach().mean(dim=1) for s in streams], dim=-1
        ).float()  # (B, N, C) -> (B, C)

    def close(self) -> None:
        for h in self.handles:
            h.remove()


@torch.no_grad()
def describe(model, probe, root, shapes, cfg, scales, device) -> dict[int, dict]:
    r"""Per design its surface mesh (for the guardrail) and its latents (one forward pass)."""
    out = {}
    for i, shape in enumerate(shapes, 1):
        domain = DomainMesh.load(sample_path(root, shape, cfg.load))
        batch = {
            k: v[None]
            for k, v in sample_inputs(
                domain, scales, cfg.coordinate_scale_mm, cfg.token_voxel_mm
            ).items()
        }
        model(**model_inputs(batch, device))
        out[shape] = {
            "surface": domain.boundaries["surface"],
            **{k: v[0].cpu().double() for k, v in probe.latents.items()},
        }
        if i % 100 == 0:
            print(f"{i}/{len(shapes)} designs of {root}", flush=True)
    return out


def mahalanobis(reference: np.ndarray):
    r"""Squared Mahalanobis distance from the training latents, with a Ledoit-Wolf covariance."""
    fit = LedoitWolf().fit(reference)
    precision, mean = np.linalg.inv(fit.covariance_), fit.location_
    return lambda x: np.einsum("ij,jk,ik->i", x - mean, precision, x - mean)


@hydra.main(version_base=None, config_path="conf", config_name="ood")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scales, rng = load_scales(resolve(cfg.scales)), np.random.default_rng(cfg.seed)
    ref_split, query_split = (
        load_split(resolve(cfg.reference.split)),
        load_split(resolve(cfg.query.split)),
    )
    train = ref_split["train"]
    if cfg.reference.designs and cfg.reference.designs < len(train):
        train = sorted(rng.choice(train, cfg.reference.designs, replace=False).tolist())
    n = cfg.max_designs
    groups = {
        "deepjeb_train": (cfg.reference.root, train[:n]),
        "deepjeb_val": (cfg.reference.root, ref_split["val"][:n]),
        "deepjeb_test": (cfg.reference.root, ref_split["test"][:n]),
        "simjeb": (
            cfg.query.root,
            sorted(s for k in cfg.query.subsets for s in query_split[k])[:n],
        ),
    }
    train = groups["deepjeb_train"][1]

    model = load_model(cfg.checkpoint, None, device)
    probe = LatentProbe(model, cfg.block)
    designs = {
        g: describe(model, probe, root, shapes, cfg, scales, device)
        for g, (root, shapes) in groups.items()
    }
    probe.close()

    scores = defaultdict(dict)  # detector -> group -> scores in design order
    flags = defaultdict(dict)
    # geometry guardrail: fitted on the training surfaces; WARN or REJECT flags a design
    guardrail = GeometryGuardrail(
        method="gmm",
        gmm_components=1,
        warn_pct=99.0,
        reject_pct=99.9,
        device=str(device),
    )
    guardrail.fit([d["surface"] for d in designs["deepjeb_train"].values()])
    for g, d in designs.items():
        surfaces = [v["surface"] for v in d.values()]
        features = torch.stack(
            [extract_features(m.to(device), return_tensor=True) for m in surfaces]
        )
        scores["geometry_guardrail"][g] = (
            guardrail.density.score(features).cpu().numpy().astype(float)
        )  # -log density
        flags["geometry_guardrail"][g] = np.array(
            [r["status"] != "OK" for r in guardrail.query(surfaces)]
        )
    # embedded guard: kNN on the normalized pooled tokenizer latent, its own threshold
    encoder = {
        g: torch.stack([v["encoder"] for v in d.values()]).float()
        for g, d in designs.items()
    }
    guard = OODGuard(
        buffer_size=len(train),
        geometry_embed_dim=encoder["deepjeb_train"].shape[1],
        knn_k=cfg.knn_k,
    )
    guard.collect(geometry_latent=encoder["deepjeb_train"])
    guard.compute_threshold()
    for g, z in encoder.items():
        scores["embedded_guard"][g] = guard.score_geometry(z).numpy().astype(float)
        flags["embedded_guard"][g] = scores["embedded_guard"][g] > float(
            guard.knn_threshold
        )
    # latent Mahalanobis distances: threshold from the validation designs
    for name, key in (
        ("mahalanobis_encoder", "encoder"),
        (f"mahalanobis_block{cfg.block + 1}", "block"),
    ):
        latents = {
            g: torch.stack([v[key] for v in d.values()]).numpy()
            for g, d in designs.items()
        }
        distance = mahalanobis(latents["deepjeb_train"])
        scores[name] = {g: distance(x) for g, x in latents.items()}
        threshold = float(np.percentile(scores[name]["deepjeb_val"], cfg.percentile))
        flags[name] = {g: s > threshold for g, s in scores[name].items()}

    errors = {}
    if cfg.query.errors:
        per_design = defaultdict(list)
        for row in csv.DictReader(open(resolve(cfg.query.errors))):
            per_design[int(row["shape_id"])].append(float(row["von_mises_rel_l2"]))
        errors = {
            s: float(np.mean(v)) for s, v in per_design.items() if len(v) == len(LOADS)
        }
    simjeb = groups["simjeb"][1]
    summary = {
        "reference_designs": len(train),
        "block": cfg.block,
        "load": cfg.load,
        "detectors": {},
    }
    for name in scores:
        s, f = scores[name], flags[name]
        entry = {
            "deepjeb_test_flagged": float(f["deepjeb_test"].mean()),
            "simjeb_flagged": float(f["simjeb"].mean()),
            "auc_simjeb_vs_deepjeb_test": float(
                roc_auc_score(
                    np.r_[np.zeros(len(s["deepjeb_test"])), np.ones(len(s["simjeb"]))],
                    np.r_[s["deepjeb_test"], s["simjeb"]],
                )
            ),
        }
        known = [i for i, shape in enumerate(simjeb) if shape in errors]
        if len(known) > 2:
            entry["spearman_with_stress_error"] = float(
                spearmanr(s["simjeb"][known], [errors[simjeb[i]] for i in known])[0]
            )
        summary["detectors"][name] = entry
    output = resolve(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "scores.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "group",
                "shape_id",
                *[f"{n}_{c}" for n in scores for c in ("score", "flagged")],
            ]
        )
        for g, (_, shapes) in groups.items():
            for i, shape in enumerate(shapes):
                writer.writerow(
                    [
                        g,
                        shape,
                        *[
                            v
                            for n in scores
                            for v in (f"{scores[n][g][i]:.6g}", int(flags[n][g][i]))
                        ],
                    ]
                )
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    plot(scores, flags, output / "scores.png")
    print(json.dumps(summary, indent=1))


def plot(scores: dict, flags: dict, path) -> None:
    r"""Score distributions of DeepJEB test and SimJEB designs per detector."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(scores), figsize=(4 * len(scores), 3))
    for ax, (name, s) in zip(np.atleast_1d(axes), scores.items()):
        values = np.r_[s["deepjeb_test"], s["simjeb"]]
        log = values.min() > 0 and values.max() / values.min() > 100
        bins = np.geomspace(values.min(), values.max(), 40) if log else 40
        ax.hist(s["deepjeb_test"], bins=bins, alpha=0.6, label="DeepJEB test")
        ax.hist(s["simjeb"], bins=bins, alpha=0.6, label="SimJEB")
        ax.set_xscale("log" if log else "linear")
        ax.set_title(
            f"{name}\nflagged: DeepJEB {100 * flags[name]['deepjeb_test'].mean():.0f}%, SimJEB {100 * flags[name]['simjeb'].mean():.0f}%",
            fontsize=8,
        )
        ax.set_xlabel("score")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
