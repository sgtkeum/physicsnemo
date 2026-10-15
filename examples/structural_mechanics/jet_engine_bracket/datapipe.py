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

r"""Bracket samples as GeoTransolver inputs and normalized targets.

Every mesh node is a query point. Per node the model sees its coordinates (in units of
``coordinate_scale_mm``) and a one-hot of its region (interior, free surface, clamped bolt
interface, loaded clevis interface); the geometry context is every node voxel-downsampled at
``token_voxel_mm`` (one node per voxel), so it describes the solid, not only its surface; the
global context is the one-hot of the load case. Targets are the displacement and the von Mises
stress divided by per-load-case scales (:func:`load_scales`), so that the four load cases have
comparable magnitudes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from bracket_io import sample_path
from nastran import LOADS

from physicsnemo.mesh import DomainMesh

#: Number of node regions (see ``bracket_io.py``)
REGIONS = 4
#: Channels of the model output: displacement (3) and von Mises stress (1)
OUTPUTS = 4


def load_scales(path: str | Path) -> dict[str, dict[str, float]]:
    r"""Per-load-case normalization scales ``{"displacement": {load: mm}, "von_mises": {load: MPa}}``."""
    data = json.loads(Path(path).read_text())
    return {
        f: {load: float(data[f][load]) for load in LOADS}
        for f in ("displacement", "von_mises")
    }


def voxel_first(points: np.ndarray, voxel: float) -> np.ndarray:
    r"""Indices of the first point in every voxel of edge ``voxel``, in point order."""
    keys = np.floor(points / voxel).astype(np.int64)
    return np.sort(np.unique(keys, axis=0, return_index=True)[1])


def sample_inputs(
    domain: DomainMesh,
    scales: dict[str, dict[str, float]],
    coordinate_scale_mm: float = 100.0,
    token_voxel_mm: float = 2.0,
) -> dict[str, torch.Tensor]:
    r"""Model inputs and normalized targets of one sample.

    Parameters
    ----------
    domain : physicsnemo.mesh.DomainMesh
        A sample in the layout of ``bracket_io.py``.
    scales : dict[str, dict[str, float]]
        Per-load-case scales from :func:`load_scales`.
    coordinate_scale_mm : float, optional, default=100.0
        Length unit of the model coordinates, in mm.
    token_voxel_mm : float, optional, default=2.0
        Voxel edge of the geometry context, in mm.

    Returns
    -------
    dict[str, torch.Tensor]
        ``local_embedding`` :math:`(N, 3 + 4)`, ``local_positions`` :math:`(N, 3)`, ``geometry``
        :math:`(M, 3)`, ``global_embedding`` :math:`(1, 4)`, ``target`` :math:`(N, 4)` (normalized
        displacement and von Mises stress), ``scale`` :math:`(4,)` (multiplies ``target`` back to
        mm and MPa), ``shape_id`` and ``load_id``.
    """
    load_id = int(domain.global_data["load_id"])
    load = LOADS[load_id]
    points = domain.interior.points.float() / coordinate_scale_mm  # (N, 3)
    region = torch.nn.functional.one_hot(
        domain.interior.point_data["region_id"].long(), REGIONS
    ).float()  # (N, 4)
    keep = voxel_first(points.numpy(), token_voxel_mm / coordinate_scale_mm)
    scale = torch.tensor(
        [scales["displacement"][load]] * 3 + [scales["von_mises"][load]]
    )
    target = (
        torch.cat(
            [
                domain.interior.point_data["displacement"].float(),
                domain.interior.point_data["von_mises"].float()[:, None],
            ],
            dim=1,
        )
        / scale
    )
    return {
        "local_embedding": torch.cat([points, region], dim=1),
        "local_positions": points,
        "geometry": points[torch.from_numpy(keep)],
        "global_embedding": domain.global_data["load_onehot"].float()[None],
        "target": target,
        "scale": scale,
        "shape_id": torch.tensor(int(domain.global_data["shape_id"])),
        "load_id": torch.tensor(load_id),
    }


class BracketDataset(torch.utils.data.Dataset):
    r"""Every (design, load case) of a converted bracket dataset, as :func:`sample_inputs`.

    Parameters
    ----------
    root : str | pathlib.Path
        Converted dataset (``convert_deepjeb.py`` or ``convert_simjeb.py``).
    shapes : list[int]
        Design ids.
    scales : dict[str, dict[str, float]]
        Per-load-case scales from :func:`load_scales`.
    loads : tuple[str, ...], optional, default=LOADS
        Load cases.
    coordinate_scale_mm : float, optional, default=100.0
        Length unit of the model coordinates, in mm.
    token_voxel_mm : float, optional, default=2.0
        Voxel edge of the geometry context, in mm.
    """

    def __init__(
        self,
        root: str | Path,
        shapes: list[int],
        scales: dict[str, dict[str, float]],
        loads: tuple[str, ...] = LOADS,
        coordinate_scale_mm: float = 100.0,
        token_voxel_mm: float = 2.0,
    ) -> None:
        self.root, self.scales = Path(root), scales
        self.coordinate_scale_mm, self.token_voxel_mm = (
            coordinate_scale_mm,
            token_voxel_mm,
        )
        self.items = [(int(s), load) for s in shapes for load in loads]
        missing = [
            sample_path(self.root, s, load)
            for s, load in self.items
            if not sample_path(self.root, s, load).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} samples missing under {self.root}, e.g. {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shape, load = self.items[index]
        domain = DomainMesh.load(sample_path(self.root, shape, load))
        return sample_inputs(
            domain, self.scales, self.coordinate_scale_mm, self.token_voxel_mm
        )


def model_inputs(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    r"""The keyword arguments of :class:`~physicsnemo.models.geotransolver.GeoTransolver` from a batch."""
    keys = ("local_embedding", "local_positions", "geometry", "global_embedding")
    return {k: batch[k].to(device, non_blocking=True) for k in keys}
