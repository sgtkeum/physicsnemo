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

"""Tests of the deck reader, the sample layout, the datapipe and the latent Mahalanobis distance."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bracket_io import boundary_triangles, sample_path, save_design  # noqa: E402
from datapipe import BracketDataset, voxel_first  # noqa: E402
from nastran import LOADS, number, parse_deck  # noqa: E402
from ood import mahalanobis  # noqa: E402

from physicsnemo.mesh import DomainMesh  # noqa: E402

SCALES = {
    "displacement": dict.fromkeys(LOADS, 0.1),
    "von_mises": dict.fromkeys(LOADS, 50.0),
}


def card(*fields):
    return "".join(f"{f:<8}" for f in fields) + "\n"


@pytest.fixture
def deck(tmp_path):
    """A one-tetrahedron deck: nodes 1-3 clamped through an RBE2, node 4 loaded through an RBE3."""
    text = [
        "SUBCASE 1\n  LABEL = vertical\n  SPC = 1\n  LOAD = 11\n",
        "SUBCASE 2\n  LABEL = horizontal\n  SPC = 1\n  LOAD = 12\n",
        "SUBCASE 3\n  LABEL = diagonal\n  SPC = 1\n  LOAD = 13\n",
        "SUBCASE 4\n  LABEL = torsion\n  SPC = 1\n  LOAD = 14\n",
        "BEGIN BULK\n",
    ]
    for node, xyz in {
        1: (0, 0, 0),
        2: (1, 0, 0),
        3: (0, 1, 0),
        4: (0, 0, 1),
        100: (0.3, 0.3, 0),
        200: (0, 0, 2),
    }.items():
        text.append(card("GRID", node, "", *[f"{v:.1f}" for v in xyz]))
    text += [
        card("CTETRA", 1, 1, 1, 2, 3, 4),
        card("PSOLID", 1, 1),
        card("MAT1", 1, "2.1+5", "", "0.3"),
        card("RBE2", 10, 100, "123456", 1, 2, 3),
        card("RBE3", 20, "", 200, "123456", "1.0", "123", 4),
        card("SPC", 1, 100, "123456", "0.0"),
    ]
    for sid, vector in ((11, (0, 0, -1)), (12, (0, 1, 0)), (13, (0, 1, -1))):
        text.append(card("FORCE", sid, 200, "", "1000.", *[f"{v:.1f}" for v in vector]))
    text += [card("MOMENT", 14, 200, "", "500.", "1.0", "0.0", "0.0"), "ENDDATA\n"]
    path = tmp_path / "bracket.fem"
    path.write_text("".join(text))
    return path


def test_number_shorthand():
    assert number("1.5-3") == pytest.approx(1.5e-3)
    assert number("2.1+5") == pytest.approx(2.1e5)


def test_parse_deck(deck):
    d = parse_deck(deck)
    assert (
        d["tets"].shape == (1, 4)
        and d["supports"] == {1, 2, 3}
        and d["loaded"] == {4}
        and d["references"] == {100, 200}
    )
    assert d["young_modulus"] == pytest.approx(2.1e5) and d[
        "poisson_ratio"
    ] == pytest.approx(0.3)
    assert d["conditions"]["tor"]["type"] == "MOMENT" and np.allclose(
        d["conditions"]["ver"]["vector"], [0, 0, -1000]
    )
    assert np.allclose(d["conditions"]["ver"]["reference_position"], [0, 0, 2])


def test_boundary_triangles():
    one = np.array([[0, 1, 2, 3]])
    assert len(boundary_triangles(one)) == 4
    two = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])  # the shared face is interior
    assert len(boundary_triangles(two)) == 6
    tet10 = np.arange(10)[None]
    triangles = boundary_triangles(tet10)
    assert len(triangles) == 16 and set(np.unique(triangles)) == set(range(10))


def test_voxel_first():
    points = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [1.2, 0.0, 0.0], [0.05, 0.0, 0.0]]
    )
    assert voxel_first(points, 1.0).tolist() == [0, 2]


def test_sample_roundtrip_and_inputs(deck, tmp_path):
    d = parse_deck(deck)
    points = np.array([d["grids"][i] for i in (1, 2, 3, 4)], dtype=np.float64) * 10.0
    region = np.array([2, 2, 2, 3])
    rng = np.random.default_rng(0)
    fields = {
        load: {
            "displacement": rng.normal(size=(4, 3)),
            "von_mises": rng.uniform(1, 2, size=4),
        }
        for load in LOADS
    }
    save_design(
        tmp_path / "data", 7, points, np.array([[0, 1, 2, 3]]), region, fields, d
    )
    domain = DomainMesh.load(sample_path(tmp_path / "data", 7, "dia"))
    assert int(domain.global_data["load_id"]) == 2 and domain.boundaries[
        "surface"
    ].cells.shape == (4, 3)
    sample = BracketDataset(
        tmp_path / "data", [7], SCALES, coordinate_scale_mm=10.0, token_voxel_mm=1.0
    )[2]
    assert sample["local_embedding"].shape == (4, 7) and sample[
        "global_embedding"
    ].tolist() == [[0, 0, 1, 0]]
    assert torch.allclose(
        sample["local_positions"], torch.tensor(points / 10.0, dtype=torch.float32)
    )
    assert torch.allclose(
        sample["target"] * sample["scale"],
        torch.tensor(
            np.c_[fields["dia"]["displacement"], fields["dia"]["von_mises"]],
            dtype=torch.float32,
        ),
        atol=1e-6,
    )
    assert sample["geometry"].shape[0] == 4


def test_mahalanobis_matches_definition():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(2000, 3)) @ np.diag([1.0, 2.0, 0.5])
    query = rng.normal(size=(5, 3))
    expected = np.einsum(
        "ij,jk,ik->i",
        query - reference.mean(0),
        np.linalg.inv(np.cov(reference.T)),
        query - reference.mean(0),
    )
    assert np.allclose(mahalanobis(reference)(query), expected, rtol=0.02)
