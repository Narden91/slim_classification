# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""SLIM_GSGP crossover operators.

SLIM individuals are represented as an ordered list of *blocks* (trees). Unlike
classic GP crossover (subtree swapping), SLIM crossover should operate on blocks
to preserve the semantic-building nature of the representation.

This module provides two crossover alternatives:

1) One-point *block swap* crossover:
   offspring = prefix(parent1) + suffix(parent2)

2) Uniform *block mix* crossover:
   offspring blocks are chosen position-wise from either parent.

Both operators are designed to be fast and to work even when
``reconstruct=False`` by recombining the cached per-block semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional

import torch

from slim_gsgp.algorithms.SLIM_GSGP.representations.individual import Individual


@dataclass(frozen=True, slots=True)
class BlockView:
    """A lightweight view of an individual's blocks and cached attributes."""

    blocks: Optional[List]
    train_semantics: torch.Tensor
    test_semantics: Optional[torch.Tensor]
    nodes_collection: List[int]
    depth_collection: List[int]


def _build_offspring(
    *,
    blocks: Optional[List],
    train_semantics: torch.Tensor,
    test_semantics: Optional[torch.Tensor],
    nodes_collection: List[int],
    depth_collection: List[int],
    reconstruct: bool,
) -> Individual:
    """Build an offspring Individual from chosen blocks + cached semantics."""

    offspring = Individual(
        collection=blocks if reconstruct else None,
        train_semantics=train_semantics,
        test_semantics=test_semantics,
        reconstruct=reconstruct,
    )

    offspring.size = len(nodes_collection)
    offspring.nodes_collection = list(nodes_collection)
    offspring.nodes_count = sum(offspring.nodes_collection) + (offspring.size - 1)

    offspring.depth_collection = list(depth_collection)
    offspring.depth = (
        max(
            [
                depth - (i - 1) if i != 0 else depth
                for i, depth in enumerate(offspring.depth_collection)
            ]
        )
        + (offspring.size - 1)
    )

    return offspring


def _as_block_view(parent: Individual) -> BlockView:
    if not hasattr(parent, "train_semantics") or parent.train_semantics is None:
        raise ValueError("Parent must have cached train_semantics for crossover")
    if not hasattr(parent, "nodes_collection") or not hasattr(parent, "depth_collection"):
        raise ValueError("Parent must have nodes_collection/depth_collection")

    return BlockView(
        blocks=getattr(parent, "collection", None),
        train_semantics=parent.train_semantics,
        test_semantics=parent.test_semantics,
        nodes_collection=list(parent.nodes_collection),
        depth_collection=list(parent.depth_collection),
    )


def one_point_block_crossover(
    parent1: Individual,
    parent2: Individual,
    *,
    reconstruct: bool = True,
) -> Individual:
    """One-point block swap crossover.

    Offspring keeps a prefix of ``parent1`` and concatenates a suffix of
    ``parent2``.

    Notes
    -----
    - Always produces at least one block.
    - Cut points include the possibility of taking all blocks from one parent.
    """

    p1 = _as_block_view(parent1)
    p2 = _as_block_view(parent2)

    size1 = len(p1.nodes_collection)
    size2 = len(p2.nodes_collection)

    # Choose how many blocks to keep from parent1 (at least 1)
    cut1 = 1 if size1 == 1 else random.randint(1, size1)
    # Choose where to start taking blocks from parent2 (can be size2 -> empty suffix)
    cut2 = 1 if size2 == 1 else random.randint(1, size2)

    # Selected block indices
    p1_idx = list(range(0, cut1))
    p2_idx = list(range(cut2, size2))

    if reconstruct:
        if p1.blocks is None or p2.blocks is None:
            raise ValueError("reconstruct=True requires parents to have collections")
        blocks = [p1.blocks[i] for i in p1_idx] + [p2.blocks[i] for i in p2_idx]
    else:
        blocks = None

    target_device = (
        p1.train_semantics.device
        if isinstance(p1.train_semantics, torch.Tensor)
        else (p2.train_semantics.device if isinstance(p2.train_semantics, torch.Tensor) else None)
    )

    train_parts = [p1.train_semantics[i] for i in p1_idx] + [p2.train_semantics[i] for i in p2_idx]
    if target_device is not None:
        train_parts = [t.to(target_device) if t.device != target_device else t for t in train_parts]
    train_semantics = torch.stack(train_parts)

    test_semantics: Optional[torch.Tensor]
    if p1.test_semantics is not None and p2.test_semantics is not None:
        test_parts = [p1.test_semantics[i] for i in p1_idx] + [p2.test_semantics[i] for i in p2_idx]
        if target_device is not None:
            test_parts = [t.to(target_device) if t.device != target_device else t for t in test_parts]
        test_semantics = torch.stack(test_parts)
    else:
        test_semantics = None

    nodes_collection = [p1.nodes_collection[i] for i in p1_idx] + [
        p2.nodes_collection[i] for i in p2_idx
    ]
    depth_collection = [p1.depth_collection[i] for i in p1_idx] + [
        p2.depth_collection[i] for i in p2_idx
    ]

    return _build_offspring(
        blocks=blocks,
        train_semantics=train_semantics,
        test_semantics=test_semantics,
        nodes_collection=nodes_collection,
        depth_collection=depth_collection,
        reconstruct=reconstruct,
    )


def uniform_block_crossover(
    parent1: Individual,
    parent2: Individual,
    *,
    reconstruct: bool = True,
    p_choose_parent1: float = 0.5,
) -> Individual:
    """Uniform block mix crossover.

    Builds an offspring by selecting blocks position-wise from either parent.
    If one parent is shorter, remaining blocks are taken from the longer parent.

    Parameters
    ----------
    p_choose_parent1:
        Probability of selecting the block from ``parent1`` when both parents
        have a block at the current position.
    """

    if not 0.0 <= p_choose_parent1 <= 1.0:
        raise ValueError("p_choose_parent1 must be in [0, 1]")

    p1 = _as_block_view(parent1)
    p2 = _as_block_view(parent2)

    size1 = len(p1.nodes_collection)
    size2 = len(p2.nodes_collection)

    target_size = max(size1, size2)

    chosen_from_p1: List[bool] = []
    for i in range(target_size):
        if i >= size1:
            chosen_from_p1.append(False)
        elif i >= size2:
            chosen_from_p1.append(True)
        else:
            chosen_from_p1.append(random.random() < p_choose_parent1)

    if reconstruct:
        if p1.blocks is None or p2.blocks is None:
            raise ValueError("reconstruct=True requires parents to have collections")
        blocks = [
            (p1.blocks[i] if chosen_from_p1[i] else p2.blocks[i])
            for i in range(target_size)
        ]
    else:
        blocks = None

    target_device = (
        p1.train_semantics.device
        if isinstance(p1.train_semantics, torch.Tensor)
        else (p2.train_semantics.device if isinstance(p2.train_semantics, torch.Tensor) else None)
    )

    train_parts = [
        (p1.train_semantics[i] if chosen_from_p1[i] else p2.train_semantics[i])
        for i in range(target_size)
    ]
    if target_device is not None:
        train_parts = [t.to(target_device) if t.device != target_device else t for t in train_parts]
    train_semantics = torch.stack(train_parts)

    test_semantics: Optional[torch.Tensor]
    if p1.test_semantics is not None and p2.test_semantics is not None:
        test_parts = [
            (p1.test_semantics[i] if chosen_from_p1[i] else p2.test_semantics[i])
            for i in range(target_size)
        ]
        if target_device is not None:
            test_parts = [t.to(target_device) if t.device != target_device else t for t in test_parts]
        test_semantics = torch.stack(test_parts)
    else:
        test_semantics = None

    nodes_collection = [
        (p1.nodes_collection[i] if chosen_from_p1[i] else p2.nodes_collection[i])
        for i in range(target_size)
    ]
    depth_collection = [
        (p1.depth_collection[i] if chosen_from_p1[i] else p2.depth_collection[i])
        for i in range(target_size)
    ]

    return _build_offspring(
        blocks=blocks,
        train_semantics=train_semantics,
        test_semantics=test_semantics,
        nodes_collection=nodes_collection,
        depth_collection=depth_collection,
        reconstruct=reconstruct,
    )
