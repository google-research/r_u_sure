# coding=utf-8
# Copyright 2023 The R-U-SURE Authors.
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

"""Types of sequence nodes for sequence-based idiom completion algorithm.

The sequence node representation is intended as an extensible representation of
sequences for the idiom completion algorithms, with fewer assumptions than the
old MatchableTree representation. Goals:
- Express trees as a generalization of sequences, instead of requiring sequences
  to be interpreted as special cases of trees. This makes it possible to explore
  additional cost functions that depend on sequences only.
- Consolidate complexity by moving as much as possible into preprocessing logic,
  so that the fundamental algorithms can focus on supporting the generic
  representation in an easy-to-understand way.

A sequence of SequenceNode objects will include information about both the
sequence of textual characters in a suggestion, and information about where
uncertainty regions can be inserted into a prototype suggestion to construct a
specific template.
"""
from __future__ import annotations

import dataclasses
from typing import Any, List, Union


@dataclasses.dataclass
class TextDecorationNode:
  """Node representing a span of text that does not participate in matching.

  This node can be used to represent text that should be output when rendering
  a template, but doesn't need to be the same across suggestions. For instance,
  whitespace.

  Attributes:
    text_contents: Contents of the node.
  """
  text_contents: str

  def size_for_utility(self) -> float:
    """Returns how much this node is "worth" for utility computations."""
    # By default we still count characters inside decorations.
    # We may want to override this in some cases.
    return len(self.text_contents)


@dataclasses.dataclass
class TextTokenNode:
  """An individual token that participates in matching.

  This node represents text that should participate in matching. Two nodes match
  if they have the same contents.

  Attributes:
    text_contents: Contents of the node.
    match_type: Optional type of token. Two token nodes will only match if they
      are the same type.
  """
  text_contents: str
  match_type: Any = None

  def size_for_utility(self) -> float:
    """Returns how much this node is "worth" for utility computations."""
    return len(self.text_contents)


@dataclasses.dataclass
class RegionStartNode:
  """A location in the sequence where an uncertainty region can start."""


@dataclasses.dataclass
class RegionEndNode:
  """A location in the sequence where an uncertainty region can end."""


@dataclasses.dataclass
class EarlyExitNode:
  """A location where we can end the suggestion early."""


@dataclasses.dataclass
class GroupNode:
  """A parent node containing a group of children.

  Group nodes allow encoding of tree structures, and interact with both
  uncertainty region generation and matching:
  - An uncertainty region will either be entirely inside a group, or entirely
    outside; regions cannot start outside a group but end inside one, or vice
    versa.
  - Two child nodes can only match if their parent group nodes match.

  Attributes:
    children: List of children of the node.
    match_type: Optional type of group. Two group nodes will only match if they
      are the same type.
  """
  children: List[SequenceNode]
  match_type: Any = None

  def size_for_utility(self) -> float:
    """Returns how much this node is "worth" for utility computations."""
    # By default, group nodes are not worth anything; all utility comes from
    # children. This can be overridden.
    return 0.0


SequenceNode = Union[TextDecorationNode, TextTokenNode, RegionStartNode,
                     RegionEndNode, EarlyExitNode, GroupNode]


@dataclasses.dataclass(frozen=True, order=True)
class NestedNodePath:
  """A node position in a nested sequence.

  Node paths are orderable, with the path to a node preceding the path to any
  of its children.

  Attributes:
    path: Sequence of indices required to get to this node from the top-level
      node list.
  """
  path: tuple[int, ...]

  @classmethod
  def root(cls) -> NestedNodePath:
    """Returns the empty path."""
    return NestedNodePath(())

  def extended_with(self, index: int) -> NestedNodePath:
    """Returns a copy of this position with an extra index appended."""
    return NestedNodePath(path=self.path + (index,))
