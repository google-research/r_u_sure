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

"""Shared decision types for region DAGs.

This module contains a shared set of decision keys and values that can be
used to build DAGs over the same search space.

There are two types of DAG builder that use this shared search space:
- The uncertainty-aware edit DAG builder uses this space to parameterize
  its edit distance calculation, which assigns lower penalties for edits inside
  flagged regions (and interprets the extracted regions as UNSURE).
- The constraint DAG builder enforces the constraint that regions must start
  and end at the same level in the input tree. This can be combined with either
  of the two above DAGs to construct a well-formed set of regions.
"""
from __future__ import annotations

import dataclasses
import enum
import html
from typing import Iterable, NamedTuple

from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.numba_helpers import register_enum_hash
from r_u_sure.tree_structure import packed_sequence_nodes

Cost = gated_state_dag.Cost

PackedSequenceNodeStorage = packed_sequence_nodes.PackedSequenceNodeStorage
PackedSequenceNodeID = packed_sequence_nodes.PackedSequenceNodeID
PSNCategory = packed_sequence_nodes.PackedSequenceNodeCategory


################################################################################
# Decision types
################################################################################

# These decisions are shared between different DAG builders.


class DecisionCategory(enum.IntEnum):  # must be sortable in Numba code
  """A type of decision we can make."""

  REGION_SHOULD_START = enum.auto()
  REGION_SHOULD_END = enum.auto()
  NODE_IN_REGION = enum.auto()
  SHOULD_EARLY_EXIT = enum.auto()

  # Numba-compatible __hash__ implementation.
  __hash__ = register_enum_hash.jitable_enum_hash

  def shortname(self) -> str:
    if self == DecisionCategory.REGION_SHOULD_START:
      return "start"
    elif self == DecisionCategory.REGION_SHOULD_END:
      return "end"
    elif self == DecisionCategory.NODE_IN_REGION:
      return "in"
    elif self == DecisionCategory.SHOULD_EARLY_EXIT:
      return "exit"
    else:
      raise ValueError(self)


class DecisionKey(NamedTuple):
  """The location and type of a decision. Ordered by path."""

  prototype_preorder_index: int
  category: DecisionCategory

  def summarize_assignment(self, value) -> str:
    """Summarizes an assignment to this decision."""
    return f"{self.category.shortname()}: {value.shortname()}"


class DecisionValue(enum.IntEnum):
  """Values for our decisions. Must be orderable (arbitrarily)."""

  # Boolean decisions. We order "false" before "true" so that, when we break
  # ties, we tend to pick "false" (e.g. don't put nodes in low confidence
  # regions and don't early exit)
  FALSE = enum.auto()
  TRUE = enum.auto()
  # Invalid decisions
  NOT_APPLICABLE = enum.auto()

  # Numba-compatible __hash__ implementation.
  __hash__ = register_enum_hash.jitable_enum_hash

  def shortname(self) -> str:
    """Summarizes a value."""
    if self == DecisionValue.TRUE:
      return "true"
    elif self == DecisionValue.FALSE:
      return "false"
    elif self == DecisionValue.NOT_APPLICABLE:
      return "N/A"
    else:
      raise ValueError(self)


################################################################################
# Solution postprocessing
################################################################################


@dataclasses.dataclass
class TokenWithRegionMembership:
  """A string token with region membership."""

  token_or_decoration: str
  in_annotated_region: bool


def extract_sequence_with_regions(
    prototype: PackedSequenceNodeStorage,
    assignments: dict[DecisionKey, DecisionValue],
    empty_region_marker: str = "",
) -> Iterable[TokenWithRegionMembership]:
  """Yields a sequence of tokens tagged with their confidence.

  Args:
    prototype: The prototype used to generate these assignments.
    assignments: The sequence of variable assignments specifying the regions.
      This must include assignments of "unused" variables to None; these
      assignments appear in the wrapped version of the PathGraph but are not
      included in the AnnotatedDirectedAcyclicGraph itself.
    empty_region_marker: Marker to use for empty annotated regions, instead of
      their raw text contents (which is just the empty string).

  Yields:
    Tokens in the sequence, tagged with region membership.
  """
  in_region = False
  empty_region = False
  for node_id in prototype.preorder_traversal:
    if node_id.category == PSNCategory.TEXT_DECORATION_NODE:
      node = prototype.text_decoration_nodes[node_id.index_in_category]
      yield TokenWithRegionMembership(node.text_contents, in_region)
      empty_region = False
    elif node_id.category == PSNCategory.TEXT_TOKEN_NODE:
      node = prototype.text_token_nodes[node_id.index_in_category]
      yield TokenWithRegionMembership(node.text_contents, in_region)
      empty_region = False
    elif node_id.category == PSNCategory.REGION_START_NODE:
      should_start = assignments[
          DecisionKey(
              node_id.preorder_index,
              DecisionCategory.REGION_SHOULD_START,
          )
      ]
      if should_start == DecisionValue.TRUE:
        in_region = True
        empty_region = True
    elif node_id.category == PSNCategory.REGION_END_NODE:
      should_end = assignments[
          DecisionKey(
              node_id.preorder_index, DecisionCategory.REGION_SHOULD_END
          )
      ]
      if should_end == DecisionValue.TRUE:
        assert in_region
        if empty_region:
          yield TokenWithRegionMembership(empty_region_marker, True)
        in_region = False
        empty_region = False
    elif node_id.category == PSNCategory.EARLY_EXIT_NODE:
      should_early_exit = assignments[
          DecisionKey(
              node_id.preorder_index, DecisionCategory.SHOULD_EARLY_EXIT
          )
      ]
      if should_early_exit == DecisionValue.TRUE:
        break
    elif node_id.category == PSNCategory.GROUP_NODE:
      # Group node children will be visited next regardless.
      pass
    else:
      raise ValueError(node_id)


def render_regions_to_html(
    prototype: PackedSequenceNodeStorage,
    assignments: dict[DecisionKey, DecisionValue],
    region_text_color: str = "darkorange",
    region_border_color: str = "orange",
    region_background_color: str = "yellow",
) -> str:
  """Renders annotated regions from a set of assignments.

  Args:
    prototype: The prototype used to generate these assignments.
    assignments: The sequence of variable assignments specifying the regions.
      This must include assignments of "unused" variables to None; these
      assignments appear in the wrapped version of the PathGraph but are not
      included in the AnnotatedDirectedAcyclicGraph itself.
    region_text_color: Color of text in annotated regions.
    region_border_color: Border to draw around annotated regions.
    region_background_color: Background color for annotated regions.

  Returns:
    The annotated sequence, represented as an HTML string.
  """
  parts = []
  parts.append(
      f"""<style>
        .regions_root {{
            white-space: pre;
            font-family: monospace;
            color: black;
            font-weight: bold;
        }}
        .regions_root .in_region {{
            color: {region_text_color};
            border: solid 1px {region_border_color};
            border-radius: 3px;
            background-color: {region_background_color};
        }}
      </style><span class="regions_root">"""
  )
  was_in_region = False
  for token_with_confidence in extract_sequence_with_regions(
      prototype,
      assignments,
      empty_region_marker="︙",
  ):
    if token_with_confidence.in_annotated_region and not was_in_region:
      parts.append('<span class="in_region">')
    if not token_with_confidence.in_annotated_region and was_in_region:
      parts.append("</span>")

    parts.append(html.escape(token_with_confidence.token_or_decoration))
    was_in_region = token_with_confidence.in_annotated_region

  if was_in_region:
    parts.append("</span>")

  parts.append("</span>")
  return "".join(parts)
