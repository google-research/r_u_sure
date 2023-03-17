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

"""Uncertainty-aware edit distance DAG builder.

This cost function is based on a two-tiered edit distance calculation. The goal
is to add "distrust regions" around parts of a suggestion, which decreases the
utility of those regions, but also decreases the penalty for incorrect parts.
This can then be used to identify the parts of the suggestion that are most
likely to require edits.

The edit distance calculation consists of a few types of reward and penalty,
which depend on whether we are in a high-confidence or low-confidence region:
1. The reward (or penalty) for correctly-predicted tokens
2. The penalty for deleting incorrect tokens
3. The penalty for inserting each missing token (perhaps to replace deleted
   ones)
4. An extra penalty for realizing we need to edit some number of tokens at a
   particular location: one for where we start editing, and one for where we
   stop editing.

Note that cost 3 must be the same in either high or low confidence
regions, since the benefit of flagging a possible insert shouldn't scale with
the size of the inserted code. When the two differ, this can cause confusing
spurious low-confidence regions around unrelated tokens, just to help with
editing adjacent tokens. Additionally, if these are different,
then it may be necessary to search for where to insert within a large set of
deleted tokens, whereas right now insertions are always handled after deletions.

The "Edit DAG" is based on a matching between the prototype and a target node,
and is responsible for computing the edit-distance between the prototype and
target under a particular set of confidence regions. This DAG ensures that
matching nodes must match recursively, so that subtrees are either matched
together or inserted/deleted, not inlined into their parents. This DAG does NOT
ensure that high and low confidence regions are valid subexpression spans; that
is handled by the "Constraint DAG" .

As an optimization / simplification, we process consecutive deletions before
insertions at the same location, and we treat region start/end nodes and
early exit nodes as if they were deletions. As a consequence, we can
aggressively prune the graphs for reachability after construction. One caveat
is that, if we choose to early exit, we will never process any subsequent
insertions; this should be the correct thing to do anyway as long as the
insertion cost is nonnegative.

NOTE: Currently, the fixed edit costs are determined by the location
in the code where the edit starts. Some edits might start in a low-confidence
region but end in a high-confidence region (e.g. if you delete more code than
was suggested). These edits will be treated asymmetrically from edits that
start in a high-confidence region but end in a low-confidence region. However,
it's a bit of a hassle to make it symmetric, since we asymmetrically process
all prototype-only nodes (including low-confidence start and stop) before
handling insertions, and only exit the editing state after handling insertions.
So we might end up finishing the low-confidence region before deciding we are
done editing.
Hopefully this is a minor-enough issue to not matter in practice, since ideally
few edits will span across high and low confidence regions together.
"""
from __future__ import annotations

import collections
import dataclasses
import enum
import html
import textwrap
from typing import Any, Callable, Iterable, NamedTuple, Optional

import numba
from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.decision_diagrams import packed_dags
from r_u_sure.edit_distance_utility import region_decisions
from r_u_sure.numba_helpers import numba_type_util
from r_u_sure.numba_helpers import register_enum_hash
from r_u_sure.rendering import dag_annotator
from r_u_sure.rendering import rendering
from r_u_sure.rendering import svg_renderer
from r_u_sure.tree_structure import packed_sequence_nodes
from r_u_sure.tree_structure import sequence_nodes

SequenceNode = sequence_nodes.SequenceNode

PackedSequenceNodeStorage = packed_sequence_nodes.PackedSequenceNodeStorage
PackedSequenceNodeID = packed_sequence_nodes.PackedSequenceNodeID
PSNCategory = packed_sequence_nodes.PackedSequenceNodeCategory

SharedVariableAssignment = gated_state_dag.SharedVariableAssignment
Edge = gated_state_dag.Edge
Cost = gated_state_dag.Cost

DecisionCategory = region_decisions.DecisionCategory
DecisionKey = region_decisions.DecisionKey
DecisionValue = region_decisions.DecisionValue

ROOT_PREORDER_INDEX = packed_sequence_nodes.ROOT_SEQUENCE_PREORDER_INDEX
NO_PREORDER_INDEX = packed_sequence_nodes.NO_PREORDER_INDEX
INVALID_NODE_ID = packed_sequence_nodes.INVALID_NODE_ID


@dataclasses.dataclass
class EditDagRenderConfig:
  """Parameters for rendering the edit DAG."""

  font_size: float = 12
  state_width: float = 170
  state_height: float = 20
  gap_horizontal: float = 130
  gap_vertical: float = 12 * 5
  state_separation: float = 12 * 3


################################################################################
# Cost-function configuration
################################################################################


class UtilitiesAndCostsForNode(NamedTuple):
  """Collection of utilities/costs for a node.

  Attributes:
    high_confidence_match_utility: Utility of correctly predicting this node in
      a high-confidence region.
    high_confidence_delete_cost: Cost of deleting this node inside a
      high-confidence region.
    low_confidence_match_utility: Utility of correctly predicting this node in a
      low-confidence (distrusted) region.
    low_confidence_delete_cost: Cost of deleting this node inside a
      low-confidence (distrusted) region.
    insert_cost: Cost of inserting this node. Must be the same in high and low
      confidence regions.
  """

  high_confidence_match_utility: Cost
  high_confidence_delete_cost: Cost
  low_confidence_match_utility: Cost
  low_confidence_delete_cost: Cost
  insert_cost: Cost


class TrustRegionUtilityParameters(NamedTuple):
  """Parameters for the distrust-region utility family.

  Attributes:
    token_node_utility_fn: Function computing utilities for a token node.
    low_confidence_region_cost: Extra cost for each low confidence region; this
      can be used to encourage having fewer such regions.
    high_confidence_start_editing_cost: Cost for starting to edit nodes in a
      high-confidence region, independent of the number of nodes edited.
    low_confidence_start_editing_cost: Cost for starting to edit nodes in a
      low-confidence region, independent of the number of nodes edited.
  """

  token_node_utility_fn: Callable[
      [packed_sequence_nodes.PackedTextTokenNode], UtilitiesAndCostsForNode
  ]
  low_confidence_region_cost: Cost
  high_confidence_start_editing_cost: Cost
  low_confidence_start_editing_cost: Cost


def make_character_count_cost_config(
    high_confidence_match_utility_per_char: Cost,
    high_confidence_delete_cost_per_char: Cost,
    low_confidence_match_utility_per_char: Cost,
    low_confidence_delete_cost_per_char: Cost,
    insert_cost_per_char: Cost,
    low_confidence_region_cost: Cost,
    high_confidence_start_editing_cost: Cost,
    low_confidence_start_editing_cost: Cost,
) -> TrustRegionUtilityParameters:
  """Constructs a cost configuration based on character counts.

  Args:
    high_confidence_match_utility_per_char: Per-character scaling term for
      high_confidence_match_utility.
    high_confidence_delete_cost_per_char: Per-character scaling term for
      high_confidence_delete_cost.
    low_confidence_match_utility_per_char: Per-character scaling term for
      low_confidence_match_utility.
    low_confidence_delete_cost_per_char: Per-character scaling term for
      low_confidence_delete_cost.
    insert_cost_per_char: Per-character scaling term for insert_cost.
    low_confidence_region_cost: Extra penalty for each low confidence region.
    high_confidence_start_editing_cost: Cost for starting to insert nodes in a
      high-confidence region, independent of the number of nodes inserted.
    low_confidence_start_editing_cost: Cost for starting to insert nodes in a
      low-confidence region, independent of the number of nodes inserted.

  Returns:
    A configuration for the utility graph. All functions are registered as
    numba jit-able.
  """

  @numba.extending.register_jitable
  def token_node_utility_fn(
      token: packed_sequence_nodes.PackedTextTokenNode,
  ) -> UtilitiesAndCostsForNode:
    length = len(token.text_contents)
    return UtilitiesAndCostsForNode(
        high_confidence_match_utility=(
            high_confidence_match_utility_per_char * length
        ),
        high_confidence_delete_cost=(
            high_confidence_delete_cost_per_char * length
        ),
        low_confidence_match_utility=(
            low_confidence_match_utility_per_char * length
        ),
        low_confidence_delete_cost=(
            low_confidence_delete_cost_per_char * length
        ),
        insert_cost=(insert_cost_per_char * length),
    )

  return TrustRegionUtilityParameters(
      token_node_utility_fn=token_node_utility_fn,
      low_confidence_region_cost=low_confidence_region_cost,
      high_confidence_start_editing_cost=high_confidence_start_editing_cost,
      low_confidence_start_editing_cost=low_confidence_start_editing_cost,
  )


################################################################################
# State types
################################################################################


class EditDagStateAlignment(enum.Enum):
  """Categories for states in the edit DAG."""

  # We are in the process of matching two (sub)sequences at the same depth.
  # We can freely insert, delete, or match compatible nodes.
  # To reduce redundant paths, we force deletions and prototype-only control
  # nodes to precede insertions and target-only decorations.

  # Advance past a prototype-only node (decoration, early exit, region
  # start or region end) and stay in PROCESS_PROTOTYPE, or transition
  # to MATCH, or transition to MAY_DELETE and pay a penalty.
  PROCESS_PROTOTYPE = enum.auto()
  # Delete a node (or advance in prototype) and stay in MAY_DELETE, or
  # transition to MAY_INSERT.
  MAY_DELETE = enum.auto()
  # Insert a node and stay in MAY_INSERT, or transition to MATCH.
  MAY_INSERT = enum.auto()
  # Match and transition back to MAY_DELETE, or insert decoration nodes and
  # stay in MATCH (since those are ignored), or end the subproblem.
  MATCH = enum.auto()

  # Note: this ordering implies that, even if we start editing inside a
  # low-confidence region, we might finished the low-confidence region while
  # inside the MAY_DELETE state, and thus be nominally in a high-confidence
  # region by the time we process MAY_INSERT. This is part of the reason we do
  # NOT have separate insertion costs for the different types of region
  # (in addition to the counterintuitive behavior of adding low-confidence
  # regions just to avoid rare but very long insertions).

  # We are processing a subtree in the prototype or target that is being
  # inserted or deleted in one go, without matching.
  RECURSIVELY_DELETING = enum.auto()
  RECURSIVELY_INSERTING = enum.auto()

  # One extra final state; we jump here for early exits.
  SPECIAL_FINAL_STATE = enum.auto()

  # Numba-compatible __hash__ implementation.
  __hash__ = register_enum_hash.jitable_enum_hash

  def shortname(self) -> str:
    """Summarizes a state category."""
    if self == EditDagStateAlignment.PROCESS_PROTOTYPE:
      return "Advance"
    if self == EditDagStateAlignment.MAY_DELETE:
      return "Delete"
    elif self == EditDagStateAlignment.MAY_INSERT:
      return "Insert"
    elif self == EditDagStateAlignment.MATCH:
      return "Match"
    elif self == EditDagStateAlignment.RECURSIVELY_DELETING:
      return "Delete (forced)"
    elif self == EditDagStateAlignment.RECURSIVELY_INSERTING:
      return "Insert (forced)"
    elif self == EditDagStateAlignment.SPECIAL_FINAL_STATE:
      return "Final State"
    else:
      raise ValueError(self)


# We start by advancing in the prototype and possibly allowing deletions...
MATCH_START_ALIGNMENT = EditDagStateAlignment.PROCESS_PROTOTYPE
# ... and we end after optionally allowing insertions.
MATCH_END_ALIGNMENT = EditDagStateAlignment.MATCH


class EditDagStateConfidence(enum.Enum):
  """Categories for confidence levels in the edit DAG."""

  HIGH_CONFIDENCE = enum.auto()
  LOW_CONFIDENCE = enum.auto()
  NOT_APPLICABLE = enum.auto()

  # Numba-compatible __hash__ implementation.
  __hash__ = register_enum_hash.jitable_enum_hash

  def shortname(self) -> str:
    """Summarizes a state category."""
    if self == EditDagStateConfidence.HIGH_CONFIDENCE:
      return "High"
    elif self == EditDagStateConfidence.LOW_CONFIDENCE:
      return "Low"
    elif self == EditDagStateConfidence.NOT_APPLICABLE:
      return "N/A"
    else:
      raise ValueError(self)


class EditDagSubproblem(NamedTuple):
  """Identifier for a subproblem.

  Primarily used for rendering. For simplicity we only render subgraphs that
  match, and don't bother rendering subproblem regions for inserted/deleted
  nodes.
  """

  prototype_preorder_index: int
  target_preorder_index: int


class EditDagState(NamedTuple):
  """A state in the DAG.

  Attributes:
    alignment: The type of alignment we have at this state.
    confidence: Whether we are in a high or low confidence region.
    prototype_sequence_preorder_index: Identifier for the prototype sequence we
      are processing.
    before_prototype_node: Position in that sequence we are currently
      processing.
    target_sequence_preorder_index: Identifier for the target sequence we are
      processing.
    before_target_node: Position in that sequence we are currently processing.
  """

  alignment: EditDagStateAlignment
  confidence: EditDagStateConfidence
  prototype_sequence_preorder_index: int
  before_prototype_node: int
  target_sequence_preorder_index: int
  before_target_node: int


FINAL_STATE = EditDagState(
    alignment=EditDagStateAlignment.SPECIAL_FINAL_STATE,
    confidence=EditDagStateConfidence.NOT_APPLICABLE,
    prototype_sequence_preorder_index=packed_sequence_nodes.NO_PREORDER_INDEX,
    before_prototype_node=-1,
    target_sequence_preorder_index=packed_sequence_nodes.NO_PREORDER_INDEX,
    before_target_node=-1,
)

################################################################################
# Edge info type
################################################################################


class EditAction(enum.Enum):
  """A type of action represented by a single edge."""

  KEEP = enum.auto()
  DELETE = enum.auto()
  INSERT = enum.auto()

  PROTOTYPE_DECORATION = enum.auto()
  TARGET_DECORATION = enum.auto()

  START_LOW_CONFIDENCE = enum.auto()
  START_EDITING = enum.auto()
  EARLY_EXIT = enum.auto()


class EditDagEdgeInfo(NamedTuple):
  """Edge info for the edit dag.

  Attributes:
    prototype_node_preorder_index: Preorder index of the prototype node
      corresponding to this edit, if applicable.
    target_node_preorder_index: Preorder index of the target node corresponding
      to this edit, if applicable.
    edit_action: Type of edit we are making.
    confidence: Confidence level of this edit.
  """

  prototype_node_preorder_index: int
  target_node_preorder_index: int
  edit_action: EditAction
  confidence: EditDagStateConfidence


EXAMPLE_EDGE_INFO = numba_type_util.PretendOptional(
    EditDagEdgeInfo(
        prototype_node_preorder_index=NO_PREORDER_INDEX,
        target_node_preorder_index=NO_PREORDER_INDEX,
        edit_action=EditAction.KEEP,
        confidence=EditDagStateConfidence.HIGH_CONFIDENCE,
    )
)

################################################################################
# Edit types and rendering info
################################################################################


class EditDagRenderData(NamedTuple):
  """Helper data passed around to help render the edit DAG.

  Attributes:
    subproblem_list: List of subproblems we encounter, for rendering purposes.
  """

  subproblem_list: list[EditDagSubproblem]


################################################################################
# Main logic
################################################################################

EXAMPLE_STATE = EditDagState(
    alignment=EditDagStateAlignment.RECURSIVELY_INSERTING,
    confidence=EditDagStateConfidence.LOW_CONFIDENCE,
    prototype_sequence_preorder_index=ROOT_PREORDER_INDEX,
    before_prototype_node=0,
    target_sequence_preorder_index=ROOT_PREORDER_INDEX,
    before_target_node=0,
)
EXAMPLE_VARIABLE_KEY = DecisionKey(
    prototype_preorder_index=ROOT_PREORDER_INDEX,
    category=DecisionCategory.REGION_SHOULD_START,
)
EXAMPLE_SUBPROBLEM = EditDagSubproblem(
    prototype_preorder_index=ROOT_PREORDER_INDEX,
    target_preorder_index=ROOT_PREORDER_INDEX,
)

STATE_NUMBA_TYPE = numba.typeof(EXAMPLE_STATE)
# pytype: disable=wrong-arg-types
EDGE_NUMBA_TYPE = numba.typeof(
    Edge(
        source=EXAMPLE_STATE,
        dest=EXAMPLE_STATE,
        cost=0.0,
        required_assignment=numba_type_util.PretendOptional(
            SharedVariableAssignment(
                key=EXAMPLE_VARIABLE_KEY,
                value=DecisionValue.NOT_APPLICABLE,
            )
        ),
        info=EXAMPLE_EDGE_INFO,
    )
)
# pytype: enable=wrong-arg-types
SUBPROBLEM_NUMBA_TYPE = numba.typeof(EXAMPLE_SUBPROBLEM)


@numba.extending.register_jitable(inline="always")
def node_in_low_confidence_value(
    confidence: EditDagStateConfidence,
) -> DecisionValue:
  """Map a confidence to a value for a NODE_IN_LOW_CONFIDENCE assignment."""
  if confidence == EditDagStateConfidence.LOW_CONFIDENCE:
    return DecisionValue.TRUE
  else:
    return DecisionValue.FALSE


def make_edit_dag_builder(
    parameters: TrustRegionUtilityParameters,
    with_numba: bool = False,
) -> Callable[..., tuple[gated_state_dag.CompleteStateDAG, EditDagRenderData],]:
  """Builds either Numba or pure python version of graph construction logic."""
  # Numba gets confused about the type of our parameters tuple, since it
  # contains functions. We can bypass this by just unpacking it here and
  # closing over all the values separately.
  low_confidence_region_cost = parameters.low_confidence_region_cost
  token_node_utility_fn = parameters.token_node_utility_fn
  high_confidence_start_editing_cost = (
      parameters.high_confidence_start_editing_cost
  )
  low_confidence_start_editing_cost = (
      parameters.low_confidence_start_editing_cost
  )

  maybe_jit = numba.njit if with_numba else lambda fn: fn

  # We will also be (ab)using ternary expressions (`X if Y else Z`) to add
  # metadata when called from pure Python. The ternaries are only used to
  # suspend evaluation of this metadata so that numba doesn't try to typecheck
  # it. Semantically, it's just a wrapper around a single value, and the other
  # value is always just None, so it's clearer to allow the ternary to be
  # multiple lines.

  # pylint: disable=g-long-ternary

  def construct_edit_dag(
      prototype: PackedSequenceNodeStorage,
      target: PackedSequenceNodeStorage,
  ) -> tuple[gated_state_dag.CompleteStateDAG, EditDagRenderData]:
    """Constructs the edit DAG.

    Args:
      prototype: Node sequence representing the prototype, which should include
        region nodes (but not early exit nodes).
      target: Node sequence representing a possible end state of the user's
        code, which should NOT include region nodes (or early exit nodes).

    Returns:
      A graph and a set of subproblems for use in rendering later.
    """
    # Construct the graph.
    initial_state = EditDagState(
        alignment=MATCH_START_ALIGNMENT,
        confidence=EditDagStateConfidence.HIGH_CONFIDENCE,
        prototype_sequence_preorder_index=ROOT_PREORDER_INDEX,
        before_prototype_node=0,
        target_sequence_preorder_index=ROOT_PREORDER_INDEX,
        before_target_node=0,
    )
    penultimate_state = EditDagState(
        alignment=MATCH_END_ALIGNMENT,
        confidence=EditDagStateConfidence.HIGH_CONFIDENCE,
        prototype_sequence_preorder_index=ROOT_PREORDER_INDEX,
        before_prototype_node=len(prototype.root_sequence),
        target_sequence_preorder_index=ROOT_PREORDER_INDEX,
        before_target_node=len(target.root_sequence),
    )

    if with_numba:
      render_data = EditDagRenderData(
          subproblem_list=numba.typed.List.empty_list(SUBPROBLEM_NUMBA_TYPE)
      )
      incomplete_graph = gated_state_dag.PartialStateDAG(
          initial_state=initial_state,
          edges=numba.typed.List.empty_list(EDGE_NUMBA_TYPE),
          seen_outgoing=numba.typed.Dict.empty(
              STATE_NUMBA_TYPE, STATE_NUMBA_TYPE
          ),
      )
    else:
      render_data = EditDagRenderData(subproblem_list=[])
      incomplete_graph = gated_state_dag.partial_state_dag_starting_from(
          initial_state
      )

    # Match the subsequences.
    build_matching_subproblem(
        prototype_storage=prototype,
        prototype_node_ids=prototype.root_sequence,
        prototype_sequence_preorder_index=ROOT_PREORDER_INDEX,
        target_storage=target,
        target_node_ids=target.root_sequence,
        target_sequence_preorder_index=ROOT_PREORDER_INDEX,
        incomplete_graph=incomplete_graph,
        render_data=render_data,
    )

    # If we didn't early exit, connect up to the final state.
    gated_state_dag.partial_state_dag_add_edge(
        incomplete_graph,
        Edge(source=penultimate_state, dest=FINAL_STATE, cost=0),
    )

    # Finish the graph.
    complete_dag = gated_state_dag.partial_state_dag_finish(
        incomplete_graph, FINAL_STATE
    )

    return complete_dag, render_data

  @maybe_jit
  def maybe_insert_node_or_skip_decoration_in_target(
      alignment_type: EditDagStateAlignment,
      target_node_id: PackedSequenceNodeID,
      target_storage: PackedSequenceNodeStorage,
      before_prototype_node: int,
      prototype_sequence_preorder_index: int,
      incomplete_graph: gated_state_dag.PartialStateDAG,
      allow_insert: bool,
  ):
    """Possibly inserts a node from the target, or skips a decoration.

    Args:
      alignment_type: The type of alignment to use. Generally either MAY_INSERT
        or RECURSIVELY_INSERTING depending on whether we are allowed to stop
        inserting. Can also be MATCH with `allow_insert=False`.
      target_node_id: The node we are considering inserting.
      target_storage: Storage for the target nodes.
      before_prototype_node: Location in the prototype (sub)sequence.
      prototype_sequence_preorder_index: Identifier for the prototype
        (sub)sequence.
      incomplete_graph: Graph that we should extend with this subgraph.
      allow_insert: If True, allows inserting groups and tokens. If False, only
        allows skipping past decoration nodes. (Inserting decoration nodes is
        allowed even if we haven't paid the edit penalty.)
    """
    if target_node_id.category == PSNCategory.INVALID:
      return

    position_in_parent_info = target_storage.parents_and_offsets_from_preorder[
        target_node_id.preorder_index
    ]
    target_position = position_in_parent_info.index_of_child_in_parent
    target_sequence_preorder_index = (
        position_in_parent_info.parent_preorder_index
    )

    # pylint: disable=cell-var-from-loop
    def relative_state_for_insert(
        confidence,
        advance_target=False,
    ):
      delta_j = 1 if advance_target else 0
      return EditDagState(
          alignment=alignment_type,
          confidence=confidence,
          before_prototype_node=before_prototype_node,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_target_node=target_position + delta_j,
          target_sequence_preorder_index=target_sequence_preorder_index,
      )

    # pylint: enable=cell-var-from-loop

    # Skip/insert decorations in the target.
    if target_node_id.category == PSNCategory.TEXT_DECORATION_NODE:
      target_node = target_storage.text_decoration_nodes[
          target_node_id.index_in_category
      ]
      for confidence in (
          EditDagStateConfidence.HIGH_CONFIDENCE,
          EditDagStateConfidence.LOW_CONFIDENCE,
      ):
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=relative_state_for_insert(confidence=confidence),
                dest=relative_state_for_insert(
                    confidence=confidence, advance_target=True
                ),
                cost=0,
                info=EditDagEdgeInfo(
                    prototype_node_preorder_index=NO_PREORDER_INDEX,
                    target_node_preorder_index=target_node_id.preorder_index,
                    edit_action=EditAction.TARGET_DECORATION,
                    confidence=confidence,
                ),
            ),
        )

    if allow_insert:
      # Insert tokens.
      if target_node_id.category == PSNCategory.TEXT_TOKEN_NODE:
        target_node = target_storage.text_token_nodes[
            target_node_id.index_in_category
        ]
        for confidence in (
            EditDagStateConfidence.HIGH_CONFIDENCE,
            EditDagStateConfidence.LOW_CONFIDENCE,
        ):
          node_utilities = token_node_utility_fn(target_node)
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state_for_insert(confidence=confidence),
                  dest=relative_state_for_insert(
                      confidence=confidence, advance_target=True
                  ),
                  cost=node_utilities.insert_cost,
                  info=EditDagEdgeInfo(
                      prototype_node_preorder_index=NO_PREORDER_INDEX,
                      target_node_preorder_index=target_node_id.preorder_index,
                      edit_action=EditAction.INSERT,
                      # Do not mark insertions with a confidence, since the
                      # span we are inserting into may cover multiple confidence
                      # levels.
                      confidence=EditDagStateConfidence.NOT_APPLICABLE,
                  ),
              ),
          )

      # Recursively insert groups. (Group nodes themselves don't have any
      # costs currently.)
      if target_node_id.category == PSNCategory.GROUP_NODE:
        target_node = target_storage.group_nodes[
            target_node_id.index_in_category
        ]
        # Connect up to subproblem.
        for confidence in (
            EditDagStateConfidence.HIGH_CONFIDENCE,
            EditDagStateConfidence.LOW_CONFIDENCE,
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state_for_insert(confidence=confidence),
                  dest=EditDagState(
                      alignment=EditDagStateAlignment.RECURSIVELY_INSERTING,
                      confidence=confidence,
                      # Stationary in the prototype.
                      prototype_sequence_preorder_index=(
                          prototype_sequence_preorder_index
                      ),
                      before_prototype_node=before_prototype_node,
                      # Inserting the group node's children from the target.
                      target_sequence_preorder_index=(
                          target_node_id.preorder_index
                      ),
                      before_target_node=0,
                  ),
                  cost=0,
              ),
          )
        # Generate recursive insert subproblem, to handle all the children.
        build_recursive_insert_subsequence(
            target_storage=target_storage,
            target_node_ids=target_node.children_ids,
            fixed_before_prototype_node=before_prototype_node,
            fixed_prototype_sequence_preorder_index=(
                prototype_sequence_preorder_index
            ),
            incomplete_graph=incomplete_graph,
        )
        # Connect back to this level.
        for confidence in (
            EditDagStateConfidence.HIGH_CONFIDENCE,
            EditDagStateConfidence.LOW_CONFIDENCE,
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=EditDagState(
                      alignment=EditDagStateAlignment.RECURSIVELY_INSERTING,
                      confidence=confidence,
                      # Stationary in the prototype.
                      prototype_sequence_preorder_index=(
                          prototype_sequence_preorder_index
                      ),
                      before_prototype_node=before_prototype_node,
                      # Inserting the group node's children from the target.
                      target_sequence_preorder_index=(
                          target_node_id.preorder_index
                      ),
                      before_target_node=len(target_node.children_ids),
                  ),
                  dest=relative_state_for_insert(
                      confidence=confidence,
                      advance_target=True,
                  ),
                  cost=0,
              ),
          )

  @maybe_jit
  def maybe_delete_node_or_advance_in_prototype(
      alignment_type: EditDagStateAlignment,
      prototype_node_id: PackedSequenceNodeID,
      prototype_storage: PackedSequenceNodeStorage,
      before_target_node: int,
      target_sequence_preorder_index: int,
      incomplete_graph: gated_state_dag.PartialStateDAG,
      allow_delete: bool,
  ):
    """Possibly deletes a node or advances in the prototype.

    This function handles all actions for the nodes in the prototype that do
    not require advancing in the target, which includes both deleting tokens
    and groups, skipping decorations, and changing confidence levels.

    Args:
      alignment_type: The type of alignment to use. If `allow_delete=True`, will
        be either MAY_DELETE or RECURSIVELY_DELETING depending on whether we are
        allowed to stop deleting. Can also be PROCESS_PROTOTYPE with
        `allow_delete=False`.
      prototype_node_id: The node we are considering deleting.
      prototype_storage: Storage for the prototype nodes.
      before_target_node: Location in the target (sub)sequence.
      target_sequence_preorder_index: Identifier for the target (sub)sequence.
      incomplete_graph: Graph that we should extend with this subgraph.
      allow_delete: Whether to allow deleting tokens and groups from the
        prototype. If False, only allows advancing past control nodes and
        skipping decorations.
    """
    if prototype_node_id.category == PSNCategory.INVALID:
      return

    position_in_parent_info = (
        prototype_storage.parents_and_offsets_from_preorder[
            prototype_node_id.preorder_index
        ]
    )
    prototype_position = position_in_parent_info.index_of_child_in_parent
    prototype_sequence_preorder_index = (
        position_in_parent_info.parent_preorder_index
    )

    # pylint: disable=cell-var-from-loop
    def relative_state_for_delete(
        confidence,
        advance_prototype=False,
    ):
      delta_i = 1 if advance_prototype else 0
      return EditDagState(
          alignment=alignment_type,
          confidence=confidence,
          before_prototype_node=prototype_position + delta_i,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_target_node=before_target_node,
          target_sequence_preorder_index=target_sequence_preorder_index,
      )

    # Skip/delete decorations in the prototype.
    if prototype_node_id.category == PSNCategory.TEXT_DECORATION_NODE:
      prototype_node = prototype_storage.text_decoration_nodes[
          prototype_node_id.index_in_category
      ]
      for confidence in (
          EditDagStateConfidence.HIGH_CONFIDENCE,
          EditDagStateConfidence.LOW_CONFIDENCE,
      ):
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=relative_state_for_delete(confidence=confidence),
                dest=relative_state_for_delete(
                    confidence=confidence, advance_prototype=True
                ),
                cost=0,
                info=EditDagEdgeInfo(
                    prototype_node_preorder_index=prototype_node_id.preorder_index,
                    target_node_preorder_index=NO_PREORDER_INDEX,
                    edit_action=EditAction.PROTOTYPE_DECORATION,
                    confidence=confidence,
                ),
            ),
        )

    if allow_delete:
      # Delete tokens from the prototype.
      if prototype_node_id.category == PSNCategory.TEXT_TOKEN_NODE:
        prototype_node = prototype_storage.text_token_nodes[
            prototype_node_id.index_in_category
        ]
        for confidence in (
            EditDagStateConfidence.HIGH_CONFIDENCE,
            EditDagStateConfidence.LOW_CONFIDENCE,
        ):
          node_utilities = token_node_utility_fn(prototype_node)
          if confidence == EditDagStateConfidence.HIGH_CONFIDENCE:
            delete_cost = node_utilities.high_confidence_delete_cost
          else:
            delete_cost = node_utilities.low_confidence_delete_cost

          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state_for_delete(confidence=confidence),
                  dest=relative_state_for_delete(
                      confidence=confidence, advance_prototype=True
                  ),
                  cost=delete_cost,
                  required_assignment=SharedVariableAssignment(
                      key=DecisionKey(
                          prototype_node_id.preorder_index,
                          DecisionCategory.NODE_IN_REGION,
                      ),
                      value=node_in_low_confidence_value(confidence),
                  ),
                  info=EditDagEdgeInfo(
                      prototype_node_preorder_index=prototype_node_id.preorder_index,
                      target_node_preorder_index=NO_PREORDER_INDEX,
                      edit_action=EditAction.DELETE,
                      confidence=confidence,
                  ),
              ),
          )

      # Recursively delete groups. (Group nodes themselves don't have any
      # costs currently.)
      if prototype_node_id.category == PSNCategory.GROUP_NODE:
        prototype_node = prototype_storage.group_nodes[
            prototype_node_id.index_in_category
        ]
        # Connect up to subproblem.
        for confidence in (
            EditDagStateConfidence.HIGH_CONFIDENCE,
            EditDagStateConfidence.LOW_CONFIDENCE,
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state_for_delete(confidence=confidence),
                  dest=EditDagState(
                      alignment=EditDagStateAlignment.RECURSIVELY_DELETING,
                      confidence=confidence,
                      # Deleting from the prototype subsequence from this
                      # group node.
                      prototype_sequence_preorder_index=(
                          prototype_node_id.preorder_index
                      ),
                      before_prototype_node=0,
                      # Stationary in the target.
                      target_sequence_preorder_index=(
                          target_sequence_preorder_index
                      ),
                      before_target_node=before_target_node,
                  ),
                  cost=0,
                  required_assignment=SharedVariableAssignment(
                      key=DecisionKey(
                          prototype_node_id.preorder_index,
                          DecisionCategory.NODE_IN_REGION,
                      ),
                      value=node_in_low_confidence_value(confidence),
                  ),
              ),
          )
        # Generate recursive delete subproblem, to handle all the children.
        build_recursive_delete_subsequence(
            prototype_storage=prototype_storage,
            prototype_node_ids=prototype_node.children_ids,
            fixed_before_target_node=before_target_node,
            fixed_target_sequence_preorder_index=target_sequence_preorder_index,
            incomplete_graph=incomplete_graph,
        )
        # Connect back to this level.
        for confidence in (
            EditDagStateConfidence.HIGH_CONFIDENCE,
            EditDagStateConfidence.LOW_CONFIDENCE,
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=EditDagState(
                      alignment=EditDagStateAlignment.RECURSIVELY_DELETING,
                      confidence=confidence,
                      # Deleting from the prototype subsequence from this
                      # group node.
                      prototype_sequence_preorder_index=(
                          prototype_node_id.preorder_index
                      ),
                      before_prototype_node=len(prototype_node.children_ids),
                      # Stationary in the target.
                      target_sequence_preorder_index=(
                          target_sequence_preorder_index
                      ),
                      before_target_node=before_target_node,
                  ),
                  dest=relative_state_for_delete(
                      confidence=confidence, advance_prototype=True
                  ),
                  cost=0,
              ),
          )

    ###########################################################
    #### Handle high/low confidence regions and early exit ####
    ###########################################################

    if prototype_node_id.category == PSNCategory.REGION_START_NODE:
      for i in range(3):
        # note: this loop is unrolled like this to help numba infer the
        # optional type of `info` below
        if i == 0:
          source_conf = EditDagStateConfidence.HIGH_CONFIDENCE
          dest_conf = EditDagStateConfidence.HIGH_CONFIDENCE
          decision_value = DecisionValue.FALSE
          cost = 0.0
          info = None
        elif i == 1:
          source_conf = EditDagStateConfidence.HIGH_CONFIDENCE
          dest_conf = EditDagStateConfidence.LOW_CONFIDENCE
          decision_value = DecisionValue.TRUE
          cost = low_confidence_region_cost
          info = EditDagEdgeInfo(
              prototype_node_preorder_index=prototype_node_id.preorder_index,
              target_node_preorder_index=NO_PREORDER_INDEX,
              edit_action=EditAction.START_LOW_CONFIDENCE,
              confidence=EditDagStateConfidence.NOT_APPLICABLE,
          )
        elif i == 2:
          source_conf = EditDagStateConfidence.LOW_CONFIDENCE
          dest_conf = EditDagStateConfidence.LOW_CONFIDENCE
          decision_value = DecisionValue.NOT_APPLICABLE
          cost = 0.0
          info = None
        else:
          raise NotImplementedError
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=relative_state_for_delete(confidence=source_conf),
                dest=relative_state_for_delete(
                    confidence=dest_conf,
                    advance_prototype=True,
                ),
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.REGION_SHOULD_START,
                    ),
                    value=decision_value,
                ),
                cost=cost,
                info=info,
            ),
        )

    if prototype_node_id.category == PSNCategory.REGION_END_NODE:
      for source_conf, dest_conf, decision_value in (
          # Don't end a low confidence region.
          (
              EditDagStateConfidence.LOW_CONFIDENCE,
              EditDagStateConfidence.LOW_CONFIDENCE,
              DecisionValue.FALSE,
          ),
          # End a low confidence region.
          (
              EditDagStateConfidence.LOW_CONFIDENCE,
              EditDagStateConfidence.HIGH_CONFIDENCE,
              DecisionValue.TRUE,
          ),
          # Stay outside of a low confidence region.
          (
              EditDagStateConfidence.HIGH_CONFIDENCE,
              EditDagStateConfidence.HIGH_CONFIDENCE,
              DecisionValue.NOT_APPLICABLE,
          ),
      ):
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=relative_state_for_delete(confidence=source_conf),
                dest=relative_state_for_delete(
                    confidence=dest_conf,
                    advance_prototype=True,
                ),
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.REGION_SHOULD_END,
                    ),
                    value=decision_value,
                ),
                cost=0.0,
            ),
        )

    if prototype_node_id.category == PSNCategory.EARLY_EXIT_NODE:
      for confidence in (
          EditDagStateConfidence.HIGH_CONFIDENCE,
          EditDagStateConfidence.LOW_CONFIDENCE,
      ):
        # Allowed to early exit.
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=relative_state_for_delete(confidence=confidence),
                dest=FINAL_STATE,
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.SHOULD_EARLY_EXIT,
                    ),
                    value=DecisionValue.TRUE,
                ),
                cost=0.0,
                info=EditDagEdgeInfo(
                    prototype_node_preorder_index=prototype_node_id.preorder_index,
                    target_node_preorder_index=NO_PREORDER_INDEX,
                    edit_action=EditAction.EARLY_EXIT,
                    confidence=confidence,
                ),
            ),
        )
        # Allowed to not early exit.
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=relative_state_for_delete(confidence=confidence),
                dest=relative_state_for_delete(
                    confidence=confidence, advance_prototype=True
                ),
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.SHOULD_EARLY_EXIT,
                    ),
                    value=DecisionValue.FALSE,
                ),
                cost=0.0,
            ),
        )

  @maybe_jit
  def build_matching_subproblem(
      prototype_storage: PackedSequenceNodeStorage,
      prototype_node_ids: list[PackedSequenceNodeID],
      prototype_sequence_preorder_index: int,
      target_storage: PackedSequenceNodeStorage,
      target_node_ids: list[PackedSequenceNodeID],
      target_sequence_preorder_index: int,
      incomplete_graph: gated_state_dag.PartialStateDAG,
      render_data: EditDagRenderData,
  ) -> None:
    """Builds a subgraph matching two subsequences.

    This subgraph starts from the two states:

      EditDagState(
          alignment=MATCH_START_ALIGNMENT,
          confidence=confidence,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_prototype_node=0,
          target_sequence_preorder_index=target_sequence_preorder_index,
          before_target_node=0,
      )
      for confidence in [
          EditDagStateConfidence.HIGH_CONFIDENCE,
          EditDagStateConfidence.LOW_CONFIDENCE,
      ]

    and ends at the two states:

      EditDagState(
          alignment=MATCH_END_ALIGNMENT,
          confidence=confidence,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_prototype_node=len(prototype_node_ids),
          target_sequence_preorder_index=target_sequence_preorder_index,
          before_target_node=len(target_node_ids),
      )
      for confidence in [
          EditDagStateConfidence.HIGH_CONFIDENCE,
          EditDagStateConfidence.LOW_CONFIDENCE,
      ]

    Paths between these states represent edit sequences, where nodes can either
    be matched or not.

    For implementation simplicity, we may generate a bunch of redundant edges
    that will end up not being reachable. For instance, if none of the nodes
    match, the only possible path will insert everything in the target, then
    delete everything in the prototype, so we don't need to consider every
    possible alignment of the two. A future optimization could try to avoid
    adding these. (However, these will be pruned by the reachability analysis
    anyways.)

    Args:
      prototype_storage: Storage for the prototype.
      prototype_node_ids: Subsequence of prototype node IDs to process.
      prototype_sequence_preorder_index: Identifier for this subsequence, either
        -1 for the root sequence or the preorder index of the GroupNode.
      target_storage: Storage for the target.
      target_node_ids: Subsequence of target node IDs to process.
      target_sequence_preorder_index: Identifier for this subsequence, as above.
      incomplete_graph: Graph that we should extend with this subgraph.
      render_data: Data for rendering.
    """
    render_data.subproblem_list.append(
        EditDagSubproblem(
            prototype_preorder_index=prototype_sequence_preorder_index,
            target_preorder_index=target_sequence_preorder_index,
        )
    )

    # We extend one-past-the-end of the prototype, to handle any remaining
    # (decoration) nodes in the target.
    for prototype_position in range(len(prototype_node_ids) + 1):
      if prototype_position < len(prototype_node_ids):
        prototype_node_id = prototype_node_ids[prototype_position]
      else:
        prototype_node_id = INVALID_NODE_ID

      # We similarly extend past the end of the target to handle remaining
      # (decoration, or prototype / early exit) nodes in the prototype.
      for target_position in range(len(target_node_ids) + 1):
        if target_position < len(target_node_ids):
          target_node_id = target_node_ids[target_position]
        else:
          target_node_id = INVALID_NODE_ID

        # Position-relative helper to save on boilerplate
        # pylint: disable=cell-var-from-loop
        def relative_state(
            alignment,
            confidence,
            advance_prototype=False,
            advance_target=False,
        ):
          delta_i = 1 if advance_prototype else 0
          delta_j = 1 if advance_target else 0
          return EditDagState(
              alignment=alignment,
              confidence=confidence,
              before_prototype_node=prototype_position + delta_i,
              prototype_sequence_preorder_index=(
                  prototype_sequence_preorder_index
              ),
              before_target_node=target_position + delta_j,
              target_sequence_preorder_index=target_sequence_preorder_index,
          )

        # pylint: enable=cell-var-from-loop

        ######################################################################
        #### Handle control nodes and decorations (in PROCESS_PROTOTYPE) ####
        ######################################################################

        maybe_delete_node_or_advance_in_prototype(
            alignment_type=EditDagStateAlignment.PROCESS_PROTOTYPE,
            prototype_node_id=prototype_node_id,
            prototype_storage=prototype_storage,
            before_target_node=target_position,
            target_sequence_preorder_index=target_sequence_preorder_index,
            incomplete_graph=incomplete_graph,
            allow_delete=False,
        )

        ##################################################################
        #### Decide whether to edit (PROCESS_PROTOTYPE -> MAY_DELETE, ####
        ####                         PROCESS_PROTOTYPE -> MATCH)      ####
        ##################################################################

        # These are no-ops in terms of position, they are just used to give a
        # one-way transition.
        for confidence in (
            EditDagStateConfidence.HIGH_CONFIDENCE,
            EditDagStateConfidence.LOW_CONFIDENCE,
        ):
          # Can start editing for a penalty.
          if confidence == EditDagStateConfidence.HIGH_CONFIDENCE:
            start_editing_cost = high_confidence_start_editing_cost
          else:
            start_editing_cost = low_confidence_start_editing_cost
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(
                      alignment=EditDagStateAlignment.PROCESS_PROTOTYPE,
                      confidence=confidence,
                  ),
                  dest=relative_state(
                      alignment=EditDagStateAlignment.MAY_DELETE,
                      confidence=confidence,
                  ),
                  cost=start_editing_cost,
                  info=EditDagEdgeInfo(
                      prototype_node_preorder_index=NO_PREORDER_INDEX,
                      target_node_preorder_index=NO_PREORDER_INDEX,
                      edit_action=EditAction.START_EDITING,
                      confidence=confidence,
                  ),
              ),
          )
          # Or can match from here for free.
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(
                      alignment=EditDagStateAlignment.PROCESS_PROTOTYPE,
                      confidence=confidence,
                  ),
                  dest=relative_state(
                      alignment=EditDagStateAlignment.MATCH,
                      confidence=confidence,
                  ),
                  cost=0.0,
              ),
          )

        ##########################################
        #### Handle deletions (in MAY_DELETE) ####
        ##########################################

        maybe_delete_node_or_advance_in_prototype(
            alignment_type=EditDagStateAlignment.MAY_DELETE,
            prototype_node_id=prototype_node_id,
            prototype_storage=prototype_storage,
            before_target_node=target_position,
            target_sequence_preorder_index=target_sequence_preorder_index,
            incomplete_graph=incomplete_graph,
            allow_delete=True,
        )

        ##################################################
        #### Stop deleting (MAY_DELETE -> MAY_INSERT) ####
        ##################################################

        # These are no-ops in terms of position, they are just used to give a
        # one-way transition from deleting to inserting.
        for confidence in (
            EditDagStateConfidence.HIGH_CONFIDENCE,
            EditDagStateConfidence.LOW_CONFIDENCE,
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(
                      alignment=EditDagStateAlignment.MAY_DELETE,
                      confidence=confidence,
                  ),
                  dest=relative_state(
                      alignment=EditDagStateAlignment.MAY_INSERT,
                      confidence=confidence,
                  ),
                  cost=0.0,
              ),
          )

        ###########################################
        #### Handle insertions (in MAY_INSERT) ####
        ###########################################

        maybe_insert_node_or_skip_decoration_in_target(
            alignment_type=EditDagStateAlignment.MAY_INSERT,
            target_node_id=target_node_id,
            target_storage=target_storage,
            before_prototype_node=prototype_position,
            prototype_sequence_preorder_index=prototype_sequence_preorder_index,
            incomplete_graph=incomplete_graph,
            allow_insert=True,
        )

        ##############################################
        #### Stop inserting (MAY_INSERT -> MATCH) ####
        ##############################################

        # These are no-ops in terms of position, they are just used to give a
        # one-way transition from inserting to matching.
        for confidence in (
            EditDagStateConfidence.HIGH_CONFIDENCE,
            EditDagStateConfidence.LOW_CONFIDENCE,
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(
                      alignment=EditDagStateAlignment.MAY_INSERT,
                      confidence=confidence,
                  ),
                  dest=relative_state(
                      alignment=EditDagStateAlignment.MATCH,
                      confidence=confidence,
                  ),
                  cost=0.0,
              ),
          )

        #######################################
        #### Insert decorations (in MATCH) ####
        #######################################

        # We can insert decoration nodes even if we are in the MATCH state,
        # since decoration nodes don't need to match with anything.
        maybe_insert_node_or_skip_decoration_in_target(
            alignment_type=EditDagStateAlignment.MATCH,
            target_node_id=target_node_id,
            target_storage=target_storage,
            before_prototype_node=prototype_position,
            prototype_sequence_preorder_index=prototype_sequence_preorder_index,
            incomplete_graph=incomplete_graph,
            allow_insert=False,  # only decorations allowed here!
        )

        #####################################################
        #### Handle matches (MATCH -> ADVANCE_PROTOTYPE) ####
        #####################################################

        # Match compatible tokens.
        if (
            target_node_id.category == PSNCategory.TEXT_TOKEN_NODE
            and prototype_node_id.category == PSNCategory.TEXT_TOKEN_NODE
        ):
          prototype_node = prototype_storage.text_token_nodes[
              prototype_node_id.index_in_category
          ]
          target_node = target_storage.text_token_nodes[
              target_node_id.index_in_category
          ]
          if (
              target_node.text_contents == prototype_node.text_contents
              and target_node.match_type == prototype_node.match_type
          ):
            # Compatible tokens!
            for confidence in (
                EditDagStateConfidence.HIGH_CONFIDENCE,
                EditDagStateConfidence.LOW_CONFIDENCE,
            ):
              node_utilities = token_node_utility_fn(prototype_node)
              if confidence == EditDagStateConfidence.HIGH_CONFIDENCE:
                keep_utility = node_utilities.high_confidence_match_utility
              else:
                keep_utility = node_utilities.low_confidence_match_utility

              gated_state_dag.partial_state_dag_add_edge(
                  incomplete_graph,
                  Edge(
                      source=relative_state(
                          alignment=EditDagStateAlignment.MATCH,
                          confidence=confidence,
                      ),
                      dest=relative_state(
                          alignment=EditDagStateAlignment.PROCESS_PROTOTYPE,
                          confidence=confidence,
                          advance_prototype=True,
                          advance_target=True,
                      ),
                      cost=-keep_utility,
                      required_assignment=SharedVariableAssignment(
                          key=DecisionKey(
                              prototype_node_id.preorder_index,
                              DecisionCategory.NODE_IN_REGION,
                          ),
                          value=node_in_low_confidence_value(confidence),
                      ),
                      info=EditDagEdgeInfo(
                          prototype_node_preorder_index=prototype_node_id.preorder_index,
                          target_node_preorder_index=target_node_id.preorder_index,
                          edit_action=EditAction.KEEP,
                          confidence=confidence,
                      ),
                  ),
              )

        # Match compatible groups (recursively).
        if (
            prototype_node_id.category == PSNCategory.GROUP_NODE
            and target_node_id.category == PSNCategory.GROUP_NODE
        ):
          prototype_node = prototype_storage.group_nodes[
              prototype_node_id.index_in_category
          ]
          target_node = target_storage.group_nodes[
              target_node_id.index_in_category
          ]
          if target_node.match_type == prototype_node.match_type:
            # Connect up to subproblem.
            for confidence in (
                EditDagStateConfidence.HIGH_CONFIDENCE,
                EditDagStateConfidence.LOW_CONFIDENCE,
            ):
              gated_state_dag.partial_state_dag_add_edge(
                  incomplete_graph,
                  Edge(
                      source=relative_state(
                          alignment=EditDagStateAlignment.MATCH,
                          confidence=confidence,
                      ),
                      dest=EditDagState(
                          alignment=MATCH_START_ALIGNMENT,
                          confidence=confidence,
                          # Start at the beginning of both the prototype and
                          # target subsequences.
                          prototype_sequence_preorder_index=(
                              prototype_node_id.preorder_index
                          ),
                          before_prototype_node=0,
                          target_sequence_preorder_index=(
                              target_node_id.preorder_index
                          ),
                          before_target_node=0,
                      ),
                      cost=0,
                      required_assignment=SharedVariableAssignment(
                          key=DecisionKey(
                              prototype_node_id.preorder_index,
                              DecisionCategory.NODE_IN_REGION,
                          ),
                          value=node_in_low_confidence_value(confidence),
                      ),
                  ),
              )
            # Generate recursive matching subproblem, for the pairwise children.
            build_matching_subproblem(
                prototype_storage=prototype_storage,
                prototype_node_ids=prototype_node.children_ids,
                prototype_sequence_preorder_index=(
                    prototype_node_id.preorder_index
                ),
                target_storage=target_storage,
                target_node_ids=target_node.children_ids,
                target_sequence_preorder_index=target_node_id.preorder_index,
                incomplete_graph=incomplete_graph,
                render_data=render_data,
            )
            # Connect back to this level.
            for confidence in (
                EditDagStateConfidence.HIGH_CONFIDENCE,
                EditDagStateConfidence.LOW_CONFIDENCE,
            ):
              gated_state_dag.partial_state_dag_add_edge(
                  incomplete_graph,
                  Edge(
                      source=EditDagState(
                          alignment=MATCH_END_ALIGNMENT,
                          confidence=confidence,
                          # Finish at the end of both the prototype and target
                          # subsequences.
                          prototype_sequence_preorder_index=(
                              prototype_node_id.preorder_index
                          ),
                          before_prototype_node=len(
                              prototype_node.children_ids
                          ),
                          target_sequence_preorder_index=(
                              target_node_id.preorder_index
                          ),
                          before_target_node=len(target_node.children_ids),
                      ),
                      dest=relative_state(
                          alignment=EditDagStateAlignment.PROCESS_PROTOTYPE,
                          confidence=confidence,
                          advance_prototype=True,
                          advance_target=True,
                      ),
                      cost=0,
                  ),
              )

        # (end of target processing)
      # (end of prototype)
    # End of `build_matching_subproblem`; the caller will handle connections
    # from this subgraph to the parent graph if applicable.

  @maybe_jit
  def build_recursive_delete_subsequence(
      prototype_storage: PackedSequenceNodeStorage,
      prototype_node_ids: list[PackedSequenceNodeID],
      fixed_before_target_node: int,
      fixed_target_sequence_preorder_index: int,
      incomplete_graph: gated_state_dag.PartialStateDAG,
  ) -> None:
    """Builds a subgraph recursively deleting all nodes in `prototype_node_ids`.

    This subgraph starts from the two states:

      EditDagState(
          alignment=EditDagStateAlignment.RECURSIVELY_DELETING,
          confidence=confidence,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_prototype_node=0,
          target_sequence_preorder_index=target_sequence_preorder_index,
          before_target_node=before_target_node,
      )
      for confidence in [
          EditDagStateConfidence.HIGH_CONFIDENCE,
          EditDagStateConfidence.LOW_CONFIDENCE,
      ]

    and ends at the two states:

      EditDagState(
          alignment=EditDagStateAlignment.RECURSIVELY_DELETING,
          confidence=confidence,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_prototype_node=len(prototype_node_ids),
          target_sequence_preorder_index=target_sequence_preorder_index,
          before_target_node=before_target_node,
      )
      for confidence in [
          EditDagStateConfidence.HIGH_CONFIDENCE,
          EditDagStateConfidence.LOW_CONFIDENCE,
      ]

    Paths through this subgraph are constrained to only delete nodes from the
    prototype, and cannot advance in the target sequence at all. Confidence
    levels can change when we see region start/end nodes.

    Note: prototype_sequence_preorder_index is an identifier for this
    subsequence, either -1 for the root sequence or the preorder index of the
    GroupNode, and is inferred from node IDs.

    Args:
      prototype_storage: Storage for the prototype.
      prototype_node_ids: Subsequence of prototype node IDs to process.
      fixed_before_target_node: Position in the subsequence in the target from
        which this deletion request originated. As above, this is propagated
        through states but isn't directly used.
      fixed_target_sequence_preorder_index: Identifier for the subsequence in
        the target from which this deletion request originated. This is
        propagated through the states to disambiguate from other possible
        deletions, but isn't directly used otherwise.
      incomplete_graph: Graph that we should extend with this subgraph.
    """
    for prototype_node_id in prototype_node_ids:
      maybe_delete_node_or_advance_in_prototype(
          alignment_type=EditDagStateAlignment.RECURSIVELY_DELETING,
          prototype_node_id=prototype_node_id,
          prototype_storage=prototype_storage,
          before_target_node=fixed_before_target_node,
          target_sequence_preorder_index=fixed_target_sequence_preorder_index,
          incomplete_graph=incomplete_graph,
          allow_delete=True,
      )

  @maybe_jit
  def build_recursive_insert_subsequence(
      target_storage: PackedSequenceNodeStorage,
      target_node_ids: list[PackedSequenceNodeID],
      fixed_before_prototype_node: int,
      fixed_prototype_sequence_preorder_index: int,
      incomplete_graph: gated_state_dag.PartialStateDAG,
  ) -> None:
    """Builds a subgraph recursively inserting all nodes in `target_node_ids`.

    This subgraph starts from the two states:

      EditDagState(
          alignment=EditDagStateAlignment.RECURSIVELY_INSERTING,
          confidence=confidence,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_prototype_node=before_prototype_node,
          target_sequence_preorder_index=target_sequence_preorder_index,
          before_target_node=0,
      )
      for confidence in [
          EditDagStateConfidence.HIGH_CONFIDENCE,
          EditDagStateConfidence.LOW_CONFIDENCE,
      ]

    and ends at the two states:

      EditDagState(
          alignment=EditDagStateAlignment.RECURSIVELY_INSERTING,
          confidence=confidence,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_prototype_node=before_prototype_node,
          target_sequence_preorder_index=target_sequence_preorder_index,
          before_target_node=len(target_node_ids),
      )
      for confidence in [
          EditDagStateConfidence.HIGH_CONFIDENCE,
          EditDagStateConfidence.LOW_CONFIDENCE,
      ]

    Paths through this subgraph are constrained to only insert nodes from the
    target, and cannot advance in the prototype sequence at all. Confidence
    levels will not change, since confidence is only modified in the prototype.

    Note: target_sequence_preorder_index is an identifier for this
    subsequence, either -1 for the root sequence or the preorder index of the
    GroupNode, and is inferred from node IDs.

    Args:
      target_storage: Storage for the target.
      target_node_ids: Subsequence of target node IDs to process.
      fixed_before_prototype_node: Position in the subsequence in the prototype
        from which this insertion request originated. As above, this is
        propagated through states but isn't directly used.
      fixed_prototype_sequence_preorder_index: Identifier for the subsequence in
        the prototype from which this insertion request originated. This is
        propagated through the states to disambiguate from other possible
        insertions, but isn't directly used otherwise.
      incomplete_graph: Graph that we should extend with this subgraph.
    """
    for target_node_id in target_node_ids:
      maybe_insert_node_or_skip_decoration_in_target(
          alignment_type=EditDagStateAlignment.RECURSIVELY_INSERTING,
          target_node_id=target_node_id,
          target_storage=target_storage,
          before_prototype_node=fixed_before_prototype_node,
          prototype_sequence_preorder_index=(
              fixed_prototype_sequence_preorder_index
          ),
          incomplete_graph=incomplete_graph,
          allow_insert=True,
      )

  # pylint: enable=g-long-ternary

  # Return the final builder function.
  return construct_edit_dag


################################################################################
# Graph packing
################################################################################


def pack_dag(
    dag: gated_state_dag.CompleteStateDAG,
) -> tuple[
    packed_dags.PackedStateDAG, packed_dags.PackedStateDAGConversionData
]:
  """Packing function for DAGs generated by this module."""
  return packed_dags.convert_dag_to_packed(
      dag=dag,
      missing_assignment_value=DecisionValue.NOT_APPLICABLE,
      variable_value_ordering=list(DecisionValue),
  )


def make_specialized_dag_packer(
    states_rewritten_as_integers: bool = False,
) -> tuple[
    Callable[
        [gated_state_dag.CompleteStateDAG],
        tuple[
            packed_dags.PackedStateDAG,
            packed_dags.PackedStateDAGConversionData,
        ],
    ],
    Callable[
        [gated_state_dag.CompleteStateDAG], gated_state_dag.CompleteStateDAG
    ],
]:
  """Returns a specialized helper to pack a graph."""
  (
      jit_packer,
      graph_converter,
  ) = packed_dags.make_specialized_fn__convert_dag_to_packed(
      example_state=42 if states_rewritten_as_integers else EXAMPLE_STATE,
      example_variable_key=EXAMPLE_VARIABLE_KEY,
      example_variable_value=DecisionValue.NOT_APPLICABLE,
      example_info=EXAMPLE_EDGE_INFO,
  )
  variable_value_ordering = tuple(DecisionValue)

  @numba.njit
  def packer(dag: gated_state_dag.CompleteStateDAG):
    """Numba-compiled packing function for DAGs generated by this module."""
    return jit_packer(
        dag=dag,
        missing_assignment_value=DecisionValue.NOT_APPLICABLE,
        variable_value_ordering=numba.typed.List(variable_value_ordering),
    )

  return packer, graph_converter  # pytype: disable=bad-return-type


################################################################################
# Solution postprocessing
################################################################################


def _group_edit_sequence(
    path: list[Edge],
    prototype: PackedSequenceNodeStorage,
    target: PackedSequenceNodeStorage,
) -> Iterable[
    tuple[str, Optional[EditAction], Optional[EditDagStateConfidence], float]
]:
  """Groups parts of an edit sequence based on edit type.

  Args:
    path: Path through an edit DAG, representing an edit sequence.
    prototype: The prototype node for this DAG.
    target: The target node for this DAG.

  Yields:
    Tuples (content, edit_action, confidence, total_cost), combining edits
    based on edit action and confidence, and rewriting decorations to appear
    like normal insertions and deletions. Intended for use in visualizations.
  """
  # Group nodes based on their edit actions.
  current_edit_action = None
  current_confidence = None
  current_parts = []
  current_cost = 0.0
  # Statefully remember the last edit action that was set, to determine how
  # to render decoration nodes.
  last_edit_action_for_decorations = None
  for edge in path:
    if edge.info is not None:
      assert isinstance(edge.info, EditDagEdgeInfo)
      cost = edge.cost
      if edge.info.prototype_node_preorder_index != NO_PREORDER_INDEX:
        content = packed_sequence_nodes.render_text_contents_of_node(
            prototype.preorder_traversal[
                edge.info.prototype_node_preorder_index
            ],
            prototype,
        )
      elif edge.info.target_node_preorder_index != NO_PREORDER_INDEX:
        content = packed_sequence_nodes.render_text_contents_of_node(
            target.preorder_traversal[edge.info.target_node_preorder_index],
            target,
        )
      else:
        content = ""
      edit_action = edge.info.edit_action
      confidence = edge.info.confidence

    elif edge.cost:
      cost = edge.cost
      content = ""
      edit_action = None
      confidence = None

    else:
      continue

    # Handle decorations separately.
    if edit_action == EditAction.PROTOTYPE_DECORATION:
      # Try to follow the previous action.
      if last_edit_action_for_decorations == EditAction.DELETE:
        edit_action = EditAction.DELETE
      else:
        edit_action = EditAction.KEEP

    elif edit_action == EditAction.TARGET_DECORATION:
      if last_edit_action_for_decorations == EditAction.KEEP:
        # Skip this node to simplify the rendering, since we will just use the
        # decorations (e.g. whitespace) from the prototype.
        continue
      else:
        # Rewrite this to look like a normal insert.
        edit_action = EditAction.INSERT
        confidence = EditDagStateConfidence.NOT_APPLICABLE

    if edit_action != current_edit_action or confidence != current_confidence:
      if current_edit_action is not None or current_cost != 0.0:
        yield (
            "".join(current_parts),
            current_edit_action,
            current_confidence,
            current_cost,
        )

      current_edit_action = edit_action
      current_confidence = confidence
      current_parts = []
      current_cost = 0.0

    if (
        edit_action is not None
        and edit_action != EditAction.START_LOW_CONFIDENCE
        and edit_action != EditAction.START_EDITING
    ):
      last_edit_action_for_decorations = edit_action

    current_parts.append(content)
    current_cost += cost

  if current_edit_action is not None or current_cost != 0.0:
    yield (
        "".join(current_parts),
        current_edit_action,
        current_confidence,
        current_cost,
    )


def extract_edit_sequence_text(
    path: list[Edge],
    prototype: PackedSequenceNodeStorage,
    target: PackedSequenceNodeStorage,
) -> str:
  """Extracts a text representation of the edit sequence from a solution path.

  Args:
    path: A path in a utility graph built by `construct_utility_graph`.
    prototype: Corresponding prototype for this graph.
    target: Corresponding target for this graph.

  Returns:
    An edit script for transforming the prototype into a prefix of the target.
  """
  output_parts = []
  total_cost = 0.0

  for content, edit_action, confidence, cost in _group_edit_sequence(
      path, prototype, target
  ):
    edit_action_name = None if edit_action is None else edit_action.name
    confidence_name = None if confidence is None else confidence.name
    output_parts.append(
        f"{edit_action_name}, {confidence_name}: {repr(content)} => {cost}\n"
    )
    total_cost += cost
  output_parts.append(f"FINISH (total cost: {total_cost})")
  return "".join(output_parts)


def extract_edit_sequence_html(
    path: list[Edge],
    prototype: PackedSequenceNodeStorage,
    target: PackedSequenceNodeStorage,
    prefix: str = "",
    start_editing_marker: str = "፠",
    style_override: str = "",
) -> str:
  """Extracts a HTML representation of the edit sequence from a solution path.

  Args:
    path: A path in a utility graph built by `construct_utility_graph`.
    prototype: Corresponding prototype for this graph.
    target: Corresponding target for this graph.
    prefix: Prefix to render, e.g. the context provided to the model.
    start_editing_marker: Marker to render whenever an edit starts.
    style_override: Optional override for CSS styling information.

  Returns:
    An HTML source string giving a visual representation of the edit sequence
    for this path.
  """
  output_parts = []
  if style_override:
    style = style_override
  else:
    style = textwrap.dedent(
        """\
        <style>
        .edit_sequence_root {
            white-space: pre;
            font-family: monospace;
        }
        .edit_sequence_root .prefix {
            font-weight: bold;
            color: black;
        }
        .edit_sequence_root .high_confidence,
        .edit_sequence_root .insert.was_high_confidence{
            font-weight: normal;
        }
        .edit_sequence_root .low_confidence,
        .edit_sequence_root .insert.was_low_confidence {
            font-weight: normal;
            background-color: #ffcc99;
        }
        .edit_sequence_root .keep {
            color: black;
        }
        .edit_sequence_root .delete {
            color: darkred;
            text-decoration: line-through;
        }
        .edit_sequence_root .insert {
            color: darkgreen;
            text-decoration: underline;
            font-weight: normal;
        }
        .edit_sequence_root .start_editing {
            color: red;
            font-weight: bold;
        }
        </style>"""
    )
  output_parts.append(style)
  output_parts.append('<span class="edit_sequence_root">')
  output_parts.append(f'<span class="prefix">{html.escape(prefix)}</span>')

  last_confidence = EditDagStateConfidence.NOT_APPLICABLE
  for content, edit_action, confidence, _ in _group_edit_sequence(
      path, prototype, target
  ):
    # print(content, edit_action, confidence)
    if edit_action == EditAction.START_EDITING:
      content = start_editing_marker
    elif not content:
      continue

    class_names = []
    if edit_action == EditAction.INSERT:
      class_names.append("insert")
      if last_confidence == EditDagStateConfidence.HIGH_CONFIDENCE:
        class_names.append("was_high_confidence")
      elif last_confidence == EditDagStateConfidence.LOW_CONFIDENCE:
        class_names.append("was_low_confidence")
    elif edit_action == EditAction.DELETE:
      class_names.append("delete")
    elif edit_action == EditAction.KEEP:
      class_names.append("keep")
    elif edit_action == EditAction.START_EDITING:
      class_names.append("start_editing")

    if confidence == EditDagStateConfidence.HIGH_CONFIDENCE:
      class_names.append("high_confidence")
    elif confidence == EditDagStateConfidence.LOW_CONFIDENCE:
      class_names.append("low_confidence")

    last_confidence = confidence

    output_parts.append(
        f'<span class="{" ".join(class_names)}">{html.escape(content)}</span>'
    )

  output_parts.append("</span>")
  return "".join(output_parts)


def extract_edit_summary_metrics(
    path: list[Edge],
    prototype: PackedSequenceNodeStorage,
    target: PackedSequenceNodeStorage,
) -> dict[str, float]:
  """Extracts summary metrics for this edit path.

  Args:
    path: A path in a utility graph built by `construct_utility_graph`.
    prototype: Corresponding prototype for this graph.
    target: Corresponding target for this graph.

  Returns:
    A breakdown of costs and types of edit.
  """
  result = collections.defaultdict(lambda: 0)
  for edge in path:
    if edge.info is not None:
      assert isinstance(edge.info, EditDagEdgeInfo)
      if edge.info.prototype_node_preorder_index != NO_PREORDER_INDEX:
        content = packed_sequence_nodes.render_text_contents_of_node(
            prototype.preorder_traversal[
                edge.info.prototype_node_preorder_index
            ],
            prototype,
        )
      elif edge.info.target_node_preorder_index != NO_PREORDER_INDEX:
        content = packed_sequence_nodes.render_text_contents_of_node(
            target.preorder_traversal[edge.info.target_node_preorder_index],
            target,
        )
      else:
        content = ""

      edit_action_name = edge.info.edit_action.name
      confidence_name = edge.info.confidence.name
      result[f"{edit_action_name}_{confidence_name}_cost"] += edge.cost
      result[f"{edit_action_name}_{confidence_name}_chars"] += len(content)
      result[f"{edit_action_name}_{confidence_name}_edges"] += 1

  return dict(result)


################################################################################
# Rendering annotations
################################################################################

STATE_OFFSET_MAPPING = {
    # Normal states
    (
        EditDagStateAlignment.PROCESS_PROTOTYPE,
        EditDagStateConfidence.HIGH_CONFIDENCE,
    ): 0,
    (
        EditDagStateAlignment.PROCESS_PROTOTYPE,
        EditDagStateConfidence.LOW_CONFIDENCE,
    ): 1,
    (
        EditDagStateAlignment.MAY_DELETE,
        EditDagStateConfidence.HIGH_CONFIDENCE,
    ): 2,
    (
        EditDagStateAlignment.MAY_DELETE,
        EditDagStateConfidence.LOW_CONFIDENCE,
    ): 3,
    (
        EditDagStateAlignment.MAY_INSERT,
        EditDagStateConfidence.HIGH_CONFIDENCE,
    ): 4,
    (
        EditDagStateAlignment.MAY_INSERT,
        EditDagStateConfidence.LOW_CONFIDENCE,
    ): 5,
    (
        EditDagStateAlignment.MATCH,
        EditDagStateConfidence.HIGH_CONFIDENCE,
    ): 6,
    (
        EditDagStateAlignment.MATCH,
        EditDagStateConfidence.LOW_CONFIDENCE,
    ): 7,
    # Recursive handling
    (
        EditDagStateAlignment.RECURSIVELY_DELETING,
        EditDagStateConfidence.HIGH_CONFIDENCE,
    ): 2,
    (
        EditDagStateAlignment.RECURSIVELY_DELETING,
        EditDagStateConfidence.LOW_CONFIDENCE,
    ): 3,
    (
        EditDagStateAlignment.RECURSIVELY_INSERTING,
        EditDagStateConfidence.HIGH_CONFIDENCE,
    ): 4,
    (
        EditDagStateAlignment.RECURSIVELY_INSERTING,
        EditDagStateConfidence.LOW_CONFIDENCE,
    ): 5,
    # Special
    (
        EditDagStateAlignment.SPECIAL_FINAL_STATE,
        EditDagStateConfidence.NOT_APPLICABLE,
    ): 0,
}
NUM_STATE_OFFSETS = max(STATE_OFFSET_MAPPING.values()) + 1


class EditDagGraphAnnotator(dag_annotator.StateDAGAnnotator):
  """Annotator for Edit DAGs."""

  prototype: PackedSequenceNodeStorage
  target: PackedSequenceNodeStorage
  render_config: EditDagRenderConfig
  render_data: EditDagRenderData
  text_annotations: list[dag_annotator.TextAnnotation]
  layout_positions: dict[Any, Any]

  def __init__(
      self,
      prototype: PackedSequenceNodeStorage,
      target: PackedSequenceNodeStorage,
      render_data: EditDagRenderData,
      render_config: EditDagRenderConfig,
  ):
    """Creates the annotator and precomputes keypoints and global annotations."""
    self.prototype = prototype
    self.target = target
    self.render_config = render_config
    self.render_data = render_data
    self.text_annotations = []
    self.layout_positions = {}

    self.precompute_layout()

  def precompute_layout(self):
    """Precomputes locations for rendering."""
    render_config = self.render_config
    # Without a type annotation, pytype tends to infer an incorrect type.
    layout_positions: dict[Any, Any] = self.layout_positions

    # Prototype is arranged on horizontal axis, along with start and end nodes.
    current_x = 0
    left = current_x
    current_x += render_config.state_width
    right = current_x
    layout_positions["start_x"] = (left, right)
    current_x += render_config.gap_horizontal

    def process_prototype(
        node_ids: list[PackedSequenceNodeID], preorder_index: int
    ):
      nonlocal current_x

      left = current_x
      layout_positions["prototype_subproblem_left_x", preorder_index] = (
          current_x
      )
      current_x += render_config.gap_horizontal

      for i, node_id in enumerate(node_ids):
        # States before these nodes
        left = current_x
        current_x += render_config.state_width
        right = current_x
        layout_positions["prototype_state_x", preorder_index, i] = (left, right)

        # Transition, possibly including a subproblem.
        if node_id.category == PSNCategory.GROUP_NODE:
          current_x += render_config.gap_horizontal
          process_prototype(
              self.prototype.group_nodes[
                  node_id.index_in_category
              ].children_ids,
              node_id.preorder_index,
          )
          current_x += render_config.gap_horizontal
        else:
          left = current_x
          current_x += render_config.gap_horizontal
          right = current_x
          # Annotate the prototype text on the top.
          bounds = rendering.BoundingBox.from_boundary(
              left=left,
              right=right,
              top=-render_config.gap_vertical - render_config.state_height,
              bottom=-render_config.gap_vertical,
          )
          if node_id.category == PSNCategory.TEXT_TOKEN_NODE:
            node = self.prototype.text_token_nodes[node_id.index_in_category]
            self.text_annotations.append(
                dag_annotator.TextAnnotation(
                    bounds=bounds,
                    display_text=repr(node.text_contents)[1:-1],
                    text_size=self.render_config.font_size,
                    hover_text=repr((node_id, node)),
                    style_tags=("text_token",),
                )
            )
          elif node_id.category == PSNCategory.TEXT_DECORATION_NODE:
            node = self.prototype.text_decoration_nodes[
                node_id.index_in_category
            ]
            self.text_annotations.append(
                dag_annotator.TextAnnotation(
                    bounds=bounds,
                    display_text=repr(node.text_contents)[1:-1],
                    text_size=self.render_config.font_size,
                    hover_text=repr((node_id, node)),
                    style_tags=("text_decoration",),
                )
            )
          elif node_id.category == PSNCategory.REGION_START_NODE:
            self.text_annotations.append(
                dag_annotator.TextAnnotation(
                    bounds=bounds,
                    display_text="⟪",
                    text_size=self.render_config.font_size,
                    hover_text=repr(node_id),
                    style_tags=("region_start",),
                )
            )
          elif node_id.category == PSNCategory.REGION_END_NODE:
            self.text_annotations.append(
                dag_annotator.TextAnnotation(
                    bounds=bounds,
                    display_text="⟫",
                    text_size=self.render_config.font_size,
                    hover_text=repr(node_id),
                    style_tags=("region_end",),
                )
            )
          elif node_id.category == PSNCategory.EARLY_EXIT_NODE:
            self.text_annotations.append(
                dag_annotator.TextAnnotation(
                    bounds=bounds,
                    display_text="➞|",
                    text_size=self.render_config.font_size,
                    hover_text=repr(node_id),
                    style_tags=("early_exit",),
                )
            )
          else:
            self.text_annotations.append(
                dag_annotator.TextAnnotation(
                    bounds=bounds,
                    display_text=repr(node_id),
                    text_size=self.render_config.font_size,
                    hover_text=repr(node_id),
                    style_tags=("unexpected_node",),
                )
            )

      # States after all nodes
      left = current_x
      current_x += render_config.state_width
      right = current_x
      layout_positions["prototype_state_x", preorder_index, len(node_ids)] = (
          left,
          right,
      )

      current_x += render_config.gap_horizontal
      layout_positions["prototype_subproblem_right_x", preorder_index] = (
          current_x
      )

    process_prototype(self.prototype.root_sequence, ROOT_PREORDER_INDEX)

    current_y = 0

    # Target is arranged vertically.
    def process_target(
        node_ids: list[PackedSequenceNodeID], preorder_index: int
    ):
      nonlocal current_y

      layout_positions["target_subproblem_top_y", preorder_index] = current_y
      current_y += render_config.gap_vertical

      for i, node_id in enumerate(node_ids):
        # States before these nodes
        for offset in range(NUM_STATE_OFFSETS):
          top = current_y
          current_y += render_config.state_height
          bottom = current_y
          layout_positions["target_state_y", preorder_index, i, offset] = (
              top,
              bottom,
          )
          current_y += render_config.state_separation
        current_y -= render_config.state_separation

        # Transition, possibly including a subproblem.
        if node_id.category == PSNCategory.GROUP_NODE:
          current_y += render_config.gap_vertical
          process_target(
              self.target.group_nodes[node_id.index_in_category].children_ids,
              node_id.preorder_index,
          )
          current_y += render_config.gap_vertical
        else:
          top = current_y
          current_y += render_config.gap_vertical
          bottom = current_y
          # Annotate the target text on the left.
          bounds = rendering.BoundingBox.from_boundary(
              left=-render_config.state_width, right=0, top=top, bottom=bottom
          )
          if node_id.category == PSNCategory.TEXT_TOKEN_NODE:
            node = self.target.text_token_nodes[node_id.index_in_category]
            self.text_annotations.append(
                dag_annotator.TextAnnotation(
                    bounds=bounds,
                    display_text=repr(node.text_contents)[1:-1],
                    text_size=self.render_config.font_size,
                    hover_text=repr((node_id, node)),
                    style_tags=("text_token",),
                )
            )
          elif node_id.category == PSNCategory.TEXT_DECORATION_NODE:
            node = self.target.text_decoration_nodes[node_id.index_in_category]
            self.text_annotations.append(
                dag_annotator.TextAnnotation(
                    bounds=bounds,
                    display_text=repr(node.text_contents)[1:-1],
                    text_size=self.render_config.font_size,
                    hover_text=repr((node_id, node)),
                    style_tags=("text_decoration",),
                )
            )
          else:
            self.text_annotations.append(
                dag_annotator.TextAnnotation(
                    bounds=bounds,
                    display_text=repr(node_id),
                    text_size=self.render_config.font_size,
                    hover_text=repr(node_id),
                    style_tags=("unexpected_node",),
                )
            )

      # States after all nodes
      for offset in range(NUM_STATE_OFFSETS):
        top = current_y
        current_y += render_config.state_height
        bottom = current_y
        layout_positions[
            "target_state_y", preorder_index, len(node_ids), offset
        ] = (
            top,
            bottom,
        )
        current_y += render_config.state_separation
      current_y -= render_config.state_separation

      current_y += render_config.gap_vertical
      layout_positions["target_subproblem_bottom_y", preorder_index] = current_y

    process_target(self.target.root_sequence, ROOT_PREORDER_INDEX)

    # Final state is appended at the end.
    current_x += render_config.gap_horizontal
    left = current_x
    current_x += render_config.state_width
    right = current_x
    layout_positions["final_state_x"] = (left, right)

    current_y += render_config.gap_vertical
    top = current_y
    current_y += render_config.state_height
    bottom = current_y
    layout_positions["final_state_y"] = (top, bottom)

  def annotate_state(
      self,
      state: EditDagState,
  ) -> dag_annotator.StateAnnotation:
    """Assigns rendering annotations to a state.

    Args:
      state: The state to assign annotations to.

    Returns:
      Information on how to render this state.
    """
    if state == FINAL_STATE:
      # Special case
      left_bound, right_bound = self.layout_positions["final_state_x"]
      top_bound, bottom_bound = self.layout_positions["final_state_y"]
      display_text = state.alignment.shortname()
    else:
      state_offset = STATE_OFFSET_MAPPING[state.alignment, state.confidence]
      left_bound, right_bound = self.layout_positions[
          "prototype_state_x",
          state.prototype_sequence_preorder_index,
          state.before_prototype_node,
      ]
      top_bound, bottom_bound = self.layout_positions[
          "target_state_y",
          state.target_sequence_preorder_index,
          state.before_target_node,
          state_offset,
      ]
      display_text = (
          f"{state.alignment.shortname()} {state.confidence.shortname()}"
      )

    return dag_annotator.StateAnnotation(
        bounds=rendering.BoundingBox.from_boundary(
            left=left_bound,
            top=top_bound,
            right=right_bound,
            bottom=bottom_bound,
        ),
        display_text=display_text,
        text_size=self.render_config.font_size,
        hover_text=repr(state),
    )

  def annotate_edge(
      self,
      edge: Edge,
  ) -> dag_annotator.EdgeAnnotation:
    """Assigns rendering annotations to an edge.

    Args:
      edge: The edge to assign annotations to.

    Returns:
      Information on how to render this edge.
    """
    style_tags = [
        f"from_{edge.source.alignment.name}",
        f"to_{edge.dest.alignment.name}",
        f"from_{edge.source.confidence.name}",
        f"to_{edge.dest.confidence.name}",
    ]
    # Handle inserted edge tracebacks.
    if isinstance(edge.info, dict) and "original_info" in edge.info:
      edge_info = edge.info["original_info"]
    else:
      edge_info = edge.info
    if edge_info is not None:
      edit_action = edge_info.edit_action
      style_tags.append(edit_action.name)
    else:
      edit_action = None

    # Disambiguate offsets based on the opposite edge.
    start_offset = STATE_OFFSET_MAPPING[
        edge.source.alignment, edge.source.confidence
    ]
    end_alignment = (start_offset + 1) / (NUM_STATE_OFFSETS + 1)
    end_offset = STATE_OFFSET_MAPPING[edge.dest.alignment, edge.dest.confidence]
    start_alignment = (end_offset + 1) / (NUM_STATE_OFFSETS + 1)

    # Try to align vertical components in a meaningful way.
    known_offsets = {
        (
            EditDagStateAlignment.PROCESS_PROTOTYPE,
            EditDagStateAlignment.MATCH,
        ): 1,
        (
            EditDagStateAlignment.PROCESS_PROTOTYPE,
            EditDagStateAlignment.MAY_DELETE,
        ): 2,
        (EditDagStateAlignment.MAY_DELETE, EditDagStateAlignment.MAY_INSERT): 3,
        (EditDagStateAlignment.MAY_INSERT, EditDagStateAlignment.MAY_INSERT): 4,
        (EditDagStateAlignment.MAY_INSERT, EditDagStateAlignment.MATCH): 5,
    }
    denominator = 6
    if (edge.source.alignment, edge.dest.alignment) in known_offsets:
      vertical_line_offset = (
          known_offsets[edge.source.alignment, edge.dest.alignment]
          / denominator
      )
      if edge.dest.confidence == EditDagStateConfidence.HIGH_CONFIDENCE:
        vertical_line_offset -= 0.1 / denominator
      else:
        vertical_line_offset += 0.1 / denominator
    else:
      vertical_line_offset = 0.63 * start_alignment + 0.37 * end_alignment

    if edge.required_assignment:
      assignment_summary = (
          f"\n{edge.required_assignment.key.category.shortname()}:"
          f" {edge.required_assignment.value.shortname()}"
      )
      # index = tuple(DecisionValue).index(edge.required_assignment.value)
      # elbow_distance = (index + 1) / (len(DecisionValue) + 1)
      style_tags.append(edge.required_assignment.key.category.name)
      style_tags.append(edge.required_assignment.value.name)
    else:
      assignment_summary = ""

    source_annotation = self.annotate_state(edge.source)
    target_annotation = self.annotate_state(edge.dest)
    if source_annotation.bounds.left == target_annotation.bounds.left:
      primary_axis = dag_annotator.ElbowEdgeOrientation.VERTICAL
      start_alignment = vertical_line_offset
      end_alignment = vertical_line_offset
      elbow_distance = 0
      elbow_distance_adjust = self.render_config.state_separation * 0.25
    else:
      primary_axis = dag_annotator.ElbowEdgeOrientation.HORIZONTAL
      elbow_distance = 1 - vertical_line_offset
      elbow_distance_adjust = 0

    # Override for early exit edges.
    if edit_action == EditAction.EARLY_EXIT:
      elbow_distance = 0
      elbow_distance_adjust = self.render_config.gap_horizontal * 0.2
      text_alignment = 0
      text_alignment_adjust = self.render_config.state_separation
    else:
      text_alignment = 0.5
      text_alignment_adjust = 0

    return dag_annotator.ElbowEdgeAnnotation(
        primary_axis=primary_axis,
        elbow_distance=elbow_distance,
        elbow_distance_adjust=elbow_distance_adjust,
        start_alignment=start_alignment,
        text_alignment=text_alignment,
        text_alignment_adjust=text_alignment_adjust,
        end_alignment=end_alignment,
        display_text=f"{edge.cost}{assignment_summary}",
        text_size=self.render_config.font_size,
        hover_text=repr(edge),
        style_tags=tuple(style_tags),
    )

  def extra_annotations(self) -> Iterable[dag_annotator.RegionAnnotation]:
    """Produces region and text annotations."""
    for subproblem in self.render_data.subproblem_list:
      left_bound = self.layout_positions[
          "prototype_subproblem_left_x", subproblem.prototype_preorder_index
      ]
      right_bound = self.layout_positions[
          "prototype_subproblem_right_x", subproblem.prototype_preorder_index
      ]
      top_bound = self.layout_positions[
          "target_subproblem_top_y", subproblem.target_preorder_index
      ]
      bottom_bound = self.layout_positions[
          "target_subproblem_bottom_y", subproblem.target_preorder_index
      ]

      yield dag_annotator.RegionAnnotation(
          bounds=rendering.BoundingBox.from_boundary(
              left=left_bound,
              top=top_bound,
              right=right_bound,
              bottom=bottom_bound,
          )
      )

    yield from self.text_annotations

  def renderer_specific_setup(self, renderer: rendering.Renderer) -> None:
    if isinstance(renderer, svg_renderer.SVGRenderer):
      style_css = """\
      rect.unexpected_node {
          fill: red;
      }

      rect.label_in_box.text_token {
          fill: #ddd;
      }
      rect.label_in_box.text_decoration {
          fill: #eee;
          stroke-dasharray: 5 5;
      }

      .edge-annotation {
        --edge-annotation-color: black;
      }
      .edge-annotation.from_HIGH_CONFIDENCE.KEEP {
        --edge-annotation-color: green;
      }
      .edge-annotation.from_HIGH_CONFIDENCE.from_MAY_INSERT,
      .edge-annotation.from_HIGH_CONFIDENCE.to_MAY_INSERT {
        --edge-annotation-color: magenta;
      }
      .edge-annotation.from_HIGH_CONFIDENCE.to_MAY_DELETE,
      .edge-annotation.to_HIGH_CONFIDENCE.DELETE {
        --edge-annotation-color: red;
      }
      .edge-annotation.from_LOW_CONFIDENCE.KEEP {
        --edge-annotation-color: teal;
      }
      .edge-annotation.from_LOW_CONFIDENCE.from_MAY_INSERT,
      .edge-annotation.from_LOW_CONFIDENCE.to_MAY_INSERT  {
        --edge-annotation-color: maroon;
      }
      .edge-annotation.from_LOW_CONFIDENCE.to_MAY_DELETE,
      .edge-annotation.to_LOW_CONFIDENCE.DELETE {
        --edge-annotation-color: orange;
      }
      .edge-annotation.REGION_SHOULD_START.TRUE,
      .edge-annotation.REGION_SHOULD_END.TRUE {
        --edge-annotation-color: #966F33;
      }
      .edge-annotation.REGION_SHOULD_START.FALSE,
      .edge-annotation.REGION_SHOULD_END.FALSE {
        --edge-annotation-color: #151B54;
      }
      .edge-annotation.REGION_SHOULD_START.NOT_APPLICABLE,
      .edge-annotation.REGION_SHOULD_END.NOT_APPLICABLE {
        --edge-annotation-color: #616D7E;
      }
      .edge-annotation.SHOULD_EARLY_EXIT.TRUE {
        --edge-annotation-color: lightblue;
      }
      .edge-annotation path {
        stroke: var(--edge-annotation-color);
      }
      .edge-annotation text {
        fill: var(--edge-annotation-color);
      }
      """
      renderer.configure_style_css(textwrap.dedent(style_css))
