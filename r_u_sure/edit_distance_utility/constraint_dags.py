# coding=utf-8
# Copyright 2025 The R-U-SURE Authors.
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

"""Trust-region constraint DAG.

The "Constraint DAG" is built from the prototype alone, and ensures that
annotated regions are always valid subexpression spans.

Note that a trust-region optimization system only needs one copy of the
constraint DAG, since it depends only on the prototype.
"""

from __future__ import annotations

import dataclasses
import enum
import textwrap
from typing import Any, Callable, Iterable, NamedTuple

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
INVALID_NODE_ID = packed_sequence_nodes.INVALID_NODE_ID


@dataclasses.dataclass
class ConstraintDagRenderConfig:
  """Parameters for rendering the constraint DAG."""

  font_size: float = 12
  state_width: float = 170
  state_height: float = 20
  gap_horizontal: float = 130
  gap_vertical: float = 12 * 5
  state_separation: float = 12 * 3
  edge_gap_horizontal: float = 10


################################################################################
# State types
################################################################################


class ConstraintDagStateCategory(enum.Enum):
  """Categories for states in the constraint DAG."""

  OUTSIDE_ANNOTATED_REGION = enum.auto()
  IN_REGION_TEMPORARY = enum.auto()
  IN_REGION_FORCED = enum.auto()

  SPECIAL_FINAL_STATE = enum.auto()

  # Numba-compatible __hash__ implementation.
  __hash__ = register_enum_hash.jitable_enum_hash

  def shortname(self) -> str:
    """Summarizes a state category."""
    if self == ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION:
      return "Outside"
    elif self == ConstraintDagStateCategory.IN_REGION_TEMPORARY:
      return "In region"
    elif self == ConstraintDagStateCategory.IN_REGION_FORCED:
      return "In region (forced)"
    elif self == ConstraintDagStateCategory.SPECIAL_FINAL_STATE:
      return "Final State"
    else:
      raise ValueError(self)


class ConstraintDagState(NamedTuple):
  """A state in the DAG.

  If we enter a GroupNode in a high-confidence state, we are allowed to move to
  the low confidence state. However, if we enter a GroupNode in a low-confidence
  state, we must stay in a low-confidence state. To disambiguate these states,
  we track the number of ancestor GroupNodes which are in a high-confidence
  state before we reached a low-confidence state.

  Attributes:
    category: Whether we are in an annotated region.
    prototype_sequence_preorder_index: Identifier for the prototype sequence we
      are processing.
    before_prototype_node: Position in that sequence we are currently
      processing.
    outside_region_nesting_level: The number of ancestors of this node that are
      group nodes not contained in an annotated region.
  """

  category: ConstraintDagStateCategory
  prototype_sequence_preorder_index: int
  before_prototype_node: int
  outside_region_nesting_level: int


FINAL_STATE = ConstraintDagState(
    category=ConstraintDagStateCategory.SPECIAL_FINAL_STATE,
    prototype_sequence_preorder_index=packed_sequence_nodes.NO_PREORDER_INDEX,
    before_prototype_node=-1,
    outside_region_nesting_level=-1,
)

################################################################################
# Edge info type
################################################################################


class ConstraintDagEdgeInfo(NamedTuple):
  """Edge info for the constraint dag.

  Attributes:
    is_early_exit: Whether this edge is an early exit edge.
  """

  is_early_exit: bool


EXAMPLE_EDGE_INFO = numba_type_util.PretendOptional(
    ConstraintDagEdgeInfo(is_early_exit=True)
)

################################################################################
# Main logic
################################################################################


EXAMPLE_STATE = ConstraintDagState(
    category=ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
    prototype_sequence_preorder_index=ROOT_PREORDER_INDEX,
    before_prototype_node=0,
    outside_region_nesting_level=0,
)
EXAMPLE_VARIABLE_KEY = DecisionKey(
    prototype_preorder_index=ROOT_PREORDER_INDEX,
    category=DecisionCategory.REGION_SHOULD_START,
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


@numba.extending.register_jitable(inline="always")
def node_in_annotated_region_value(
    category: ConstraintDagStateCategory,
) -> DecisionValue:
  """Map a category to a value for a NODE_IN_REGION assignment."""
  if category == ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION:
    return DecisionValue.FALSE
  else:
    return DecisionValue.TRUE


def make_constraint_dag_builder(
    with_numba: bool = False,
) -> Callable[..., gated_state_dag.CompleteStateDAG,]:
  """Builds either Numba or pure python version of graph construction logic."""
  maybe_jit = numba.njit if with_numba else lambda fn: fn

  def construct_constraint_dag(
      prototype: PackedSequenceNodeStorage,
  ) -> gated_state_dag.CompleteStateDAG:
    """Constructs the constraint DAG.

    Args:
      prototype: Node sequence representing the prototype, which should include
        region nodes (but not early exit nodes).

    Returns:
      The constraint DAG.
    """
    # Construct the graph.
    initial_state = ConstraintDagState(
        category=ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
        prototype_sequence_preorder_index=ROOT_PREORDER_INDEX,
        before_prototype_node=0,
        outside_region_nesting_level=0,
    )
    penultimate_state = ConstraintDagState(
        category=ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
        prototype_sequence_preorder_index=ROOT_PREORDER_INDEX,
        before_prototype_node=len(prototype.root_sequence),
        outside_region_nesting_level=0,
    )

    if with_numba:
      incomplete_graph = gated_state_dag.PartialStateDAG(
          initial_state=initial_state,
          edges=numba.typed.List.empty_list(EDGE_NUMBA_TYPE),
          seen_outgoing=numba.typed.Dict.empty(
              STATE_NUMBA_TYPE, STATE_NUMBA_TYPE
          ),
      )
    else:
      incomplete_graph = gated_state_dag.partial_state_dag_starting_from(
          initial_state
      )

    # Build constraints.
    build_prototype_subgraph_from_outside_region(
        prototype_storage=prototype,
        prototype_node_ids=prototype.root_sequence,
        prototype_sequence_preorder_index=ROOT_PREORDER_INDEX,
        outside_region_nesting_level=0,
        incomplete_graph=incomplete_graph,
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

    return complete_dag

  @maybe_jit
  def build_prototype_subgraph_from_outside_region(
      prototype_storage: PackedSequenceNodeStorage,
      prototype_node_ids: list[PackedSequenceNodeID],
      prototype_sequence_preorder_index: int,
      outside_region_nesting_level: int,
      incomplete_graph: gated_state_dag.PartialStateDAG,
  ):
    """Processes a prototype sequence, starting and ending outside a region.

    When starting outside a region, we are allowed to enter annotated
    regions temporarily, as long as we exit those regions before the end of
    the subsequence.

    This subgraph starts at the state:

      ConstraintDagState(
          category=ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_prototype_node=0,
          outside_region_nesting_level=outside_region_nesting_level,
      )

    and ends at the state:

      ConstraintDagState(
          category=ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_prototype_node=len(prototype_node_ids),
          outside_region_nesting_level=outside_region_nesting_level,
      )

    Args:
      prototype_storage: Storage for the prototype.
      prototype_node_ids: Subsequence of prototype node IDs to process.
      prototype_sequence_preorder_index: Identifier for this subsequence, either
        -1 for the root sequence or the preorder index of the GroupNode.
      outside_region_nesting_level: The current nesting level, e.g. the number
        of GroupNode ancestors that were NOT in annotated regions. Equivalently,
        the number of recursive calls.
      incomplete_graph: Graph that we should extend with this subgraph.
    """

    for prototype_position, prototype_node_id in enumerate(prototype_node_ids):
      # pylint: disable=cell-var-from-loop
      def relative_state(
          category,
          advance_prototype=False,
      ):
        delta_i = 1 if advance_prototype else 0
        return ConstraintDagState(
            category=category,
            before_prototype_node=prototype_position + delta_i,
            prototype_sequence_preorder_index=prototype_sequence_preorder_index,
            outside_region_nesting_level=outside_region_nesting_level,
        )

      # Handle decorations.
      if prototype_node_id.category == PSNCategory.TEXT_DECORATION_NODE:
        for category in (
            ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
            ConstraintDagStateCategory.IN_REGION_TEMPORARY,
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(category=category),
                  dest=relative_state(
                      category=category, advance_prototype=True
                  ),
                  cost=0,
              ),
          )

      # Handle tokens.
      if prototype_node_id.category == PSNCategory.TEXT_TOKEN_NODE:
        for category in (
            ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
            ConstraintDagStateCategory.IN_REGION_TEMPORARY,
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(category=category),
                  dest=relative_state(
                      category=category, advance_prototype=True
                  ),
                  cost=0,
                  required_assignment=SharedVariableAssignment(
                      key=DecisionKey(
                          prototype_node_id.preorder_index,
                          DecisionCategory.NODE_IN_REGION,
                      ),
                      value=node_in_annotated_region_value(category),
                  ),
              ),
          )

      # Handle groups:
      # - If we enter a group outside a region, recursively call this
      #   function to build the subgraph.
      # - If we enter a group in an annotated region, build a forced-region
      #   subgraph instead.
      if prototype_node_id.category == PSNCategory.GROUP_NODE:
        prototype_node = prototype_storage.group_nodes[
            prototype_node_id.index_in_category
        ]
        # ==== HIGH CONFIDENCE ====
        # Recursively call this function, and increase nesting level.
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=relative_state(
                    category=ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION
                ),
                dest=ConstraintDagState(
                    category=ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
                    prototype_sequence_preorder_index=(
                        prototype_node_id.preorder_index
                    ),
                    before_prototype_node=0,
                    outside_region_nesting_level=(
                        outside_region_nesting_level + 1
                    ),
                ),
                cost=0,
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.NODE_IN_REGION,
                    ),
                    value=DecisionValue.FALSE,
                ),
            ),
        )
        build_prototype_subgraph_from_outside_region(
            prototype_storage=prototype_storage,
            prototype_node_ids=prototype_node.children_ids,
            prototype_sequence_preorder_index=prototype_node_id.preorder_index,
            outside_region_nesting_level=outside_region_nesting_level + 1,
            incomplete_graph=incomplete_graph,
        )
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=ConstraintDagState(
                    category=ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
                    prototype_sequence_preorder_index=(
                        prototype_node_id.preorder_index
                    ),
                    before_prototype_node=len(prototype_node.children_ids),
                    outside_region_nesting_level=(
                        outside_region_nesting_level + 1
                    ),
                ),
                dest=relative_state(
                    category=ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
                    advance_prototype=True,
                ),
                cost=0,
            ),
        )
        # ==== LOW CONFIDENCE ====
        # Call forced-low-category helper, and do NOT increase nesting level.
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=relative_state(
                    category=ConstraintDagStateCategory.IN_REGION_TEMPORARY
                ),
                dest=ConstraintDagState(
                    category=ConstraintDagStateCategory.IN_REGION_FORCED,
                    prototype_sequence_preorder_index=(
                        prototype_node_id.preorder_index
                    ),
                    before_prototype_node=0,
                    outside_region_nesting_level=outside_region_nesting_level,
                ),
                cost=0,
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.NODE_IN_REGION,
                    ),
                    value=DecisionValue.TRUE,
                ),
            ),
        )
        build_prototype_subgraph_inside_region(
            prototype_storage=prototype_storage,
            prototype_node_ids=prototype_node.children_ids,
            prototype_sequence_preorder_index=prototype_node_id.preorder_index,
            outside_region_nesting_level=outside_region_nesting_level,
            incomplete_graph=incomplete_graph,
        )
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=ConstraintDagState(
                    category=ConstraintDagStateCategory.IN_REGION_FORCED,
                    prototype_sequence_preorder_index=(
                        prototype_node_id.preorder_index
                    ),
                    before_prototype_node=len(prototype_node.children_ids),
                    outside_region_nesting_level=outside_region_nesting_level,
                ),
                dest=relative_state(
                    category=ConstraintDagStateCategory.IN_REGION_TEMPORARY,
                    advance_prototype=True,
                ),
                cost=0,
            ),
        )

      ###########################################################
      #### Handle annotated regions and early exit ####
      ###########################################################

      if prototype_node_id.category == PSNCategory.REGION_START_NODE:
        for source_conf, dest_conf, decision_value in (
            # Don't start an annotated region.
            (
                ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
                ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
                DecisionValue.FALSE,
            ),
            # Start an annotated region.
            (
                ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
                ConstraintDagStateCategory.IN_REGION_TEMPORARY,
                DecisionValue.TRUE,
            ),
            # Stay in an annotated region.
            (
                ConstraintDagStateCategory.IN_REGION_TEMPORARY,
                ConstraintDagStateCategory.IN_REGION_TEMPORARY,
                DecisionValue.NOT_APPLICABLE,
            ),
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(category=source_conf),
                  dest=relative_state(
                      category=dest_conf,
                      advance_prototype=True,
                  ),
                  required_assignment=SharedVariableAssignment(
                      key=DecisionKey(
                          prototype_node_id.preorder_index,
                          DecisionCategory.REGION_SHOULD_START,
                      ),
                      value=decision_value,
                  ),
                  cost=0,
              ),
          )

      if prototype_node_id.category == PSNCategory.REGION_END_NODE:
        for source_conf, dest_conf, decision_value in (
            # Don't end an annotated region.
            (
                ConstraintDagStateCategory.IN_REGION_TEMPORARY,
                ConstraintDagStateCategory.IN_REGION_TEMPORARY,
                DecisionValue.FALSE,
            ),
            # End an annotated region.
            (
                ConstraintDagStateCategory.IN_REGION_TEMPORARY,
                ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
                DecisionValue.TRUE,
            ),
            # Stay outside of an annotated region.
            (
                ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
                ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
                DecisionValue.NOT_APPLICABLE,
            ),
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(category=source_conf),
                  dest=relative_state(
                      category=dest_conf,
                      advance_prototype=True,
                  ),
                  required_assignment=SharedVariableAssignment(
                      key=DecisionKey(
                          prototype_node_id.preorder_index,
                          DecisionCategory.REGION_SHOULD_END,
                      ),
                      value=decision_value,
                  ),
                  cost=0,
              ),
          )

      if prototype_node_id.category == PSNCategory.EARLY_EXIT_NODE:
        for category in (
            ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION,
            ConstraintDagStateCategory.IN_REGION_TEMPORARY,
        ):
          # Allowed to early exit.
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(category=category),
                  dest=FINAL_STATE,
                  required_assignment=SharedVariableAssignment(
                      key=DecisionKey(
                          prototype_node_id.preorder_index,
                          DecisionCategory.SHOULD_EARLY_EXIT,
                      ),
                      value=DecisionValue.TRUE,
                  ),
                  cost=0.0,
                  info=ConstraintDagEdgeInfo(is_early_exit=True),
              ),
          )
          # Allowed to not early exit.
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(category=category),
                  dest=relative_state(
                      category=category, advance_prototype=True
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
  def build_prototype_subgraph_inside_region(
      prototype_storage: PackedSequenceNodeStorage,
      prototype_node_ids: list[PackedSequenceNodeID],
      prototype_sequence_preorder_index: int,
      outside_region_nesting_level: int,
      incomplete_graph: gated_state_dag.PartialStateDAG,
  ):
    """Processes a prototype sequence, entirely in an annotated region.

    Since we started this subsequence in an annotated region, we are forced to
    remain in an annotated region.

    This subgraph starts at the state:

      ConstraintDagState(
          category=ConstraintDagStateCategory.IN_REGION_FORCED,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_prototype_node=0,
          outside_region_nesting_level=outside_region_nesting_level,
      )

    and ends at the state:

      ConstraintDagState(
          category=ConstraintDagStateCategory.IN_REGION_FORCED,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          before_prototype_node=len(prototype_node_ids),
          outside_region_nesting_level=outside_region_nesting_level,
      )

    Args:
      prototype_storage: Storage for the prototype.
      prototype_node_ids: Subsequence of prototype node IDs to process.
      prototype_sequence_preorder_index: Identifier for this subsequence, either
        -1 for the root sequence or the preorder index of the GroupNode.
      outside_region_nesting_level: The current nesting level, e.g. the number
        of GroupNode ancestors that were outside annotated regions.
        Equivalently, the number of recursive calls.
      incomplete_graph: Graph that we should extend with this subgraph.
    """
    for prototype_position, prototype_node_id in enumerate(prototype_node_ids):
      state_before = ConstraintDagState(
          category=ConstraintDagStateCategory.IN_REGION_FORCED,
          before_prototype_node=prototype_position,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          outside_region_nesting_level=outside_region_nesting_level,
      )
      state_after = ConstraintDagState(
          category=ConstraintDagStateCategory.IN_REGION_FORCED,
          before_prototype_node=prototype_position + 1,
          prototype_sequence_preorder_index=prototype_sequence_preorder_index,
          outside_region_nesting_level=outside_region_nesting_level,
      )

      # Handle decorations.
      if prototype_node_id.category == PSNCategory.TEXT_DECORATION_NODE:
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(source=state_before, dest=state_after, cost=0),
        )

      # Handle tokens.
      if prototype_node_id.category == PSNCategory.TEXT_TOKEN_NODE:
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=state_before,
                dest=state_after,
                cost=0,
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.NODE_IN_REGION,
                    ),
                    value=DecisionValue.TRUE,
                ),
            ),
        )

      # Handle groups (forced in region recursively)
      if prototype_node_id.category == PSNCategory.GROUP_NODE:
        prototype_node = prototype_storage.group_nodes[
            prototype_node_id.index_in_category
        ]
        # Call forced-region helper, and do NOT increase nesting level.
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=state_before,
                dest=ConstraintDagState(
                    category=ConstraintDagStateCategory.IN_REGION_FORCED,
                    prototype_sequence_preorder_index=(
                        prototype_node_id.preorder_index
                    ),
                    before_prototype_node=0,
                    outside_region_nesting_level=outside_region_nesting_level,
                ),
                cost=0,
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.NODE_IN_REGION,
                    ),
                    value=DecisionValue.TRUE,
                ),
            ),
        )
        build_prototype_subgraph_inside_region(
            prototype_storage=prototype_storage,
            prototype_node_ids=prototype_node.children_ids,
            prototype_sequence_preorder_index=prototype_node_id.preorder_index,
            outside_region_nesting_level=outside_region_nesting_level,
            incomplete_graph=incomplete_graph,
        )
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=ConstraintDagState(
                    category=ConstraintDagStateCategory.IN_REGION_FORCED,
                    prototype_sequence_preorder_index=(
                        prototype_node_id.preorder_index
                    ),
                    before_prototype_node=len(prototype_node.children_ids),
                    outside_region_nesting_level=outside_region_nesting_level,
                ),
                dest=state_after,
                cost=0,
            ),
        )

      ###########################################################
      #### Handle annotated regions and early exit ####
      ###########################################################

      # Forced to remain in the annotated region.
      if prototype_node_id.category == PSNCategory.REGION_START_NODE:
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=state_before,
                dest=state_after,
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.REGION_SHOULD_START,
                    ),
                    value=DecisionValue.NOT_APPLICABLE,
                ),
                cost=0,
            ),
        )

      if prototype_node_id.category == PSNCategory.REGION_END_NODE:
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=state_before,
                dest=state_after,
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.REGION_SHOULD_END,
                    ),
                    value=DecisionValue.FALSE,
                ),
                cost=0,
            ),
        )

      if prototype_node_id.category == PSNCategory.EARLY_EXIT_NODE:
        # Allowed to early exit.
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=state_before,
                dest=FINAL_STATE,
                required_assignment=SharedVariableAssignment(
                    key=DecisionKey(
                        prototype_node_id.preorder_index,
                        DecisionCategory.SHOULD_EARLY_EXIT,
                    ),
                    value=DecisionValue.TRUE,
                ),
                cost=0.0,
                info=ConstraintDagEdgeInfo(is_early_exit=True),
            ),
        )
        # Allowed to not early exit.
        gated_state_dag.partial_state_dag_add_edge(
            incomplete_graph,
            Edge(
                source=state_before,
                dest=state_after,
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

  # Return the final builder function.
  return construct_constraint_dag


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
# Rendering annotations
################################################################################

STATE_OFFSET_MAPPING = {
    ConstraintDagStateCategory.IN_REGION_TEMPORARY: 0,
    ConstraintDagStateCategory.IN_REGION_FORCED: 0,
    ConstraintDagStateCategory.OUTSIDE_ANNOTATED_REGION: 1,
    ConstraintDagStateCategory.SPECIAL_FINAL_STATE: 0,
}
NUM_STATE_OFFSETS = 2


class ConstraintDagGraphAnnotator(dag_annotator.StateDAGAnnotator):
  """Annotator for Edit DAGs."""

  prototype: PackedSequenceNodeStorage
  render_config: ConstraintDagRenderConfig
  text_annotations: list[dag_annotator.TextAnnotation]
  layout_positions: dict[Any, Any]

  def __init__(
      self,
      prototype: PackedSequenceNodeStorage,
      render_config: ConstraintDagRenderConfig,
  ):
    """Creates the annotator and precomputes keypoints and global annotations."""
    self.prototype = prototype
    self.render_config = render_config
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
          shift = (
              render_config.gap_vertical
              + render_config.state_height
              + render_config.state_separation
          )
          bounds = rendering.BoundingBox.from_boundary(
              left=left,
              right=right,
              top=-shift - render_config.state_height,
              bottom=-shift,
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
      layout_positions["prototype_subproblem_right_x", preorder_index] = right

      # Final state
      left = current_x
      current_x += render_config.state_width
      right = current_x
      layout_positions["final_state_x"] = (left, right)

    process_prototype(self.prototype.root_sequence, ROOT_PREORDER_INDEX)

  def annotate_state(
      self,
      state: ConstraintDagState,
  ) -> dag_annotator.StateAnnotation:
    """Assigns rendering annotations to a state.

    Args:
      state: The state to assign annotations to.

    Returns:
      Information on how to render this state.
    """
    if state == FINAL_STATE:
      left_bound, right_bound = self.layout_positions["final_state_x"]
      bottom_bound = -self.render_config.gap_vertical
      top_bound = bottom_bound - self.render_config.state_height
    else:
      left_bound, right_bound = self.layout_positions[
          "prototype_state_x",
          state.prototype_sequence_preorder_index,
          state.before_prototype_node,
      ]
      level_offset = (
          2 * self.render_config.state_height
          + self.render_config.state_separation
          + self.render_config.gap_vertical
      ) * state.outside_region_nesting_level
      state_index = STATE_OFFSET_MAPPING[state.category]
      top_bound = level_offset + state_index * (
          self.render_config.state_height + self.render_config.state_separation
      )
      bottom_bound = top_bound + self.render_config.state_height

    return dag_annotator.StateAnnotation(
        bounds=rendering.BoundingBox.from_boundary(
            left=left_bound,
            top=top_bound,
            right=right_bound,
            bottom=bottom_bound,
        ),
        display_text=f"{state.category.shortname()}",
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
    style_tags = []
    if edge.required_assignment:
      assignment_summary = (
          f"\n{edge.required_assignment.key.category.shortname()}:"
          f" {edge.required_assignment.value.shortname()}"
      )
      index = tuple(DecisionValue).index(edge.required_assignment.value)
      elbow_distance = (index + 1) / (len(DecisionValue) + 1)
      style_tags.append(edge.required_assignment.key.category.name)
      style_tags.append(edge.required_assignment.value.name)
    else:
      assignment_summary = ""
      elbow_distance = 0.5

    # Disambiguate offsets based on the opposite edge.
    start_offset = STATE_OFFSET_MAPPING[edge.source.category]
    end_alignment = (start_offset + 1) / (NUM_STATE_OFFSETS + 1)
    end_offset = STATE_OFFSET_MAPPING[edge.dest.category]
    start_alignment = (end_offset + 1) / (NUM_STATE_OFFSETS + 1)

    # Handle inserted edge tracebacks.
    if isinstance(edge.info, dict) and "original_info" in edge.info:
      edge_info = edge.info["original_info"]
    else:
      edge_info = edge.info

    # Override for early exit edges.
    if edge_info is not None and edge_info.is_early_exit:
      elbow_distance = 0
      elbow_distance_adjust = self.render_config.gap_horizontal * 0.2
      text_alignment = 0
      text_alignment_adjust = -0.9 * self.render_config.state_separation
    else:
      elbow_distance_adjust = 0
      text_alignment = 0.5
      text_alignment_adjust = 0

    return dag_annotator.ElbowEdgeAnnotation(
        primary_axis=dag_annotator.ElbowEdgeOrientation.HORIZONTAL,
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
