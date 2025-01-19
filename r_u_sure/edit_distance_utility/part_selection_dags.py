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

"""Part selection DAG builder.

This utility function is designed to extract consecutive spans of the prototype
that are likely to appear in many samples in the same order, independent of
the exact position that those spans occur. For this utility function, each
annotated region represents a selected part, and we get credit for the part
only if that part exactly appears in the target.

This utility function is used to implement the useful-API-calls variant, when
combined with a heuristic that identifies candidate calls.

The DAG builder in module file uses an in-order traversal representation: we
serialize both the prototype and target trees using an in-order traversal, then
identify a contiguous subsequence. Group nodes are represented by "start group"
and "end group" markers, so any subtree or group of sibling subtrees can be
represented by a contiguous set of nodes and start/end group markers.

The DAG builder in this module does NOT ensure that the extracted
parts are valid subexpression spans; that is handled by the "Constraint DAG".

This DAG is incompatible with early exit nodes, and should only be used with
region start/end nodes.
"""
from __future__ import annotations

import dataclasses
import enum
import itertools
import textwrap
from typing import Any, Callable, Iterator, NamedTuple

import numba
from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.decision_diagrams import packed_dags
from r_u_sure.edit_distance_utility import region_decisions
from r_u_sure.numba_helpers import numba_raise
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
InOrderTraversalCategory = packed_sequence_nodes.InOrderTraversalCategory
InOrderTraversalItem = packed_sequence_nodes.InOrderTraversalItem

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
class PartSelectionDagRenderConfig:
  """Parameters for rendering the part selection DAG."""

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
    match_utility: Utility of correctly predicting this node if it is in a
      selected part.
    delete_cost: Cost of deleting this node if it is in a selected part.
  """

  match_utility: Cost
  delete_cost: Cost


class PartSelectionUtilityParameters(NamedTuple):
  """Parameters for the part selection utility.

  Attributes:
    token_node_utility_fn: Function computing utilities for a token node.
  """

  token_node_utility_fn: Callable[
      [packed_sequence_nodes.PackedTextTokenNode], UtilitiesAndCostsForNode
  ]


################################################################################
# State types
################################################################################


class DagStateCategory(enum.Enum):
  """Categories for states in the part selection DAG."""

  # We are skipping unselected nodes in the prototype.
  # We can skip a prototype node and stay in this state, or, if we are at a
  # region start node, transition to SELECT_UNMATCHED or SKIP_IN_TARGET.
  # Once we reach the end of the prototype, we finish the DAG.
  SKIP_UNSELECTED_IN_PROTOTYPE = enum.auto()
  # We have committed to select some nodes in the prototype, and we are now
  # skipping unmatched nodes in the target. We can skip target nodes and
  # stay in this state, or transition to SELECT_MATCHED.
  SKIP_IN_TARGET = enum.auto()
  # We are matching selected nodes in the prototype with nodes in the target.
  # When we reach a region end node, we can transition back to
  # SKIP_UNSELECTED_IN_PROTOTYPE.
  SELECT_MATCHED = enum.auto()
  # We are processing selected but not matched nodes in the prototype. We stay
  # in this state until reaching a region end node, at which point we can switch
  # back to SKIP_UNSELECTED_IN_PROTOTYPE.
  SELECT_UNMATCHED = enum.auto()

  # One extra final state; we jump here after finishing the prototype.
  SPECIAL_FINAL_STATE = enum.auto()

  # Numba-compatible __hash__ implementation.
  __hash__ = register_enum_hash.jitable_enum_hash

  def shortname(self) -> str:
    """Summarizes a state category."""
    if self == DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE:
      return "Skip Prototype"
    if self == DagStateCategory.SKIP_IN_TARGET:
      return "Skip Target"
    elif self == DagStateCategory.SELECT_UNMATCHED:
      return "Unmatched"
    elif self == DagStateCategory.SELECT_MATCHED:
      return "Matched"
    elif self == DagStateCategory.SPECIAL_FINAL_STATE:
      return "Final State"
    else:
      raise ValueError(self)


class PartSelectionDagState(NamedTuple):
  """A state in the DAG.

  Attributes:
    category: The type of state this is.
    prototype_in_order_traversal_before: Index in an in-order traversal of the
      prototype node that we are currently before, or -1 if not applicable.
    target_in_order_traversal_before: Index in an in-order traversal of the
      target node that we are currently before, or -1 if not applicable..
  """

  category: DagStateCategory
  prototype_in_order_traversal_before: int
  target_in_order_traversal_before: int


INITIAL_STATE = PartSelectionDagState(
    category=DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE,
    prototype_in_order_traversal_before=0,
    target_in_order_traversal_before=0,
)
FINAL_STATE = PartSelectionDagState(
    category=DagStateCategory.SPECIAL_FINAL_STATE,
    prototype_in_order_traversal_before=-1,
    target_in_order_traversal_before=-1,
)

################################################################################
# Edge info type
################################################################################


class PartSelectionDagEdgeInfo(NamedTuple):
  """Edge info for the edit dag.

  Edge info is only provided for edges that advance in either the prototype or
  the target.

  Attributes:
    category: Relevant category for the edge.
    prototype_node_preorder_index: Preorder index of the prototype node
      corresponding to this transition, if applicable.
    target_node_preorder_index: Preorder index of the target node corresponding
      to this transition, if applicable.
  """

  category: DagStateCategory
  prototype_node_preorder_index: int
  target_node_preorder_index: int


EXAMPLE_EDGE_INFO = numba_type_util.PretendOptional(
    PartSelectionDagEdgeInfo(
        category=DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE,
        prototype_node_preorder_index=NO_PREORDER_INDEX,
        target_node_preorder_index=NO_PREORDER_INDEX,
    )
)


################################################################################
# Main logic
################################################################################

EXAMPLE_VARIABLE_KEY = DecisionKey(
    prototype_preorder_index=ROOT_PREORDER_INDEX,
    category=DecisionCategory.REGION_SHOULD_START,
)

EXAMPLE_STATE = FINAL_STATE
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


def make_selection_dag_builder(
    parameters: PartSelectionUtilityParameters,
    with_numba: bool = False,
) -> Callable[..., gated_state_dag.CompleteStateDAG]:
  """Builds either Numba or pure python version of graph construction logic."""
  maybe_jit = numba.njit if with_numba else lambda fn: fn

  token_node_utility_fn = parameters.token_node_utility_fn

  def construct_selection_dag(
      prototype: PackedSequenceNodeStorage,
      target: PackedSequenceNodeStorage,
  ) -> gated_state_dag.CompleteStateDAG:
    """Constructs the selection DAG.

    Args:
      prototype: Node sequence representing the prototype, which should include
        region nodes (but not early exit nodes).
      target: Node sequence representing a possible end state of the user's
        code, which should NOT include region nodes (or early exit nodes).

    Returns:
      The DAG.
    """
    if with_numba:
      incomplete_graph = gated_state_dag.PartialStateDAG(
          initial_state=INITIAL_STATE,
          edges=numba.typed.List.empty_list(EDGE_NUMBA_TYPE),
          seen_outgoing=numba.typed.Dict.empty(
              STATE_NUMBA_TYPE, STATE_NUMBA_TYPE
          ),
      )
    else:
      incomplete_graph = gated_state_dag.partial_state_dag_starting_from(
          INITIAL_STATE
      )

    # Construct.
    _construct_jitted(prototype, target, incomplete_graph)

    # Finish the graph.
    # Finish the graph.
    complete_dag = gated_state_dag.partial_state_dag_finish(
        incomplete_graph, FINAL_STATE
    )

    return complete_dag

  @maybe_jit
  def _construct_jitted(
      prototype: PackedSequenceNodeStorage,
      target: PackedSequenceNodeStorage,
      incomplete_graph: gated_state_dag.PartialStateDAG,
  ):
    """Jit-compatible inner construction helper."""
    prototype_in_order = packed_sequence_nodes.in_order_traversal(
        prototype, strip_decorations=True
    )
    target_in_order_extended = packed_sequence_nodes.in_order_traversal(
        target, strip_decorations=True
    )
    # Add an extra item to the target, so that we continue processing prototype
    # nodes even if we have processed everything in the target.
    target_in_order_extended.append(
        InOrderTraversalItem(
            category=InOrderTraversalCategory.LEAF,
            node_id=INVALID_NODE_ID,
        )
    )

    for prototype_position, prototype_visit in enumerate(prototype_in_order):
      prototype_node_id = prototype_visit.node_id
      for target_position, target_visit in enumerate(target_in_order_extended):
        target_node_id = target_visit.node_id

        # Position-relative helper to save on boilerplate
        # pylint: disable=cell-var-from-loop
        def relative_state(
            category,
            advance_prototype=False,
            advance_target=False,
        ):
          delta_i = 1 if advance_prototype else 0
          delta_j = 1 if advance_target else 0
          return PartSelectionDagState(
              category=category,
              prototype_in_order_traversal_before=prototype_position + delta_i,
              target_in_order_traversal_before=target_position + delta_j,
          )

        # pylint: enable=cell-var-from-loop

        ######################################################################
        #### Skip prototype nodes (in SKIP_UNSELECTED_IN_PROTOTYPE)
        ######################################################################

        if prototype_node_id.category == PSNCategory.TEXT_TOKEN_NODE:
          # Can skip token nodes. Need to store metadata for them.
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          can_skip_prototype = True
          required_assignment = SharedVariableAssignment(
              key=DecisionKey(
                  category=DecisionCategory.NODE_IN_REGION,
                  prototype_preorder_index=prototype_node_id.preorder_index,
              ),
              value=DecisionValue.FALSE,
          )
        elif prototype_node_id.category == PSNCategory.GROUP_NODE:
          # Can skip group start/end virtual nodes. Need to store metadata for
          # group starts.
          can_skip_prototype = True
          if prototype_visit.category == InOrderTraversalCategory.BEFORE_GROUP:
            required_assignment = SharedVariableAssignment(
                key=DecisionKey(
                    category=DecisionCategory.NODE_IN_REGION,
                    prototype_preorder_index=prototype_node_id.preorder_index,
                ),
                value=DecisionValue.FALSE,
            )
          elif prototype_visit.category == InOrderTraversalCategory.AFTER_GROUP:
            required_assignment = None
          else:
            numba_raise.safe_raise(
                ValueError,
                ("Invalid visit type for group node:", prototype_visit),
            )
        elif prototype_node_id.category == PSNCategory.REGION_END_NODE:
          # Can skip region end nodes, since we aren't in a selected region.
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          can_skip_prototype = True
          required_assignment = SharedVariableAssignment(
              key=DecisionKey(
                  category=DecisionCategory.REGION_SHOULD_END,
                  prototype_preorder_index=prototype_node_id.preorder_index,
              ),
              value=DecisionValue.NOT_APPLICABLE,
          )
        elif prototype_node_id.category == PSNCategory.REGION_START_NODE:
          # Can skip region start nodes if we decide not to start a region.
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          can_skip_prototype = True
          required_assignment = SharedVariableAssignment(
              key=DecisionKey(
                  category=DecisionCategory.REGION_SHOULD_START,
                  prototype_preorder_index=prototype_node_id.preorder_index,
              ),
              value=DecisionValue.FALSE,
          )
        else:
          can_skip_prototype = False
          required_assignment = None

        if can_skip_prototype:
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(
                      DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE
                  ),
                  dest=relative_state(
                      DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE,
                      advance_prototype=True,
                  ),
                  cost=0,
                  required_assignment=required_assignment,
                  info=PartSelectionDagEdgeInfo(
                      category=DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE,
                      prototype_node_preorder_index=(
                          prototype_node_id.preorder_index
                      ),
                      target_node_preorder_index=-1,
                  ),
              ),
          )

        ######################################################################
        #### Handle region start
        #### (SKIP_UNSELECTED_IN_PROTOTYPE -> SKIP_IN_TARGET,
        ####  SKIP_UNSELECTED_IN_PROTOTYPE -> SELECT_UNMATCHED)
        ######################################################################

        if prototype_node_id.category == PSNCategory.REGION_START_NODE:
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          decision_key = DecisionKey(
              category=DecisionCategory.REGION_SHOULD_START,
              prototype_preorder_index=(prototype_node_id.preorder_index),
          )
          # Note: SKIP_UNSELECTED_IN_PROTOTYPE -> SKIP_UNSELECTED_IN_PROTOTYPE
          # transition is handled in the above section.

          # Can choose to start a selected region, and start skipping in
          # the target. We also advance past the region start node here.
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(
                      DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE
                  ),
                  dest=relative_state(
                      DagStateCategory.SKIP_IN_TARGET,
                      advance_prototype=True,
                  ),
                  cost=0,
                  required_assignment=SharedVariableAssignment(
                      key=decision_key,
                      value=DecisionValue.TRUE,
                  ),
              ),
          )
          # Can choose to start a selected region without finding any matched
          # part of the target; we will pay penalties for this later.
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(
                      DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE
                  ),
                  dest=relative_state(
                      DagStateCategory.SELECT_UNMATCHED,
                      advance_prototype=True,
                  ),
                  cost=0,
                  required_assignment=SharedVariableAssignment(
                      key=decision_key,
                      value=DecisionValue.TRUE,
                  ),
              ),
          )

        ######################################################################
        #### Skip target nodes (in SKIP_IN_TARGET)
        ######################################################################

        # These are the only node types we expect to see.
        if (
            target_node_id.category == PSNCategory.TEXT_TOKEN_NODE
            or target_node_id.category == PSNCategory.GROUP_NODE
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(DagStateCategory.SKIP_IN_TARGET),
                  dest=relative_state(
                      DagStateCategory.SKIP_IN_TARGET,
                      advance_target=True,
                  ),
                  cost=0,
                  required_assignment=None,
                  info=PartSelectionDagEdgeInfo(
                      category=DagStateCategory.SKIP_IN_TARGET,
                      prototype_node_preorder_index=-1,
                      target_node_preorder_index=(
                          target_node_id.preorder_index
                      ),
                  ),
              ),
          )

        ######################################################################
        #### Start matching (SKIP_IN_TARGET -> SELECT_MATCHED)
        ######################################################################

        # We can decide to start matching at any point from SKIP_IN_TARGET.
        # Note that we only enter SKIP_IN_TARGET at region start nodes in the
        # prototype, and this will immediately fail unless we are actually
        # at compatible places in the two nodes.
        if (
            target_node_id.category == PSNCategory.TEXT_TOKEN_NODE
            or target_node_id.category == PSNCategory.GROUP_NODE
        ):
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(DagStateCategory.SKIP_IN_TARGET),
                  dest=relative_state(DagStateCategory.SELECT_MATCHED),
                  cost=0,
                  required_assignment=None,
              ),
          )

        ######################################################################
        #### Match nodes (in SELECT_MATCHED)
        ######################################################################

        # Match tokens.
        if (
            prototype_node_id.category == PSNCategory.TEXT_TOKEN_NODE
            and target_node_id.category == PSNCategory.TEXT_TOKEN_NODE
        ):
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          assert target_visit.category == InOrderTraversalCategory.LEAF
          prototype_node = prototype.text_token_nodes[
              prototype_node_id.index_in_category
          ]
          target_node = target.text_token_nodes[
              target_node_id.index_in_category
          ]
          if (
              prototype_node.text_contents == target_node.text_contents
              and prototype_node.match_type == target_node.match_type
          ):
            # Utility for a correct match!
            token_match_utility = token_node_utility_fn(
                prototype_node
            ).match_utility
            gated_state_dag.partial_state_dag_add_edge(
                incomplete_graph,
                Edge(
                    source=relative_state(DagStateCategory.SELECT_MATCHED),
                    dest=relative_state(
                        DagStateCategory.SELECT_MATCHED,
                        advance_prototype=True,
                        advance_target=True,
                    ),
                    cost=-token_match_utility,
                    required_assignment=SharedVariableAssignment(
                        key=DecisionKey(
                            category=DecisionCategory.NODE_IN_REGION,
                            prototype_preorder_index=(
                                prototype_node_id.preorder_index
                            ),
                        ),
                        value=DecisionValue.TRUE,
                    ),
                    info=PartSelectionDagEdgeInfo(
                        category=DagStateCategory.SELECT_MATCHED,
                        prototype_node_preorder_index=(
                            prototype_node_id.preorder_index
                        ),
                        target_node_preorder_index=(
                            target_node_id.preorder_index
                        ),
                    ),
                ),
            )

        # Match group markers. Note that we will visit each group node twice;
        # once with the BEGIN_GROUP category and once with the END_GROUP
        # category.
        if (
            prototype_node_id.category == PSNCategory.GROUP_NODE
            and target_node_id.category == PSNCategory.GROUP_NODE
            and prototype_visit.category == target_visit.category
        ):
          prototype_node = prototype.group_nodes[
              prototype_node_id.index_in_category
          ]
          target_node = target.group_nodes[target_node_id.index_in_category]
          if prototype_node.match_type == target_node.match_type:
            if (
                prototype_visit.category
                == InOrderTraversalCategory.BEFORE_GROUP
            ):
              required_assignment = SharedVariableAssignment(
                  key=DecisionKey(
                      category=DecisionCategory.NODE_IN_REGION,
                      prototype_preorder_index=(
                          prototype_node_id.preorder_index
                      ),
                  ),
                  value=DecisionValue.TRUE,
              )
            else:
              required_assignment = None
            gated_state_dag.partial_state_dag_add_edge(
                incomplete_graph,
                Edge(
                    source=relative_state(DagStateCategory.SELECT_MATCHED),
                    dest=relative_state(
                        DagStateCategory.SELECT_MATCHED,
                        advance_prototype=True,
                        advance_target=True,
                    ),
                    cost=0,
                    required_assignment=required_assignment,
                    info=PartSelectionDagEdgeInfo(
                        category=DagStateCategory.SELECT_MATCHED,
                        prototype_node_preorder_index=(
                            prototype_node_id.preorder_index
                        ),
                        target_node_preorder_index=(
                            target_node_id.preorder_index
                        ),
                    ),
                ),
            )

        # We can skip over any region start decisions we encounter, since
        # we are already in a selected region.
        if prototype_node_id.category == PSNCategory.REGION_START_NODE:
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(DagStateCategory.SELECT_MATCHED),
                  dest=relative_state(
                      DagStateCategory.SELECT_MATCHED,
                      advance_prototype=True,
                  ),
                  cost=0,
                  required_assignment=SharedVariableAssignment(
                      key=DecisionKey(
                          category=DecisionCategory.REGION_SHOULD_START,
                          prototype_preorder_index=(
                              prototype_node_id.preorder_index
                          ),
                      ),
                      value=DecisionValue.NOT_APPLICABLE,
                  ),
                  info=PartSelectionDagEdgeInfo(
                      category=DagStateCategory.SELECT_MATCHED,
                      prototype_node_preorder_index=(
                          prototype_node_id.preorder_index
                      ),
                      target_node_preorder_index=-1,
                  ),
              ),
          )

        ######################################################################
        #### Finish matching? (SELECT_MATCHED -> SKIP_UNSELECTED_IN_PROTOTYPE)
        ####                  (SELECT_MATCHED -> SELECT_MATCHED)
        ######################################################################

        if prototype_node_id.category == PSNCategory.REGION_END_NODE:
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          decision_key = DecisionKey(
              category=DecisionCategory.REGION_SHOULD_END,
              prototype_preorder_index=prototype_node_id.preorder_index,
          )
          # Can finish matching here.
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(DagStateCategory.SELECT_MATCHED),
                  dest=relative_state(
                      DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE,
                      advance_prototype=True,
                  ),
                  cost=0,
                  required_assignment=SharedVariableAssignment(
                      key=decision_key,
                      value=DecisionValue.TRUE,
                  ),
              ),
          )
          # Can choose to continue matching.
          # Note that we don't advance in the target.
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(DagStateCategory.SELECT_MATCHED),
                  dest=relative_state(
                      DagStateCategory.SELECT_MATCHED,
                      advance_prototype=True,
                  ),
                  cost=0,
                  required_assignment=SharedVariableAssignment(
                      key=decision_key,
                      value=DecisionValue.FALSE,
                  ),
                  info=PartSelectionDagEdgeInfo(
                      category=DagStateCategory.SELECT_MATCHED,
                      prototype_node_preorder_index=(
                          prototype_node_id.preorder_index
                      ),
                      target_node_preorder_index=-1,
                  ),
              ),
          )

        ######################################################################
        #### Process unmatched prototype nodes (in SELECT_UNMATCHED)
        ######################################################################

        if prototype_node_id.category == PSNCategory.TEXT_TOKEN_NODE:
          # Can pass token nodes. Need to store metadata for them.
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          can_advance_unmatched = True
          prototype_node = prototype.text_token_nodes[
              prototype_node_id.index_in_category
          ]
          cost = token_node_utility_fn(prototype_node).delete_cost
          required_assignment = SharedVariableAssignment(
              key=DecisionKey(
                  category=DecisionCategory.NODE_IN_REGION,
                  prototype_preorder_index=prototype_node_id.preorder_index,
              ),
              value=DecisionValue.TRUE,
          )
        elif prototype_node_id.category == PSNCategory.GROUP_NODE:
          # Can pass group start/end virtual nodes. Need to store metadata for
          # group starts.
          can_advance_unmatched = True
          cost = 0.0
          if prototype_visit.category == InOrderTraversalCategory.BEFORE_GROUP:
            required_assignment = SharedVariableAssignment(
                key=DecisionKey(
                    category=DecisionCategory.NODE_IN_REGION,
                    prototype_preorder_index=prototype_node_id.preorder_index,
                ),
                value=DecisionValue.TRUE,
            )
          elif prototype_visit.category == InOrderTraversalCategory.AFTER_GROUP:
            required_assignment = None
          else:
            numba_raise.safe_raise(
                ValueError,
                ("Invalid visit type for group node:", prototype_visit),
            )
        elif prototype_node_id.category == PSNCategory.REGION_START_NODE:
          # Can advance past region start nodes.
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          can_advance_unmatched = True
          cost = 0.0
          required_assignment = SharedVariableAssignment(
              key=DecisionKey(
                  category=DecisionCategory.REGION_SHOULD_START,
                  prototype_preorder_index=prototype_node_id.preorder_index,
              ),
              value=DecisionValue.NOT_APPLICABLE,
          )
        elif prototype_node_id.category == PSNCategory.REGION_END_NODE:
          # Can skip region end nodes if we decide not to stop the region.
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          can_skip_prototype = True
          cost = 0.0
          required_assignment = SharedVariableAssignment(
              key=DecisionKey(
                  category=DecisionCategory.REGION_SHOULD_END,
                  prototype_preorder_index=prototype_node_id.preorder_index,
              ),
              value=DecisionValue.FALSE,
          )
        else:
          can_advance_unmatched = False
          required_assignment = None
          cost = 0.0

        if can_advance_unmatched:
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(DagStateCategory.SELECT_UNMATCHED),
                  dest=relative_state(
                      DagStateCategory.SELECT_UNMATCHED,
                      advance_prototype=True,
                  ),
                  cost=cost,
                  required_assignment=required_assignment,
                  info=PartSelectionDagEdgeInfo(
                      category=DagStateCategory.SELECT_UNMATCHED,
                      prototype_node_preorder_index=(
                          prototype_node_id.preorder_index
                      ),
                      target_node_preorder_index=-1,
                  ),
              ),
          )

        ######################################################################
        #### Finish unmatched selection
        #### (SELECT_UNMATCHED -> SKIP_UNSELECTED_IN_PROTOTYPE)
        ######################################################################

        if prototype_node_id.category == PSNCategory.REGION_END_NODE:
          assert prototype_visit.category == InOrderTraversalCategory.LEAF
          # Note: if we decide not to finish the region, this is handled in
          # the above section.

          # Can choose to end a selected region here.
          gated_state_dag.partial_state_dag_add_edge(
              incomplete_graph,
              Edge(
                  source=relative_state(DagStateCategory.SELECT_UNMATCHED),
                  dest=relative_state(
                      DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE,
                      advance_prototype=True,
                  ),
                  cost=0,
                  required_assignment=SharedVariableAssignment(
                      key=DecisionKey(
                          category=DecisionCategory.REGION_SHOULD_END,
                          prototype_preorder_index=(
                              prototype_node_id.preorder_index
                          ),
                      ),
                      value=DecisionValue.TRUE,
                  ),
              ),
          )

    ######################################################################
    #### Transition to final state
    ######################################################################
    # We can transition to the final state from any position in the target,
    # as long as we are at the end of the prototype.
    prototype_position = len(prototype_in_order)
    for target_position in range(len(target_in_order_extended)):
      gated_state_dag.partial_state_dag_add_edge(
          incomplete_graph,
          Edge(
              source=PartSelectionDagState(
                  category=DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE,
                  prototype_in_order_traversal_before=prototype_position,
                  target_in_order_traversal_before=target_position,
              ),
              dest=FINAL_STATE,
              cost=0,
          ),
      )

  # We can now return our top-level builder function.
  return construct_selection_dag


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


@dataclasses.dataclass
class PartSelectionGroup:
  """A grouped part of a path."""

  content_tokens: list[str]
  category: DagStateCategory
  cost: float


def extract_match_groups(
    path: list[Edge],
    prototype: PackedSequenceNodeStorage,
    target: PackedSequenceNodeStorage,
) -> Iterator[PartSelectionGroup]:
  """Groups parts of a path based on their category.

  Args:
    path: Path through an edit DAG, representing an edit sequence.
    prototype: The prototype node for this DAG.
    target: The target node for this DAG.

  Yields:
    Part selection groups.
  """

  def _get_category(edge):
    if edge.info:
      return edge.info.category
    else:
      return None

  for subpart_category, subpart in itertools.groupby(path, _get_category):
    content_tokens = []
    cost = 0.0
    for edge in subpart:
      cost += edge.cost
      if subpart_category in (
          DagStateCategory.SELECT_MATCHED,
          DagStateCategory.SELECT_UNMATCHED,
          DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE,
      ):
        prototype_preorder_index = edge.info.prototype_node_preorder_index
        if prototype_preorder_index != -1:
          prototype_node_id = prototype.preorder_traversal[
              prototype_preorder_index
          ]
          if prototype_node_id.category == PSNCategory.TEXT_TOKEN_NODE:
            prototype_node = prototype.text_token_nodes[
                prototype_node_id.index_in_category
            ]
            content_tokens.append(prototype_node.text_contents)
      elif subpart_category == DagStateCategory.SKIP_IN_TARGET:
        target_preorder_index = edge.info.target_node_preorder_index
        if target_preorder_index != -1:
          target_node_id = target.preorder_traversal[target_preorder_index]
          if target_node_id.category == PSNCategory.TEXT_TOKEN_NODE:
            target_node = target.text_token_nodes[
                target_node_id.index_in_category
            ]
            content_tokens.append(target_node.text_contents)
    if cost or content_tokens:
      yield PartSelectionGroup(
          content_tokens=content_tokens,
          category=subpart_category,
          cost=cost,
      )


################################################################################
# Rendering annotations
################################################################################

STATE_OFFSET_MAPPING = {
    DagStateCategory.SKIP_UNSELECTED_IN_PROTOTYPE: 0,
    DagStateCategory.SKIP_IN_TARGET: 1,
    DagStateCategory.SELECT_MATCHED: 2,
    DagStateCategory.SELECT_UNMATCHED: 3,
    DagStateCategory.SPECIAL_FINAL_STATE: 0,
}
NUM_STATE_OFFSETS = max(STATE_OFFSET_MAPPING.values()) + 1


class PartSelectionGraphAnnotator(dag_annotator.StateDAGAnnotator):
  """Annotator for part selection DAGs."""

  prototype: PackedSequenceNodeStorage
  target: PackedSequenceNodeStorage
  render_config: PartSelectionDagRenderConfig
  text_annotations: list[dag_annotator.TextAnnotation]
  layout_positions: dict[Any, Any]
  prototype_in_order: list[packed_sequence_nodes.InOrderTraversalItem]
  target_in_order: list[packed_sequence_nodes.InOrderTraversalItem]

  def __init__(
      self,
      prototype: PackedSequenceNodeStorage,
      target: PackedSequenceNodeStorage,
      render_config: PartSelectionDagRenderConfig,
  ):
    """Creates the annotator and precomputes keypoints and global annotations."""
    self.prototype = prototype
    self.target = target
    self.render_config = render_config
    self.text_annotations = []
    self.layout_positions = {}

    self.precompute_layout()

  def precompute_layout(self):
    """Precomputes locations for rendering."""
    render_config = self.render_config
    # Without a type annotation, pytype tends to infer an incorrect type.
    layout_positions: dict[Any, Any] = self.layout_positions

    self.prototype_in_order = packed_sequence_nodes.in_order_traversal(
        self.prototype, strip_decorations=True
    )
    self.target_in_order = packed_sequence_nodes.in_order_traversal(
        self.target, strip_decorations=True
    )

    # Offsets.
    layout_positions["y_offset_prototype_text"] = 0
    layout_positions["y_offset_matching"] = (
        render_config.state_height + render_config.gap_vertical
    )

    # Offsets and text annotations for prototype.
    current_x = 0

    for prototype_position, prototype_visit in enumerate(
        self.prototype_in_order
    ):
      node_id = prototype_visit.node_id

      state_left = current_x
      current_x += render_config.state_width
      state_right = current_x
      layout_positions["prototype_state_x", prototype_position] = (
          state_left,
          state_right,
      )

      text_left = current_x
      current_x += render_config.gap_horizontal
      text_right = current_x

      bounds = rendering.BoundingBox(
          left=text_left,
          top=layout_positions["y_offset_prototype_text"],
          width=(text_right - text_left),
          height=render_config.state_height,
      )
      if (
          prototype_visit.category == InOrderTraversalCategory.LEAF
          and node_id.category == PSNCategory.TEXT_TOKEN_NODE
      ):
        node = self.prototype.text_token_nodes[node_id.index_in_category]
        self.text_annotations.append(
            dag_annotator.TextAnnotation(
                bounds=bounds,
                display_text=repr(node.text_contents)[1:-1],
                text_size=self.render_config.font_size,
                hover_text=repr((prototype_visit, node)),
                style_tags=("text_token",),
            )
        )
      elif (
          prototype_visit.category == InOrderTraversalCategory.LEAF
          and node_id.category == PSNCategory.REGION_START_NODE
      ):
        self.text_annotations.append(
            dag_annotator.TextAnnotation(
                bounds=bounds,
                display_text="Region Start",
                text_size=self.render_config.font_size,
                hover_text=repr(prototype_visit),
                style_tags=("region_start",),
            )
        )
      elif (
          prototype_visit.category == InOrderTraversalCategory.LEAF
          and node_id.category == PSNCategory.REGION_END_NODE
      ):
        self.text_annotations.append(
            dag_annotator.TextAnnotation(
                bounds=bounds,
                display_text="Region End",
                text_size=self.render_config.font_size,
                hover_text=repr(prototype_visit),
                style_tags=("region_end",),
            )
        )
      elif (
          prototype_visit.category == InOrderTraversalCategory.BEFORE_GROUP
          and node_id.category == PSNCategory.GROUP_NODE
      ):
        self.text_annotations.append(
            dag_annotator.TextAnnotation(
                bounds=bounds,
                display_text="Group Start",
                text_size=self.render_config.font_size,
                hover_text=repr(prototype_visit),
                style_tags=("group_start",),
            )
        )
      elif (
          prototype_visit.category == InOrderTraversalCategory.AFTER_GROUP
          and node_id.category == PSNCategory.GROUP_NODE
      ):
        self.text_annotations.append(
            dag_annotator.TextAnnotation(
                bounds=bounds,
                display_text="Group End",
                text_size=self.render_config.font_size,
                hover_text=repr(prototype_visit),
                style_tags=("group_end",),
            )
        )
      else:
        self.text_annotations.append(
            dag_annotator.TextAnnotation(
                bounds=bounds,
                display_text=repr(prototype_visit),
                text_size=self.render_config.font_size,
                hover_text=repr(prototype_visit),
                style_tags=("unexpected_node",),
            )
        )

    # (after prototype)
    state_left = current_x
    current_x += render_config.state_width
    state_right = current_x
    layout_positions["prototype_state_x", len(self.prototype_in_order)] = (
        state_left,
        state_right,
    )
    current_x += render_config.gap_horizontal
    state_left = current_x
    current_x += render_config.state_width
    state_right = current_x
    layout_positions["prototype_state_x", len(self.prototype_in_order) + 1] = (
        state_left,
        state_right,
    )

    # Offsets and text annotations for target.
    current_y = layout_positions["y_offset_matching"]

    for target_position, target_visit in enumerate(self.target_in_order):
      node_id = target_visit.node_id

      for offset in range(NUM_STATE_OFFSETS):
        state_top = current_y
        current_y += render_config.state_height
        state_bottom = current_y
        layout_positions["target_state_y", target_position, offset] = (
            state_top,
            state_bottom,
        )
        current_y += render_config.state_separation
      current_y -= render_config.state_separation

      text_top = current_y
      current_y += render_config.gap_vertical
      text_bottom = current_y

      bounds = rendering.BoundingBox(
          left=-(render_config.state_width + render_config.gap_horizontal),
          top=text_top,
          width=render_config.state_width,
          height=(text_bottom - text_top),
      )
      if (
          target_visit.category == InOrderTraversalCategory.LEAF
          and node_id.category == PSNCategory.TEXT_TOKEN_NODE
      ):
        node = self.target.text_token_nodes[node_id.index_in_category]
        self.text_annotations.append(
            dag_annotator.TextAnnotation(
                bounds=bounds,
                display_text=repr(node.text_contents)[1:-1],
                text_size=self.render_config.font_size,
                hover_text=repr((target_visit, node)),
                style_tags=("text_token",),
            )
        )
      elif (
          target_visit.category == InOrderTraversalCategory.BEFORE_GROUP
          and node_id.category == PSNCategory.GROUP_NODE
      ):
        self.text_annotations.append(
            dag_annotator.TextAnnotation(
                bounds=bounds,
                display_text="Group Start",
                text_size=self.render_config.font_size,
                hover_text=repr(target_visit),
                style_tags=("group_start",),
            )
        )
      elif (
          target_visit.category == InOrderTraversalCategory.AFTER_GROUP
          and node_id.category == PSNCategory.GROUP_NODE
      ):
        self.text_annotations.append(
            dag_annotator.TextAnnotation(
                bounds=bounds,
                display_text="Group End",
                text_size=self.render_config.font_size,
                hover_text=repr(target_visit),
                style_tags=("group_end",),
            )
        )
      else:
        self.text_annotations.append(
            dag_annotator.TextAnnotation(
                bounds=bounds,
                display_text=repr(target_visit),
                text_size=self.render_config.font_size,
                hover_text=repr(target_visit),
                style_tags=("unexpected_node",),
            )
        )

    for offset in range(NUM_STATE_OFFSETS):
      state_top = current_y
      current_y += render_config.state_height
      state_bottom = current_y
      layout_positions["target_state_y", len(self.target_in_order), offset] = (
          state_top,
          state_bottom,
      )
      current_y += render_config.state_separation

  def annotate_state(
      self,
      state: PartSelectionDagState,
  ) -> dag_annotator.StateAnnotation:
    """Assigns rendering annotations to a state.

    Args:
      state: The state to assign annotations to.

    Returns:
      Information on how to render this state.
    """
    if state == FINAL_STATE:
      left_bound, right_bound = self.layout_positions[
          "prototype_state_x", len(self.prototype_in_order) + 1
      ]
      top_bound, bottom_bound = self.layout_positions["target_state_y", 0, 0]
    else:
      left_bound, right_bound = self.layout_positions[
          "prototype_state_x", state.prototype_in_order_traversal_before
      ]
      offset = STATE_OFFSET_MAPPING[state.category]
      top_bound, bottom_bound = self.layout_positions[
          "target_state_y", state.target_in_order_traversal_before, offset
      ]

    display_text = state.category.shortname()
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
        f"from_{edge.source.category.name}",
        f"to_{edge.dest.category.name}",
    ]

    # Disambiguate offsets based on the opposite edge.
    start_offset = STATE_OFFSET_MAPPING[edge.source.category]
    end_alignment = (start_offset + 1) / (NUM_STATE_OFFSETS + 1)
    end_offset = STATE_OFFSET_MAPPING[edge.dest.category]
    start_alignment = (end_offset + 1) / (NUM_STATE_OFFSETS + 1)

    # Try to align vertical components in a meaningful way.
    known_offsets = {
        (
            DagStateCategory.SKIP_IN_TARGET,
            DagStateCategory.SKIP_IN_TARGET,
        ): 1,
        (
            DagStateCategory.SKIP_IN_TARGET,
            DagStateCategory.SELECT_MATCHED,
        ): 2,
    }
    denominator = 3
    if (edge.source.category, edge.dest.category) in known_offsets:
      vertical_line_offset = (
          known_offsets[edge.source.category, edge.dest.category] / denominator
      )
    else:
      vertical_line_offset = 0.63 * start_alignment + 0.37 * end_alignment

    if edge.required_assignment:
      assignment_summary = (
          f"\n{edge.required_assignment.key.category.shortname()}:"
          f" {edge.required_assignment.value.shortname()}"
      )
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

  def extra_annotations(self) -> Iterator[dag_annotator.RegionAnnotation]:
    """Produces region and text annotations."""
    yield from self.text_annotations

  def renderer_specific_setup(self, renderer: rendering.Renderer) -> None:
    if isinstance(renderer, svg_renderer.SVGRenderer):
      style_css = """\
      rect.unexpected_node {
          fill: red;
      }

      rect.label_in_box {
          fill: #f3f3f3;
          stroke-dasharray: 2 2;
      }
      rect.label_in_box.text_token {
          fill: #ccc;
      }
      rect.label_in_box.text_decoration {
          fill: #eee;
          stroke-dasharray: 5 5;
      }

      .edge-annotation {
        --edge-annotation-color: black;
      }
      .edge-annotation.from_SELECT_MATCHED.to_SELECT_MATCHED {
        --edge-annotation-color: darkgreen;
      }
      .edge-annotation.from_SELECT_UNMATCHED.to_SELECT_UNMATCHED {
        --edge-annotation-color: darkorange;
      }
      .edge-annotation path {
        stroke: var(--edge-annotation-color);
      }
      .edge-annotation text {
        fill: var(--edge-annotation-color);
      }
      """
      renderer.configure_style_css(textwrap.dedent(style_css))
