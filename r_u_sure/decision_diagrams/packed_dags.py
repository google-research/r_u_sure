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

"""Optimized representation of a gated state DAG."""
from __future__ import annotations

import enum
import typing
from typing import Any, Callable, NamedTuple, Optional

import numba
import numpy as np
import numpy.typing  # pylint: disable=unused-import
from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.numba_helpers import numba_raise
from r_u_sure.numba_helpers import numba_type_util

State = gated_state_dag.State
SharedVariableKey = gated_state_dag.SharedVariableKey
SharedVariableValue = gated_state_dag.SharedVariableValue
EdgeMetadata = gated_state_dag.EdgeMetadata
SharedVariableAssignment = gated_state_dag.SharedVariableAssignment
Edge = gated_state_dag.Edge
CompleteStateDAG = gated_state_dag.CompleteStateDAG

# NativeRef = numba_native_ref.NativeRef

COST_DTYPE = gated_state_dag.COST_DTYPE
INFINITE_COST = gated_state_dag.INFINITE_COST
INVALID_COST = gated_state_dag.INVALID_COST


class StateWithRecentVariable(NamedTuple):
  """Wrapper around a state, tagging it with a previous variable.

  This type of state is used by the PathGraphBuilder helper class to
  enforce the variable-assignment invariants required by the PathGraph class.
  While building a graph, states are duplicated and assigned copies of
  the original state, tagged with the most recent global variable with
  a known assignment. See comments in `PathGraphBuilder.finish` for details.

  Attributes:
    original_state: State in the original graph.
    last_variable_index: Index of the most recent variable assigned along paths
      to this state.
  """

  original_state: State
  last_variable_index: int


# Structure type for packed storage of edges.
NUMPY_EDGE_STRUCT_DTYPE = np.dtype([
    ("source_index", np.int64),
    ("dest_index", np.int64),
    ("cost", COST_DTYPE),
    ("required_assignment_key_index", np.int64),
    ("required_assignment_value_index", np.int64),
])


class VariableTaggedState(NamedTuple):
  """Wrapper around a state, tagging it with a previous variable.

  This type of state is used by the PackedStateDAG helper class to
  enforce the variable-assignment invariants required for fast marginals.
  While building a graph, states are duplicated and assigned copies of
  the original state, tagged with the most recent global variable with
  a known assignment.

  Attributes:
    original_state: The state in the original graph.
    last_variable_index: Index of previous variable assigned along paths to this
      copy of the state, or -1 if no variable has been assigned.
  """

  original_state: int
  last_variable_index: int


class PackedStateDAG(NamedTuple):
  # pyformat: disable
  """Implementation data for operating on a directed acyclic graph of states.

  The main purpose of this class is to speed up the computation of
  min-marginals, which are the costs of the shortest paths that make each
  specific possible variable assignment. This is accomplished by grouping edges
  such that all edges that assign the same variable are in the same group,
  and such that processing the edges in order produces the dynamic programming
  table for filling the graph.

  As a further optimization, edges are stored in a packed format, which replaces
  most of the metadata with integer indices. This means we can iterate over the
  structure of the graph without performing redundant dictionary lookups.

  Attributes:
    num_tagged_states: The number of states.
    num_variable_keys: The number of variable keys.
    num_variable_values: The number of variable values.
    packed_edge_matrix: A 1D array with packed edge information. The packed
      edges replaces states with state indices, and variables with variable
      indices. Edges are grouped and sorted topologically, so that the edges
      that assign each variable are visited together.
    edge_group_boundary_indices: A 1D array of group boundaries. If `i` is the
      index of variable V in `variable_keys`, then:
        - edge_group_boundary_indices[2*i] gives the end of the edge group
          assigning to the variable before V, and the start of the edge group
          of edges between those that assign the previous variable and those
          that assign V
        - edge_group_boundary_indices[2*i + 1] gives the end of the edge group
          preceding assignments to V, and the start of the edge group assigning
          V. Once we reach this index, we have followed all partial paths that
          do not yet assign V.
        - edge_group_boundary_indices[2*i + 2] gives the end of the edge group
          assigning V. Every edge between edge_group_boundary_indices[2*i + 1]
          (inclusive) and edge_group_boundary_indices[2*i + 2] (exclusive)
          assigns V to some value, and all such edges are in this group. This
          index also gives the start of the edge group following the assignments
          to V, which is also (if applicable) the edge group preceding
          assignments to the variable after V.
    state_variable_boundary_indices: 1D array of state tag variable boundaries.
      For a given index `i`, the states between
      state_variable_boundary_indices[i + 1] and
      state_variable_boundary_indices[i + 2] are the set of states tagged
      with i. (Recall that -1 is a valid variable tag, indicating no
      previous variables.) These indices are related to those in
      edge_group_boundary_indices in the following way:
      - edges between edge_group_boundary_indices[2*i] and
        edge_group_boundary_indices[2*i + 1] will have both source and dest
        between state_variable_boundary_indices[i + 1] and
        state_variable_boundary_indices[i + 2]
      - edges between edge_group_boundary_indices[2*i + 1] and
        edge_group_boundary_indices[2*i + 2] will have a source between
        between state_variable_boundary_indices[i + 1] and
        state_variable_boundary_indices[i + 2], and a dest between
        state_variable_boundary_indices[i + 2] and
        state_variable_boundary_indices[i + 3].
  """
  # pyformat: enable
  num_tagged_states: int
  num_variable_keys: int
  num_variable_values: int
  packed_edge_matrix: np.typing.NDArray[np.dtype[Any]]
  edge_group_boundary_indices: np.typing.NDArray[np.int64]
  state_variable_boundary_indices: np.typing.NDArray[np.int64]


class PackedStateDAGConversionData(NamedTuple):
  """DAG information determining how to interpret a PackedStateDAG.

  Attributes:
    variable_keys: List of variable keys.
    variable_values: List of variable values, including the special None value.
    original_states: List of states in the original graph.
    original_edges: List of edges in the original graph.
    tagged_states: List of variable-tagged states, which have pairs of indices
      into `original_states` and `variable_keys`. The state at index 0 is always
      = the initial state, and the state at the last index is always the final
      state.
    tagged_edges: List of edges between variable-tagged states. The `info` field
      will optionally contain an index into `original_edges`, for edges that
      were copied from the original graph before variable tagging. Edges that
      were inserted between duplicates of the same original state will have
      `info` set to None.
    outgoing_edge_indices_for_tagged_state_index: List of lists of indices into
      `tagged_edges` and into `packed_edge_matrix`, such that if `k =
      outgoing_edge_indices_for_tagged_state_index[i][j]` for some `j` then
      `tagged_edges[k].source == tagged_states[i]` and vice versa. Used to
      reconstruct paths from their costs.
    variable_key_index_from_key: Dictionary mapping variable keys to their
      indices.
    variable_value_index_from_value: Dictionary mapping variable values to their
      indices.
  """

  variable_keys: list[SharedVariableKey]
  variable_values: list[SharedVariableValue]
  original_states: list[State]
  original_edges: list[Edge]
  tagged_states: list[VariableTaggedState]
  tagged_edges: list[Edge]
  outgoing_edge_indices_for_tagged_state_index: list[list[int]]
  variable_key_index_from_key: dict[SharedVariableKey, int]
  variable_value_index_from_value: dict[SharedVariableValue, int]


@numba.extending.register_jitable
def edge_group_boundary_before_assigning(
    packed_dag: PackedStateDAG, variable_index: int
) -> int:
  return packed_dag.edge_group_boundary_indices[2 * variable_index + 1]


@numba.extending.register_jitable
def edge_group_boundary_after_assigning(
    packed_dag: PackedStateDAG, variable_index: int
) -> int:
  return packed_dag.edge_group_boundary_indices[2 * variable_index + 2]


@numba.extending.register_jitable
def state_tag_start_boundary(
    packed_dag: PackedStateDAG, variable_index: int
) -> int:
  return packed_dag.state_variable_boundary_indices[variable_index + 1]


@numba.extending.register_jitable
def state_tag_end_boundary(
    packed_dag: PackedStateDAG, variable_index: int
) -> int:
  return packed_dag.state_variable_boundary_indices[variable_index + 2]


def make_specialized_fn__convert_dag_to_packed(
    example_state: State,
    example_variable_key: SharedVariableKey,
    example_variable_value: SharedVariableValue,
    example_info: EdgeMetadata,
    with_numba: bool = True,
) -> tuple[
    Callable[..., tuple[PackedStateDAG, PackedStateDAGConversionData]],
    Optional[
        Callable[
            [gated_state_dag.CompleteStateDAG], gated_state_dag.CompleteStateDAG
        ]
    ],
]:
  """Helper to build versions of the conversion logic.

  Args:
    example_state: An instance of the state type; only required if `with_numba =
      True`.
    example_variable_key: An instance of the variable key type; only required if
      `with_numba = True`.
    example_variable_value: An instance of the variable value type; only
      required if `with_numba = True`.
    example_info: An instance of the edge metadata info type; only required if
      `with_numba = True`.
    with_numba: Whether to jit-compile a specialized implementation using Numba.

  Returns:
    A tuple (implementation_fn, cast_fn). If `with_numba` is True,
    `implementation_fn` is a Numba-jitted version of convert_dag_to_packed,
    and `cast_fn` is a Python function which should be called on the input
    Python DAG before passing it to `implementation_fn` (this handles casting
    it to be of the correct numba type.) If `with_numba` is False,
    `implementation_fn` is a Python implementation of convert_dag_to_packed,
    and cast_fn is None.
  """
  variable_value_type = numba.typeof(example_variable_value)

  example_cost = INFINITE_COST
  example_variable_assignment = numba_type_util.PretendOptional(
      SharedVariableAssignment(
          key=example_variable_key,
          value=example_variable_value,
      )
  )

  # convince pytype we know what we are doing
  example_variable_assignment = typing.cast(Any, example_variable_assignment)
  optional_variable_assignment_type = numba.typeof(example_variable_assignment)

  example_edge = Edge(
      source=example_state,
      dest=example_state,
      cost=example_cost,
      required_assignment=example_variable_assignment,
      info=example_info,
  )
  edge_type = numba.typeof(example_edge)

  example_tagged_state = VariableTaggedState(
      original_state=example_state,
      last_variable_index=0,
  )

  example_tagged_edge = Edge(
      source=example_tagged_state,
      dest=example_tagged_state,
      cost=INFINITE_COST,
      required_assignment=example_variable_assignment,
      info=0,
  )
  tagged_edge_type = numba.typeof(example_tagged_edge)

  example_complete_dag = CompleteStateDAG(
      initial_state=example_state,
      edges=numba.typed.List.empty_list(item_type=edge_type),
      final_state=example_state,
  )
  complete_dag_type = numba.typeof(example_complete_dag)
  int_edge_tuple_type = numba.typeof((1, example_edge))
  list_of_tagged_edge_type = numba.typeof(
      numba.typed.List.empty_list(item_type=tagged_edge_type)
  )

  # pytype gets very confused about `new_dict`, `new_list`, etc
  if with_numba and not typing.TYPE_CHECKING:
    # These constructors can be used when the type can be inferred from context.
    # Numba can generally infer the types if we immediately add something to
    # the list/dict, but not if we pass the list/dict to another method first,
    # or if we put it into another list/dict.
    new_dict = numba.typed.Dict
    new_list = numba.typed.List

    # When we need an empty list of a particular type, and numba can't infer
    # the type (e.g. if we are storing an empty list in another list), we have
    # to specify the type up front.
    @numba.njit
    def new_list_of(item_type) -> Any:
      return numba.typed.List.empty_list(item_type)

    @numba.njit
    def new_dict_of(key_type, value_type) -> Any:
      return numba.typed.Dict.empty(key_type, value_type)

  else:
    # If running outside Numba, just use regular dynamically-typed lists/dicts.
    new_dict = dict
    new_list = list

    def new_list_of(item_type) -> list[Any]:  # pylint: disable=unused-argument
      return []

    def new_dict_of(key_type, value_type) -> dict[Any, Any]:  # pylint: disable=unused-argument
      return {}

  def convert_dag_to_packed(  # pylint: disable=redefined-outer-name
      dag: CompleteStateDAG,
      missing_assignment_value: SharedVariableValue,
      variable_value_ordering: Optional[list[SharedVariableValue]],
  ) -> tuple[PackedStateDAG, PackedStateDAGConversionData]:
    """Converts a DAG to make it faster to compute min-marginals.

    This function makes the following assumptions about the DAG:
      - In the sequence of edges in the DAG, edges are ordered in a
        topologically-sorted way: after adding an edge with source A and
        destination B, it is an error to add any edges with destination A.
        (Checked by `PartialStateDAG`)
      - Every path in the graph assigns each shared variable at most once (e.g.
        contains at most one edge with a required assignment to that variable
        key). (Checked here.)
      - Shared variable keys are both hashable and orderable, and if k1 < k2,
      then
        every path that assigns both k1 and k2 assigns k1 before it assigns k2.
        (Checked here.)
      - Shared variable values are both hashable and orderable (not strictly
        necessary but this makes it easier to rewrite them into integers)

    Using these assumptions, we can partition the edges in the DAG into groups:
    we track edges that occur before assigning any variable, edges that assign
    variable 1, edges between assignments to variable 1 and 2, edges that assign
    variable 2, and so on. This allows us to quickly identify all of the edges
    that could assign to a variable, group them by the value they assign, and
    determine

    There's one small complication to this: since paths are only required to
    assign each shared variable *at most* once, there is a chance that paths may
    "skip" a group that uses unassigned variables. To handle this, we first
    transform the DAG into an equivalent DAG where every variable is assigned
    *exactly once*, but some variables are assigned to the sentinel value None.

    The way this transformation works is that we transform each state in the
    original DAG into multiple pairs (most_recently_assigned_variable, state).
    We
    then add extra edges which assign variables to None and move between copies
    of the original state. (See the implementation for a more detailed
    description
    of this transformation.)
    After doing this transformation, we can divide our graph into groups
    unambiguously, such that every edge is in exactly one group and states form
    the boundaries of the groups.

    As an optimization, we then "forget" all of the original names of the
    states,
    and replace them with monotonically-increasing integer indices. These
    indices
    can be used to quickly index into a dynamic programming table without doing
    a dictionary lookup with the original state name.

    Finally, we assemble the resulting components into a form which enables us
    to quickly access them.

    Args:
      dag: Original DAG to convert. Every node in this DAG should be reachable
        from the source node. (Consider using
        `gated_state_dag.prune_to_reachable` before calling this function.)
      missing_assignment_value: When a path does not assign any value to a
        variable key, we treat it as assigning this value.
      variable_value_ordering: Optional ordering of possible variable values. If
        not provided, sorts the variable values that appear in the graph.

    Returns:
      A converted version of the DAG which is optimized to allow fast lookups of
      information needed to compute min-marginals.
    """
    # Our goal is to expand our graph such that the following hold:
    # (a) Each path in the original graph corresponds to exactly one path
    #     in the expanded graph, which assigns the original path's variables
    #     to the original path's values, and assigns all originally-unassigned
    #     variables to None.
    # (b) Each path in the expanded graph assigns all the variables, and
    #     assigns them in order.
    #
    # The algorithm starts by identifying the last variable that could possibly
    # have been assigned along any path to each state `X` (denoted
    # `largest_possible_before[X]`). Next, it expands each state
    # X to a sequence of states with None-assigning edges between them:
    #
    # (X, Vm) ----------> (X, V[m+1]) ----------> (X, V[m+2]) --...--> (X, Vn)
    #      [V[m+1] = None]      [V[m+2] = None]
    #
    # and copies each edge in the original graph to an edge in the new graph:
    #
    # (1) If an edge from X to Y assigns variable i, then copy it to an
    #     edge from (X, i-1) to (Y, i).
    # (2) If an edge from X to Y assigns nothing, then copy it to an edge
    #     from (X, largest_possible_before[X])
    #     to (Y, largest_possible_before[X]).
    #
    # For instance, if:
    # - there is a path P1 to state X that assigns variables up to V1
    # - there is a path P2 to state X that assigns variables up to V3
    # - there is an edge from X to Y that doesn't assign anything
    # - there is an edge from X to Z that assigns variable V5
    # - consequently, `largest_possible_before[X] == V3`
    # then the algorithm expands it into the following subgraph:
    #
    #   P1 -> (X, V1)
    #            | [V2=None]
    #            V
    #         (X, V2)
    #            | [V3=None]
    #            V
    #   P2 -> (X, V3) --------------------> (Y, V3)
    #            | [V4=None]
    #            V
    #         (X, V4) ---[V5=something]---> (Z, V5)
    #
    # Observe that we need to make additional copies beyond
    # `largest_possible_before[X]`, to make sure we assign V4 before assigning
    # V5.
    #
    # Proof sketch that (a) holds: By our variable ordering invariant, every
    # variable assigned by any path into state X must be smaller than
    # (i.e. be sorted before) every variable assigned by any path out of
    # state X. Also, by construction, every variable assigned by a path into
    # state X is equal to or smaller than largest_possible_before[X], which in
    # turn is smaller than every variable assigned by any path out of state X
    # So, if a path enters state X having assigned variables up to Vi, we can
    # extend that path so that it passes through
    #   (X, Vi) ... (X, largest_possible_before[X]).
    # If the next edge in the path assigns some other variable Vj, we can
    # further extend the path so that it passes through
    #   (X, largest_possible_before[X]) ... (X, Vj).
    # In both cases, the next edge along the original path now has a one-to-one
    # correspondence with the copy we made for that original edge.
    #
    # Proof sketch that (b) holds: All the paths to states of the form
    # (_, V[i+1]) go through states of the form (_, Vi) and then through an
    # edge that assigns variable i+1. So (b) holds by induction on i.
    #
    # The above strategy works as long as we have enough copies of each state to
    # process all incoming and outgoing edges. To figure out how many copies of
    # each node that we need to add, we need to compute:
    # - the last variable that we *might* have already assigned before reaching
    #   each state `X` (this is `largest_possible_before[X]`). This determines
    #   where non-variable-assigning outgoing edges go.
    # - the last variable that we are *guaranteed* to have already assigned
    #   before reaching each state `X` (which determines the first variable tag
    #   we need to make for X, and is denoted `largest_guaranteed_before[X]`).
    #   This may depend on values of `largest_possible_before[W]` for states W
    #   that are encountered before `X`.
    # - the last variable that is assigned by some edge directly after `X`
    #   (which determines the last variable tag we need to make for `X`, and is
    #   denoted `largest_immediately_after[X]`)

    # We proceed in stages.
    # First, we aggregate a bunch of simple information across edges:
    # - the sets of all assigned variable keys and values
    # - the mapping from source nodes to their outgoing edges
    # - the list of states in the order that they appear as a source
    # (We order states this way because, the first time we see a state as a
    # source, we have already seen any other states that lead to this state by
    # our edge ordering invariant, and we have not yet seen any states that this
    # state can lead to.)
    variable_keys_seen: dict[SharedVariableKey, bool] = new_dict()
    variable_values_seen: dict[SharedVariableValue, bool] = new_dict()
    variable_values_seen[missing_assignment_value] = True

    edges_by_source: dict[State, list[tuple[int, Edge]]] = new_dict()
    states_in_source_order: list[State] = new_list()
    for i, edge in enumerate(dag.edges):
      if edge.required_assignment is not None:
        variable_keys_seen[edge.required_assignment.key] = True
        variable_values_seen[edge.required_assignment.value] = True

      if edge.source not in edges_by_source:
        edges_by_source[edge.source] = new_list_of(int_edge_tuple_type)
        states_in_source_order.append(edge.source)

      edges_by_source[edge.source].append((i, edge))

    assert states_in_source_order[0] == dag.initial_state

    # The final state appears last in the ordering.
    assert dag.final_state not in edges_by_source
    states_in_source_order.append(dag.final_state)
    # Also add it to the edges_by_source dictionary so that we treat it as a
    # state with outgoing dependencies.
    edges_by_source[dag.final_state] = new_list_of(int_edge_tuple_type)

    # Compute orderings for variables.
    ordered_variable_keys = new_list(variable_keys_seen.keys())
    ordered_variable_keys.sort()
    index_of_variable_key: dict[SharedVariableKey, int] = new_dict()
    for i, variable in enumerate(ordered_variable_keys):
      index_of_variable_key[variable] = i

    if variable_value_ordering is None:
      ordered_variable_values = new_list(variable_values_seen.keys())
      ordered_variable_values.sort()
    else:
      ordered_variable_values = new_list(variable_value_ordering)

    index_of_variable_value: dict[SharedVariableValue, int] = new_dict()
    for i, value in enumerate(ordered_variable_values):
      index_of_variable_value[value] = i

    # Next, we compute `largest_possible_before`. Since this is a DAG, and we
    # have topologically-sorted edges, we can just iteratively update this for
    # each state, and once we see an edge that needs to read a value for the
    # state (by having that state as its source) we know we will never update it
    # again. No variables are possible before the initial state.
    largest_possible_before: dict[State, int] = new_dict()
    largest_possible_before[dag.initial_state] = -1
    for edge in dag.edges:
      # The largest variable index possible to have encountered when traversing
      # this edge is either the variable index on this edge, or the largest
      # index seen before this edge.
      if edge.required_assignment is not None:
        update = index_of_variable_key[edge.required_assignment.key]
      elif edge.source in largest_possible_before:
        update = largest_possible_before[edge.source]
      else:
        numba_raise.safe_raise(
            ValueError,
            (
                "Outgoing edge",
                edge,
                (
                    "has an unreachable source. Please make sure every node is"
                    " reachable using `gated_state_dag.prune_to_reachable`"
                    " before calling this function."
                ),
            ),
        )
      # Update it in our dictionary, taking the largest seen so far, since we
      # know that anything we have seen so far is possible.
      if edge.dest not in largest_possible_before:
        largest_possible_before[edge.dest] = update
      else:
        largest_possible_before[edge.dest] = max(
            largest_possible_before[edge.dest], update
        )

    # Next, we compute `largest_guaranteed_before`. For this, we use
    # `largest_possible_before` to determine variables assigned in paths out of
    # each waypoint, because we know that outgoing edges in our expanded DAG
    # will always assign everything up to that.
    # No variables are guaranteed before the initial state.
    largest_guaranteed_before = new_dict()
    largest_guaranteed_before[dag.initial_state] = -1
    for edge in dag.edges:
      # The largest variable index guaranteed to have encountered when
      # traversing this edge is either the variable index on this edge, or the
      # largest index *possible* to have seen before this edge's source. This is
      # because we will assign all variables up to that index to None while
      # processing the edge's source, before leaving it.
      if edge.required_assignment is not None:
        update = index_of_variable_key[edge.required_assignment.key]
      else:
        update = largest_possible_before[edge.source]

      if edge.dest not in largest_guaranteed_before:
        # If this is the only path we have seen, then so far we are guaranteed
        # to visit everything assigned on this path; we will update this later
        # if it turns out there are other possibilities.
        largest_guaranteed_before[edge.dest] = update
      else:
        # Otherwise update it in our dictionary, taking the SMALLEST we have
        # seen so far.
        # We take the smallest because we are looking for variables that we are
        # guaranteed to reach, so if this path doesn't reach some variables we
        # aren't actually guaranteed to
        largest_guaranteed_before[edge.dest] = min(
            largest_guaranteed_before[edge.dest], update
        )

    # Next, we compute `largest_immediately_after`. Note that this is not always
    # populated!
    # Some states may not have entries in `largest_immediately_after`.
    largest_immediately_after = new_dict()
    for edge in dag.edges:
      if edge.required_assignment is not None:
        update = index_of_variable_key[edge.required_assignment.key]
        # Check our ordering invariant. If this edge assigns a variable, then
        # no path to this edge is allowed to have assigned a larger variable
        # already.
        if update <= largest_possible_before[edge.source]:
          conflict_var = ordered_variable_keys[
              largest_possible_before[edge.source]
          ]
          numba_raise.safe_raise(
              ValueError,
              (
                  "Variable ordering violation for edge ",
                  edge,
                  "; variable ",
                  edge.required_assignment.key,
                  " assigned here precedes variable ",
                  conflict_var,
                  " already assigned on a path to this edge.",
              ),
          )
        if edge.source not in largest_immediately_after:
          largest_immediately_after[edge.source] = update
        else:
          largest_immediately_after[edge.source] = max(
              largest_immediately_after[edge.source], update
          )

    # Next, we iterate over the states and add new edges into their
    # appropriate groups, rewriting them to point to the integer state indices.
    edge_groups_before_var: list[list[Edge]] = new_list_of(
        list_of_tagged_edge_type
    )
    for _ in range(len(ordered_variable_keys) + 1):
      edge_groups_before_var.append(new_list_of(tagged_edge_type))

    edge_groups_assigning_var: list[list[Edge]] = new_list_of(
        list_of_tagged_edge_type
    )
    for _ in range(len(ordered_variable_keys)):
      edge_groups_assigning_var.append(new_list_of(tagged_edge_type))

    for state in states_in_source_order:
      # The smallest tagged copy of this state is tagged with the variable that
      # we are always guaranteed to reach along any path. (Including tags
      # earlier than this would be wasteful, since no edges use them.)
      smallest_tag_for_state = largest_guaranteed_before[state]
      # Most outgoing edges from this state can be placed at the point where we
      # know we have processed all incoming paths.
      default_outgoing_tag_for_state = largest_possible_before[state]
      # The largest tagged copy of this state depends on the outgoing edges from
      # this state.
      if state in largest_immediately_after:
        # If there is a variable that is assigned by an edge leaving this state,
        # we need make tagged copies of this state to assign all variables
        # before that one.
        largest_tag_for_state = largest_immediately_after[state] - 1
      else:
        # If none of the edges leaving this state assign a variable, we just
        # need to make sure we make enough copies to handle our incoming and
        # outgoing edges
        largest_tag_for_state = default_outgoing_tag_for_state

      # Construct our tagged copies and connect them with None-assigning edges.
      # (We don't include `largest_tag_for_state` in this iteration, since there
      # isn't a tag after it to connect to.)
      for tag in range(smallest_tag_for_state, largest_tag_for_state):
        # Connect (state, tag) to (state, tag + 1), assigning
        # ordered_variable_keys[tag+1] to None
        edge_groups_assigning_var[tag + 1].append(
            Edge(
                source=VariableTaggedState(
                    original_state=state, last_variable_index=tag
                ),
                dest=VariableTaggedState(
                    original_state=state, last_variable_index=tag + 1
                ),
                cost=0,
                required_assignment=numba_type_util.as_numba_type(
                    SharedVariableAssignment(
                        key=ordered_variable_keys[tag + 1],
                        value=missing_assignment_value,
                    ),
                    optional_variable_assignment_type,
                ),
                info=-1,
            )
        )

      # Copy all normal edges to connect the appropriate tagged states.
      for edge_original_index, edge in edges_by_source[state]:
        assert edge.source == state
        if edge.dest not in edges_by_source:
          # The destination node has no outgoing edges, and it isn't the final
          # state (which always appears as a key in `edges_by_source`) so don't
          # bother processing it.
          continue

        if edge.cost == INFINITE_COST:
          # This edge has an infinite cost, so don't bother processing it.
          continue

        if edge.required_assignment is None:
          # If the edge does not assign a variable, both of its states should
          # be tagged with `default_outgoing_tag_for_state`.
          # The next variable that will be assigned is variable
          # `default_outgoing_tag_for_state + 1`
          next_var_index = default_outgoing_tag_for_state + 1
          edge_groups_before_var[next_var_index].append(
              Edge(
                  source=VariableTaggedState(
                      original_state=state,
                      last_variable_index=default_outgoing_tag_for_state,
                  ),
                  dest=VariableTaggedState(
                      original_state=edge.dest,
                      last_variable_index=default_outgoing_tag_for_state,
                  ),
                  cost=edge.cost,
                  required_assignment=numba_type_util.as_numba_type(
                      None, optional_variable_assignment_type
                  ),
                  info=edge_original_index,
              )
          )
        else:
          # If the edge assignes to variable `i`, we copy it to connect tagged
          # nodes (source, i-1) and (dest, i).
          var_index = index_of_variable_key[edge.required_assignment.key]
          edge_groups_assigning_var[var_index].append(
              Edge(
                  source=VariableTaggedState(
                      original_state=state, last_variable_index=var_index - 1
                  ),
                  dest=VariableTaggedState(
                      original_state=edge.dest, last_variable_index=var_index
                  ),
                  cost=edge.cost,
                  required_assignment=numba_type_util.as_numba_type(
                      SharedVariableAssignment(
                          key=edge.required_assignment.key,
                          value=edge.required_assignment.value,
                      ),
                      optional_variable_assignment_type,
                  ),
                  info=edge_original_index,
              )
          )

    # We can interleave these groups to make sure they appear in the right
    # order.
    edge_groups = new_list()
    for i in range(len(ordered_variable_keys)):
      edge_groups.append(edge_groups_before_var[i])
      edge_groups.append(edge_groups_assigning_var[i])

    edge_groups.append(edge_groups_before_var[len(ordered_variable_keys)])

    # Next, we figure out state ordering. We iterate through groups of edges,
    # then edges in each group. Whenever we read from a state for the first
    # time, we then allocate that state an index into the flat DP table; the
    # goal is to ensure that states that are read at similar times end up close
    # to each other in memory.
    table_index_for_tagged_state = new_dict()
    tagged_states_in_table_order = new_list()
    total_edge_count = 0
    tagged_edges_in_table_order = new_list()
    for group in edge_groups:
      for edge in group:
        tagged_edges_in_table_order.append(edge)
        if edge.source not in table_index_for_tagged_state:
          table_index_for_tagged_state[edge.source] = len(
              tagged_states_in_table_order
          )
          tagged_states_in_table_order.append(edge.source)
        total_edge_count += 1

    assert tagged_states_in_table_order[0] == VariableTaggedState(
        original_state=dag.initial_state, last_variable_index=-1
    )

    final_tagged_state = VariableTaggedState(
        original_state=dag.final_state,
        last_variable_index=len(ordered_variable_keys) - 1,
    )
    assert final_tagged_state not in table_index_for_tagged_state
    table_index_for_tagged_state[final_tagged_state] = len(
        tagged_states_in_table_order
    )
    tagged_states_in_table_order.append(final_tagged_state)

    # Next we figure out boundaries for our states, which are points where the
    # tag changes.
    state_variable_boundary_indices = np.empty(
        len(ordered_variable_keys) + 2, np.int64
    )
    last_tag = -1
    state_variable_boundary_indices[0] = 0
    for i, state in enumerate(tagged_states_in_table_order):
      for finished_tag in range(last_tag, state.last_variable_index):
        state_variable_boundary_indices[finished_tag + 2] = i
      last_tag = state.last_variable_index
    for finished_tag in range(last_tag, len(ordered_variable_keys)):
      state_variable_boundary_indices[finished_tag + 2] = len(
          tagged_states_in_table_order
      )

    # Next, we pack all of our edges into an efficient flat array, replacing
    # states with their packed state ordering.
    edge_group_boundary_indices = np.empty(len(edge_groups) + 1, np.int64)
    packed_edge_matrix = np.empty((total_edge_count,), NUMPY_EDGE_STRUCT_DTYPE)
    current_edge_index = 0
    for group_index, group in enumerate(edge_groups):
      edge_group_boundary_indices[group_index] = current_edge_index
      for edge in group:
        if edge.required_assignment is not None:
          assignment_key_index = index_of_variable_key[
              edge.required_assignment.key
          ]
          assignment_value_index = index_of_variable_value[
              edge.required_assignment.value
          ]
        else:
          assignment_key_index = -1
          assignment_value_index = -1

        packed_edge_matrix[current_edge_index]["source_index"] = (
            table_index_for_tagged_state[edge.source]
        )
        packed_edge_matrix[current_edge_index]["dest_index"] = (
            table_index_for_tagged_state[edge.dest]
        )
        packed_edge_matrix[current_edge_index]["cost"] = edge.cost
        packed_edge_matrix[current_edge_index][
            "required_assignment_key_index"
        ] = assignment_key_index
        packed_edge_matrix[current_edge_index][
            "required_assignment_value_index"
        ] = assignment_value_index
        current_edge_index += 1

    edge_group_boundary_indices[len(edge_groups)] = current_edge_index

    # Next we construct outgoing_edge_indices_for_tagged_state_index.
    outgoing_edge_indices_for_tagged_state_index = new_list()
    for _ in range(len(tagged_states_in_table_order)):
      outgoing_edge_indices_for_tagged_state_index.append(
          new_list_of(numba.int64)
      )

    for edge_index, edge in enumerate(tagged_edges_in_table_order):
      source_index = table_index_for_tagged_state[edge.source]
      outgoing_edge_indices_for_tagged_state_index[source_index].append(
          edge_index
      )

    # Finally, we construct the graph itself.
    return (
        PackedStateDAG(
            num_tagged_states=len(tagged_states_in_table_order),
            num_variable_keys=len(ordered_variable_keys),
            num_variable_values=len(ordered_variable_values),
            packed_edge_matrix=packed_edge_matrix,
            edge_group_boundary_indices=edge_group_boundary_indices,
            state_variable_boundary_indices=state_variable_boundary_indices,
        ),
        PackedStateDAGConversionData(
            variable_keys=ordered_variable_keys,
            variable_values=ordered_variable_values,
            original_states=states_in_source_order,
            original_edges=dag.edges,
            tagged_states=tagged_states_in_table_order,
            tagged_edges=tagged_edges_in_table_order,
            outgoing_edge_indices_for_tagged_state_index=(
                outgoing_edge_indices_for_tagged_state_index
            ),
            variable_key_index_from_key=index_of_variable_key,
            variable_value_index_from_value=index_of_variable_value,
        ),
    )

  if with_numba:
    convert_dag_to_packed_jit = numba.jit(
        [
            (complete_dag_type, variable_value_type, numba.typeof(None)),
            (
                complete_dag_type,
                variable_value_type,
                numba.typeof(numba.typed.List.empty_list(variable_value_type)),
            ),
        ],
        nopython=True,
    )(convert_dag_to_packed)

    def prepare_input_dag(dag):
      """Helper function that safely converts a Python DAG into a typed form."""
      result = CompleteStateDAG(
          initial_state=dag.initial_state,
          edges=numba.typed.List.empty_list(item_type=edge_type),
          final_state=dag.final_state,
      )
      for edge in dag.edges:
        # Edges must be appended one-by-one so that Numba casts each one
        # correctly.
        result.edges.append(edge)
      return result

    return convert_dag_to_packed_jit, prepare_input_dag
  else:
    return convert_dag_to_packed, None


convert_dag_to_packed, _ = make_specialized_fn__convert_dag_to_packed(
    with_numba=False,
    # The rest of the parameters are ignored when with_numba=False.
    example_state=None,
    example_variable_key=None,
    example_variable_value=None,
    example_info=None,
)


def scale_packed_dag_costs(dag: PackedStateDAG, scale: float) -> PackedStateDAG:
  """Returns a new packed DAG whose costs are all scaled by `scale`."""
  packed_edge_matrix = np.copy(dag.packed_edge_matrix)
  packed_edge_matrix["cost"] *= scale
  return PackedStateDAG(
      num_tagged_states=dag.num_tagged_states,
      num_variable_keys=dag.num_variable_keys,
      num_variable_values=dag.num_variable_values,
      packed_edge_matrix=packed_edge_matrix,
      edge_group_boundary_indices=dag.edge_group_boundary_indices,
      state_variable_boundary_indices=dag.state_variable_boundary_indices,
  )


class PenalizedShortestPrefixSuffixTables(NamedTuple):
  """Tables containing costs of shortest paths to and from particular states.

  This class implements the dynamic programming algorithm for shortest paths
  in a DAG. It does this bidirectionally: it both computes the cost of the
  shortest path to each state, and the cost of the shortest path from each
  state. This allows us to very quickly compute min-marginals: the cost of the
  shortest path that makes a particular variable assignment. In particular, we
  can simply iterate over all of the edges that make that assignment, and add
  their costs to the costs of the shortest path to the source and the shortest
  path from the destination.

  This class also allows assignment penalties. These penalties are added to
  all of the edges that make a particular variable assignment, and can be used
  for a few purposes:
  - By assigning finite penalties to assignments, we can implement the dual
    decomposition optimization algorithm, exchanging messages between multiple
    state DAGs.
  - By assigning infinite penalties to disallowed assignments, we can prune the
    edges that make those assignments, enabling us to compute costs of
    particular variable configurations.
  - By assigning infinite penalties to only a subset of these assignments, we
    can obtain an admissible heuristic (suitable for A* or similar), in other
    words, a lower bound on the cost of making a particular set of assignments
    (by assuming all other assignments are chosen optimally for this DAG, not
    considering any others we are simultaneously optimizing over).
  - By assigning randomized penalties to assignments, we can implement a relaxed
    version of our optimization problem, potentially enabling gradient-based
    methods to optimize a variable-assignment policy.

  Furthermore, this class is designed to allow efficient updates of variable
  penalties in traversal order. Modifying the penalties for a particular
  variable invalidates the prefixes and suffixes that include that variable, but
  NOT the prefixes before it, or suffixes after it. If we want to then compute
  the min marginals of a different variable, we only need to update the table
  entries between the two. We thus track two valid pointers, which determine
  up to where in the table our prefixes and suffixes are valid, and allow
  updating them on demand.

  Note that, while the class itself is frozen (since it is a NamedTuple),
  the numpy arrays within it are not; they are mutated by the instance methods
  of this class. These should not be modified directly, otherwise outputs may
  be incorrect.

  Instances of this class are often referred to as a "memo" since they contain
  the scratch work that is referred back to by multiple functions.

  Attributes:
    dag: The DAG to compute scores for.
    penalties: float32[num_variable_keys, num_values] array, giving the penalty
      of assigning each variable this value.
    prefix_table: float32[num_states] array, giving the cost (including
      penalties) of the shortest path from the initial state to this state. Only
      guaranteed to be accurate for states whose tag is at or before
      `prefixes_valid_ending_at_tag`
    suffix_table: float32[num_states] array, giving the cost (including
      penalties) of the shortest path from this state to the final state. Only
      guaranteed to be accurate for states whose tag is at or after
      `suffixes_valid_starting_from_tag`.
    prefixes_valid_ending_at_tag: Scalar variable index indicating the variable
      index for which we can compute accurate prefix costs. `prefix_table` will
      have been updated for all states tagged with this variable or an earlier
      one. This is stored as a Numpy array with shape () so that it can be
      mutated. Starts at -2 (before the first tag, which is -1)
    suffixes_valid_starting_from_tag: Scalar variable index indicating the
      variable index for which we can compute accurate suffix costs.
      `suffix_table` will have been updated for all states tagged with this
      variable or a later one. This is stored as a Numpy array with shape () so
      that it can be mutated. Starts at num_variable_keys (after the last tag,
      which is num_variable_keys - 1)
  """

  dag: PackedStateDAG
  penalties: np.NDArray[np.float32]
  prefix_table: np.NDArray[np.float32]
  suffix_table: np.NDArray[np.float32]
  prefixes_valid_ending_at_tag: np.NDArray[(), int]
  suffixes_valid_starting_from_tag: np.NDArray[(), int]


@numba.extending.register_jitable
def empty_table(
    dag: PackedStateDAG, penalties: np.NDArray[np.float32]
) -> PenalizedShortestPrefixSuffixTables:
  """Constructs an empty table for a given dag.

  Args:
    dag: Packed state DAG that we want to compute tables for.
    penalties: Set of penalties we want to start with.

  Returns:
    New table.
  """
  return PenalizedShortestPrefixSuffixTables(
      dag=dag,
      penalties=np.copy(penalties),
      prefix_table=np.full((dag.num_tagged_states,), INVALID_COST),
      suffix_table=np.full((dag.num_tagged_states,), INVALID_COST),
      # one before the first tag (-1)
      prefixes_valid_ending_at_tag=np.array(-2),
      # one after the last tag (dag.num_variable_keys - 1)
      suffixes_valid_starting_from_tag=np.array(dag.num_variable_keys),
  )


@numba.extending.register_jitable
def empty_unpenalized_table(
    dag: PackedStateDAG,
) -> PenalizedShortestPrefixSuffixTables:
  """Constructs an empty table for a given dag with no penalties."""
  return empty_table(
      dag=dag,
      penalties=np.zeros(
          (dag.num_variable_keys, dag.num_variable_values),
          dtype=COST_DTYPE,
      ),
  )


@numba.extending.register_jitable
def copy_memo(
    memo: PenalizedShortestPrefixSuffixTables,
) -> PenalizedShortestPrefixSuffixTables:
  return PenalizedShortestPrefixSuffixTables(
      dag=memo.dag,
      penalties=np.copy(memo.penalties),
      prefix_table=np.copy(memo.prefix_table),
      suffix_table=np.copy(memo.suffix_table),
      prefixes_valid_ending_at_tag=np.copy(memo.prefixes_valid_ending_at_tag),
      suffixes_valid_starting_from_tag=np.copy(
          memo.suffixes_valid_starting_from_tag
      ),
  )


@numba.extending.register_jitable
def masked_prefixes(
    memo: PenalizedShortestPrefixSuffixTables,
) -> np.typing.NDArray[COST_DTYPE]:
  """Returns a set of prefixes, with INVALID_COST in place of uncomputed values."""
  result = np.copy(memo.prefix_table)
  # Only states tagged with this variable or before are valid. Mask out
  # the ones tagged with larger variables.
  bad_from = state_tag_end_boundary(memo.dag, memo.prefixes_valid_ending_at_tag)
  result[bad_from:] = INVALID_COST
  return result


@numba.extending.register_jitable
def masked_suffixes(
    memo: PenalizedShortestPrefixSuffixTables,
) -> np.NDArray[np.float32]:
  """Returns a set of suffixes, with INVALID_COST in place of uncomputed values."""
  result = np.copy(memo.suffix_table)
  # Only states tagged with this variable or after are valid. Mask out the
  # ones tagged with smaller variables.
  bad_to = state_tag_start_boundary(
      memo.dag, memo.suffixes_valid_starting_from_tag
  )
  result[:bad_to] = INVALID_COST
  return result


@numba.extending.register_jitable
def update_prefixes_up_to(
    memo: PenalizedShortestPrefixSuffixTables, desired_tag: int
) -> None:
  """Updates prefixes for all nodes up to a given tag.

  Args:
    memo: The tables to update.
    desired_tag: Tag that we want to be valid. Should either be an index of a
      variable in `memo.dag.variable_keys`, or -1.
  """
  assert desired_tag >= -1
  assert desired_tag < memo.dag.num_variable_keys
  was_valid_tag = memo.prefixes_valid_ending_at_tag
  if desired_tag <= was_valid_tag:
    # Already valid!
    return
  # First, we reinitialize the state values for the states we are going to
  # update. We need to update the tags following the one that is already
  # correct, up to the one we want to be valid up to.
  #
  # For instance, if was_valid_tag was 1, and desired_tag is 3, we want
  # to update these states:
  #
  #  tag -1  tag 0    tag 1     tag 2      tag 3      tag 4     tag 5
  # |------|-------|---------|--------|-----------|----------|---------|
  #        (already set)      ^^^^^^^^^^^^^^^^^^^^    (for later)
  #
  state_init_start = state_tag_start_boundary(memo.dag, was_valid_tag + 1)
  state_init_stop = state_tag_end_boundary(memo.dag, desired_tag)
  memo.prefix_table[state_init_start:state_init_stop] = INFINITE_COST
  # Next, we process all of the edges whose destination is in this region.
  # This includes edges that assign to `was_valid_tag + 1` (crossing from
  # the "tag 1" to the "tag 2" region in the example above), up until (but
  # not including) the first edge that assigns to `desired_tag + 1` (which
  # will cover all edges that assign to nodes in the "tag 3" region above)
  if was_valid_tag == -2:
    edge_start_inclusive = 0
    memo.prefix_table[0] = 0  # empty prefix has zero cost
  else:
    edge_start_inclusive = edge_group_boundary_before_assigning(
        memo.dag, was_valid_tag + 1
    )
  if desired_tag == memo.dag.num_variable_keys:
    edge_end_exclusive = memo.dag.edge_group_boundary_indices[-1]
  else:
    edge_end_exclusive = edge_group_boundary_before_assigning(
        memo.dag, desired_tag + 1
    )
  for edge_index in range(edge_start_inclusive, edge_end_exclusive):
    assert edge_index >= 0
    packed_edge = memo.dag.packed_edge_matrix[edge_index]
    source_index = packed_edge["source_index"]
    dest_index = packed_edge["dest_index"]
    cost = packed_edge["cost"]
    required_assignment_key_index = packed_edge["required_assignment_key_index"]
    required_assignment_value_index = packed_edge[
        "required_assignment_value_index"
    ]
    if required_assignment_key_index == -1:
      penalty = 0.0
    else:
      penalty = memo.penalties[
          required_assignment_key_index, required_assignment_value_index
      ]
    source_prefix_cost = memo.prefix_table[source_index]
    proposed_dest_cost = source_prefix_cost + cost + penalty
    memo.prefix_table[dest_index] = min(
        memo.prefix_table[dest_index], proposed_dest_cost
    )
  # Finally, we update our valid index.
  memo.prefixes_valid_ending_at_tag[()] = desired_tag


@numba.extending.register_jitable
def update_suffixes_up_to(
    memo: PenalizedShortestPrefixSuffixTables, desired_tag: int
) -> None:
  """Updates suffixes up to a given variable index.

  Args:
    memo: The tables to update.
    desired_tag: Tag that we want to be valid. Should either be an index of a
      variable in `memo.dag.variable_keys`, or -1.
  """
  assert desired_tag >= -1
  assert desired_tag < memo.dag.num_variable_keys
  was_valid_tag = memo.suffixes_valid_starting_from_tag
  if desired_tag >= was_valid_tag:
    # Already valid!
    return
  # First, we reinitialize the state values for the states we are going to
  # update. We need to update the tags before the one that is already
  # correct, up to (in reverse) the one we want to be valid up to.
  #
  # For instance, if was_valid_tag was 4, and desired_tag is 2,
  # we want to update these states:
  #
  #  tag -1  tag 0    tag 1     tag 2      tag 3      tag 4     tag 5
  # |------|-------|---------|--------|-----------|----------|---------|
  #         (for later)       ^^^^^^^^^^^^^^^^^^^^     (already set)
  #
  state_init_start = state_tag_start_boundary(memo.dag, desired_tag)
  state_init_stop = state_tag_end_boundary(memo.dag, was_valid_tag - 1)
  memo.suffix_table[state_init_start:state_init_stop] = INFINITE_COST
  # Next, we process all of the edges whose source is in this region, in
  # reverse order.
  # This includes edges that assign to `was_valid_tag` (crossing from
  # the "tag 3" to the "tag 4" region in the example above), up until (but
  # not including) the first edge that assigns to `desired_tag` (which
  # will cover all edges that assign to nodes in the "tag 2" region above)
  # We have to do a bit of index tweaking so that we properly handle the
  # inclusive and exclusive boundaries when iterating in reverse.
  if was_valid_tag == memo.dag.num_variable_keys:
    edge_start_inclusive = memo.dag.edge_group_boundary_indices[-1] - 1
    memo.suffix_table[-1] = 0  # empty suffix has zero cost
  else:
    edge_start_inclusive = (
        edge_group_boundary_after_assigning(memo.dag, was_valid_tag) - 1
    )
  if desired_tag == -1:
    # this is literally one before the start, not the wraparound last element
    edge_end_exclusive = -1
  else:
    edge_end_exclusive = (
        edge_group_boundary_after_assigning(memo.dag, desired_tag) - 1
    )
  for edge_index in range(edge_start_inclusive, edge_end_exclusive, -1):
    assert edge_index >= 0
    packed_edge = memo.dag.packed_edge_matrix[edge_index]
    source_index = packed_edge["source_index"]
    dest_index = packed_edge["dest_index"]
    cost = packed_edge["cost"]
    required_assignment_key_index = packed_edge["required_assignment_key_index"]
    required_assignment_value_index = packed_edge[
        "required_assignment_value_index"
    ]
    if required_assignment_key_index == -1:
      penalty = 0.0
    else:
      penalty = memo.penalties[
          required_assignment_key_index, required_assignment_value_index
      ]
    dest_suffix_cost = memo.suffix_table[dest_index]
    proposed_source_cost = dest_suffix_cost + cost + penalty
    memo.suffix_table[source_index] = min(
        memo.suffix_table[source_index], proposed_source_cost
    )
  # Finally, we update our valid index.
  memo.suffixes_valid_starting_from_tag[()] = desired_tag


@numba.extending.register_jitable
def set_penalties(
    memo: PenalizedShortestPrefixSuffixTables,
    variable_index: int,
    penalties: np.typing.NDArray[COST_DTYPE],
) -> None:
  """Sets the penalties for a particular variable, and invalidates tables.

  This method should be used to adjust variable penalties for variables, to
  ensure that validity is tracked appropriately.

  Args:
    memo: The tables to update.
    variable_index: The index of the variable we want to adjust penalties for.
    penalties: COST_DTYPE[num_variable_values] array giving the new penalites we
      want to assign.
  """
  memo.penalties[variable_index, :] = penalties
  # Changing penalties for `variable_index` invalidates the costs for any path
  # that assigns `variable_index`. Thus prefix costs to any state tagged with
  # `variable_index` or after are now invalid; the last valid tag would be
  # `variable_index - 1`.
  memo.prefixes_valid_ending_at_tag[()] = min(
      memo.prefixes_valid_ending_at_tag[()], variable_index - 1
  )
  # Similarly suffix costs from any state tagged with `variable_index - 1` or
  # before are now invalid; the earliest valid tag would be `variable_index`.
  memo.suffixes_valid_starting_from_tag[()] = max(
      memo.suffixes_valid_starting_from_tag[()], variable_index
  )


@numba.extending.register_jitable
def make_fully_valid(memo: PenalizedShortestPrefixSuffixTables) -> None:
  """Updates tables so that they reflect the current penalties."""
  update_prefixes_up_to(memo, memo.dag.num_variable_keys - 1)
  update_suffixes_up_to(memo, -1)


@numba.njit
def compute_min_marginals_for_variable(
    memo: PenalizedShortestPrefixSuffixTables, variable_index: int
) -> np.typing.NDArray[COST_DTYPE]:
  """Computes min marginals for a variable at a given index.

  This method will adaptively compute only the information necessary to
  determine min marginals for the requested index.

  Args:
    memo: The tables to use for computation.
    variable_index: Index of the variable to compute min marginals for.

  Returns:
    COST_DTYPE[num_variable_values] array, such that `result[i]` is the cost
      of the lowest-cost path that assigns
      `ordered_variable_keys[variable_index]` to `ordered_variable_values[i]`
  """
  assert variable_index >= 0
  assert variable_index < memo.dag.num_variable_keys
  # Make sure we have computed costs for paths before and after assignments
  # to this variable.
  update_prefixes_up_to(memo, variable_index - 1)
  update_suffixes_up_to(memo, variable_index)
  # Iterate across edges that assign this variable, and aggregate their costs.
  min_marginals = np.full((memo.dag.num_variable_values,), INFINITE_COST)
  edge_group_start = edge_group_boundary_before_assigning(
      memo.dag, variable_index
  )
  edge_group_end = edge_group_boundary_after_assigning(memo.dag, variable_index)
  for edge_index in range(edge_group_start, edge_group_end):
    packed_edge = memo.dag.packed_edge_matrix[edge_index]
    source_index = packed_edge["source_index"]
    dest_index = packed_edge["dest_index"]
    cost = packed_edge["cost"]
    required_assignment_key_index = packed_edge["required_assignment_key_index"]
    required_assignment_value_index = packed_edge[
        "required_assignment_value_index"
    ]
    assert required_assignment_key_index == variable_index
    penalty = memo.penalties[variable_index, required_assignment_value_index]
    path_cost = (
        memo.prefix_table[source_index]
        + cost
        + penalty
        + memo.suffix_table[dest_index]
    )
    min_marginals[required_assignment_value_index] = min(
        min_marginals[required_assignment_value_index], path_cost
    )
  return min_marginals


@numba.njit
def compute_all_min_marginals(
    memo: PenalizedShortestPrefixSuffixTables,
) -> np.typing.NDArray[COST_DTYPE]:
  """Computes all min marginals at once.

  Args:
    memo: The tables to use for computation.

  Returns:
    COST_DTYPE[num_variable_keys, num_variable_values] array, such that
      `result[k, v]` is the cost of the lowest-cost path that assigns
      `ordered_variable_keys[k]` to `ordered_variable_values[v]`
  """
  all_min_marginals = np.full(
      (memo.dag.num_variable_keys, memo.dag.num_variable_values), INVALID_COST
  )
  make_fully_valid(memo)
  for variable_index in range(memo.dag.num_variable_keys):
    all_min_marginals[variable_index, :] = compute_min_marginals_for_variable(
        memo, variable_index
    )
  return all_min_marginals


class MinimalCostComputationStrategy(enum.Enum):
  """Strategies for computing the minimum cost.

  Attributes:
    AT_END: Compute it using a forward path to the end of the problem.
    AT_START: Compute it using a reverse path to the start of the problem.
    FROM_MIDDLE: Compute it using the min-marginals in the middle of the
      problem. Requires summing over multiple values, but can take advantage of
      partially-computed prefixes and suffixes, and automatically uses them if
      applicable.
  """

  AT_END = enum.auto()
  AT_START = enum.auto()
  FROM_MIDDLE = enum.auto()


@numba.extending.register_jitable
def compute_minimal_cost(
    memo: PenalizedShortestPrefixSuffixTables,
    strategy: MinimalCostComputationStrategy = MinimalCostComputationStrategy.FROM_MIDDLE,
) -> COST_DTYPE:
  """Computes the cost of the shortest path through the (penalized) DAG.

  Args:
    memo: The tables to use for computation.
    strategy: How to compute the cost.

  Returns:
    Cost of the optimal path.
  """
  if strategy == MinimalCostComputationStrategy.AT_START:
    update_suffixes_up_to(memo, -1)
    return memo.suffix_table[0]
  elif strategy == MinimalCostComputationStrategy.AT_END:
    update_prefixes_up_to(memo, memo.dag.num_variable_keys - 1)
    return memo.prefix_table[-1]
  elif strategy == MinimalCostComputationStrategy.FROM_MIDDLE:
    target_var = memo.suffixes_valid_starting_from_tag[()]
    target_var = max(0, target_var)
    target_var = min(target_var, memo.dag.num_variable_keys - 1)
    min_marginals = compute_min_marginals_for_variable(
        memo, variable_index=target_var
    )
    return np.min(min_marginals)
  else:
    raise ValueError("Unrecognized minimal cost strategy")


def extract_minimal_cost_path_and_assignments(
    memo: PenalizedShortestPrefixSuffixTables,
    conversion_data: PackedStateDAGConversionData,
    tagged: bool = False,
) -> tuple[list[Edge], dict[SharedVariableKey, SharedVariableValue]]:
  """Extracts a path that attains the minimal cost, with its assignments.

  Should not be called if the best cost is infinite.

  Args:
    memo: The prefix/suffix table to use when computing the path.
    conversion_data: Data about how the graph was converted, used to reconstruct
      edges.
    tagged: Whether to return tagged edges instead of original edges.

  Returns:
    A list of edges in the original graph (without variable tags) that
    reconstitutes the original path, and a dictionary mapping variable keys
    to their values.
  """
  # We use suffix costs to determine which edge to take at each state.
  update_suffixes_up_to(memo, -1)
  state_index = 0
  path = []
  assignments = {}
  final_state_index = memo.dag.num_tagged_states - 1
  while state_index < final_state_index:
    outgoing_edge_indices = (
        conversion_data.outgoing_edge_indices_for_tagged_state_index[
            state_index
        ]
    )
    assert outgoing_edge_indices
    best_index = outgoing_edge_indices[0]
    best_cost_to_go = INFINITE_COST
    best_next_state = memo.dag.num_tagged_states  # invalid value
    for outgoing_edge_index in outgoing_edge_indices:
      packed_edge = memo.dag.packed_edge_matrix[outgoing_edge_index]
      dest_index = packed_edge["dest_index"]
      cost = packed_edge["cost"]
      required_assignment_key_index = packed_edge[
          "required_assignment_key_index"
      ]
      required_assignment_value_index = packed_edge[
          "required_assignment_value_index"
      ]
      if required_assignment_key_index == -1:
        penalty = 0.0
      else:
        penalty = memo.penalties[
            required_assignment_key_index, required_assignment_value_index
        ]
      dest_suffix_cost = memo.suffix_table[dest_index]
      cost_to_go = dest_suffix_cost + cost + penalty
      if cost_to_go < best_cost_to_go:
        best_cost_to_go = cost_to_go
        best_index = outgoing_edge_index
        best_next_state = dest_index

    state_index = best_next_state
    tagged_edge = conversion_data.tagged_edges[best_index]

    if tagged_edge.required_assignment is not None:
      assignments[tagged_edge.required_assignment.key] = (
          tagged_edge.required_assignment.value
      )

    if tagged:
      path.append(tagged_edge)
    else:
      original_edge_index = tagged_edge.info
      if original_edge_index != -1:
        path.append(conversion_data.original_edges[original_edge_index])

  return path, assignments


def constrained_best_path(
    dag: PackedStateDAG,
    conversion_data: PackedStateDAGConversionData,
    constraints: dict[SharedVariableKey, SharedVariableValue],
) -> tuple[list[Edge], dict[SharedVariableKey, SharedVariableValue]]:
  """Computes the best path through a DAG under a set of variable constraints.

  This is a helper function intended to make it easier to solve constrained
  shortest path problems for a single DAG.

  Args:
    dag: The DAG to process.
    conversion_data: Data regarding the conversion to a packed DAG.
    constraints: Set of assignments that must hold.

  Returns:
    A list of edges in the original graph (without variable tags) that
    reconstitutes the original path, and a dictionary mapping variable keys
    to their values, which will always include the constraints.
  """
  memo = empty_unpenalized_table(dag)
  for key, value in constraints.items():
    key_index = conversion_data.variable_key_index_from_key[key]
    value_index = conversion_data.variable_value_index_from_value[value]
    var_penalties = np.full((dag.num_variable_values,), INFINITE_COST)
    var_penalties[value_index] = 0
    set_penalties(memo, key_index, var_penalties)

  return extract_minimal_cost_path_and_assignments(
      memo, conversion_data, tagged=False
  )
