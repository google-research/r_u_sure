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

"""Gated state directed-acyclic-graph abstraction.

A gated state DAG is a directed acyclic graph whose nodes are called states,
and whose edges are "gated" by shared variables that determine which edges
can be used.

Equivalently, it is what we refer to in the paper as a multivalued, ordered,
nondeterministic, weighted decision diagram, which can be reduced to a binary
decision diagram over indicator variables.
"""
from __future__ import annotations

import contextlib
import math
import traceback
from typing import Any, Optional, NamedTuple

import numba
import numpy as np

from r_u_sure.numba_helpers import numba_raise


#### Shared data structures ####
# These data structures represent the structure of the graph. They are used
# both in high-level wrappers and inside the low-level representation.

State = Any
SharedVariableKey = Any
SharedVariableValue = Any
EdgeMetadata = Any

# Currently using floating point numbers, although they have numerical
# precision issues occasionally. Alternative: move to some more complex version
# of fixed-point numbers. However, must still support infinite values and
# handle them separately when adding.
Cost = float
COST_DTYPE = np.float64
INFINITE_COST = np.inf
INVALID_COST = np.nan


class SharedVariableAssignment(NamedTuple):
  """An assignment to a shared variable."""

  key: Any
  value: Any


class Edge(NamedTuple):
  """An edge in a state DAG."""

  source: State
  dest: State
  cost: Cost
  required_assignment: Optional[SharedVariableAssignment] = None
  info: EdgeMetadata = None


class CompleteStateDAG(NamedTuple):
  """Unprocessed complete dag, useful for e.g. visualization."""

  initial_state: State
  edges: list[Edge]
  final_state: State


def extract_state_list(dag: CompleteStateDAG) -> list[State]:
  """Computes the set of states in a dag.

  Args:
    dag: The dag we want to query.

  Returns:
    A list of unique states, ordered by their appearance in `edges`.

  Raises:
    ValueError: If this graph has not been finished.
  """
  seen_states = set()
  result = []

  for edge in dag.edges:
    if edge.source not in seen_states:
      seen_states.add(edge.source)
      result.append(edge.source)
    if edge.dest not in seen_states:
      seen_states.add(edge.dest)
      result.append(edge.dest)

  return result


def compute_reachable_states(dag: CompleteStateDAG) -> set[State]:
  """Finds the set of states that are part of a finite path.

  Args:
    dag: The dag we want to query.

  Returns:
    A set containing only the states for which both:
    - there exists a sequence of finite edges from the start state to this
    one.
    - there exists a sequence of finite edges from this one to the end state.

  Raises:
    ValueError: If this graph has not been finished.
  """
  relevant_edges = [edge for edge in dag.edges if edge.cost < math.inf]

  # Walk graph forwards (topologically)
  forward_reachable = {dag.initial_state}
  for edge in relevant_edges:
    if edge.source in forward_reachable:
      forward_reachable.add(edge.dest)

  # Walk graph in reverse (topologically)
  backward_reachable = {dag.final_state}
  for edge in reversed(relevant_edges):
    if edge.dest in backward_reachable:
      backward_reachable.add(edge.source)

  return forward_reachable & backward_reachable


def prune_to_reachable(dag: CompleteStateDAG) -> CompleteStateDAG:
  """Prunes this graph to its reachable states."""
  reachable_states = compute_reachable_states(dag)

  edge_subset = []
  for edge in dag.edges:
    if (
        edge.cost < INFINITE_COST
        and edge.source in reachable_states
        and edge.dest in reachable_states
    ):
      edge_subset.append(edge)

  return CompleteStateDAG(
      initial_state=dag.initial_state,
      edges=edge_subset,
      final_state=dag.final_state,
  )


@numba.extending.overload(prune_to_reachable)
def _prune_to_reachable_overload(dag: CompleteStateDAG):
  """Numba overload for `prune_to_reachable`."""
  del dag

  def impl(dag: CompleteStateDAG):
    relevant_edges = numba.typed.List()
    for edge in dag.edges:
      if edge.cost < INFINITE_COST:
        relevant_edges.append(edge)

    # Walk graph forwards (topologically)
    forward_reachable = numba.typed.Dict()
    forward_reachable[dag.initial_state] = True
    for edge in relevant_edges:
      if edge.source in forward_reachable:
        forward_reachable[edge.dest] = True

    # Walk graph in reverse (topologically)
    backward_reachable = numba.typed.Dict()
    backward_reachable[dag.final_state] = True
    for edge in relevant_edges[::-1]:
      if edge.dest in backward_reachable:
        backward_reachable[edge.source] = True

    edge_subset = numba.typed.List()
    for edge in relevant_edges:
      if edge.source in forward_reachable and edge.dest in backward_reachable:
        edge_subset.append(edge)

    return CompleteStateDAG(
        initial_state=dag.initial_state,
        edges=edge_subset,
        final_state=dag.final_state,
    )

  return impl


@numba.njit
def prune_to_reachable_jit(dag: CompleteStateDAG) -> CompleteStateDAG:
  """Alias for `prune_to_reachable` that always runs under JIT."""
  return prune_to_reachable(dag)


class PartialStateDAG(NamedTuple):
  """Partial DAG tracking states during graph construction.

  This class enforces that the graph being built is a DAG by requiring that
  edges be added in a topologically-sorted way: after adding an edge with source
  A and destination B, it is an error to add any edges with destination A.

  Attributes:
    initial_state: Initial state for the DAG.
    edges: List of edges.
    seen_outgoing: Dictionary that tracks which states we have seen an outgoing
      edge for, and the most recent destination seen.
  """

  initial_state: State
  edges: list[Edge]
  seen_outgoing: dict[State, State]


def partial_state_dag_starting_from(initial_state: State) -> PartialStateDAG:
  """Creates a new PartialStateDAG from an initial state."""
  return PartialStateDAG(
      initial_state=initial_state, edges=[], seen_outgoing={}
  )


EDGE_TRACEBACKS_ENABLED = False


@contextlib.contextmanager
def adding_edge_tracebacks():
  """Context manager to add tracebacks to edges."""
  global EDGE_TRACEBACKS_ENABLED
  EDGE_TRACEBACKS_ENABLED = True
  try:
    yield
  finally:
    EDGE_TRACEBACKS_ENABLED = False


def partial_state_dag_add_edge(dag: PartialStateDAG, edge: Edge):
  """Adds an edge to the DAG."""
  if edge.dest in dag.seen_outgoing:
    raise ValueError(
        (
            f"Incoming edge {edge} to {edge.dest} added after an outgoing edge"
            f" from {edge.dest} to {dag.seen_outgoing[edge.source]}"
        ),
    )

  if EDGE_TRACEBACKS_ENABLED:
    # Extract only the direct caller of this function.
    tb_frame = traceback.extract_stack(limit=2)[0]
    if isinstance(edge.info, dict):
      new_info = {**edge.info, "traceback": tb_frame}
    else:
      new_info = {"original_info": edge.info, "traceback": tb_frame}
    edge = Edge(
        source=edge.source,
        dest=edge.dest,
        cost=edge.cost,
        required_assignment=edge.required_assignment,
        info=new_info,
    )

  dag.seen_outgoing[edge.source] = edge.dest
  dag.edges.append(edge)


CHECK_ORDERING_UNDER_JIT = False


@numba.extending.overload(partial_state_dag_add_edge)
def _partial_state_dag_add_edge_overload(dag: PartialStateDAG, edge: Edge):
  """Numba overload for `partial_state_dag_add_edge`."""
  del dag, edge

  if CHECK_ORDERING_UNDER_JIT:

    def impl(dag: PartialStateDAG, edge: Edge):
      """Adds an edge to the DAG."""
      if edge.dest in dag.seen_outgoing:
        numba_raise.safe_raise(
            ValueError,
            (
                "Incoming edge",
                edge,
                "to",
                edge.dest,
                "added after an outgoing edge from",
                edge.dest,
                "to",
                dag.seen_outgoing[edge.source],
            ),
        )

      dag.seen_outgoing[edge.source] = edge.dest
      # Note: this line may cast `edge` to an optional type if any of its fields
      # were None. This is handled by the import of `patch_namedtuple_cast``.
      dag.edges.append(edge)

  else:

    def impl(dag: PartialStateDAG, edge: Edge):
      """Adds an edge to the DAG, without adding it to `seen_outgoing`."""
      dag.edges.append(edge)

  return impl


def partial_state_dag_finish(
    dag: PartialStateDAG, final_state: State
) -> CompleteStateDAG:
  """Finishes construction, producing a complete DAG."""
  return CompleteStateDAG(
      initial_state=dag.initial_state,
      edges=dag.edges,
      final_state=final_state,
  )


def strip_traceback_from_edge(edge: Edge) -> Edge:
  """Removes edge tracebacks from an edge if they were added."""
  if isinstance(edge.info, dict):
    if "original_info" in edge.info:
      original_info = edge.info["original_info"]
    elif "traceback" in edge.info:
      original_info = dict(edge.info)
      del original_info["traceback"]
    else:
      return edge
  else:
    return edge
  return Edge(
      source=edge.source,
      dest=edge.dest,
      cost=edge.cost,
      required_assignment=edge.required_assignment,
      info=original_info,
  )


@numba.njit
def prune_unreachable_and_rewrite_states(
    dag: CompleteStateDAG,
    scratch_table: np.NDArray[np.int32] = None,
) -> CompleteStateDAG:
  """Prunes unreachable edges and rewrite states to consecutive integers.

  This function is designed to be as fast as possible in exchange for using
  a fairly large amount of scratch memory. After running this function, other
  DAG processing steps (in particular, packing) should run much faster as well,
  since all states are mapped to unique consecutive integers, making them
  fast to insert and look up in dictionaries.

  Note that, since this method rewrites state identity, the graph may no longer
  be renderable afterward, since renderers often use the state identity to
  determine how the state should be drawn.

  Args:
    dag: The DAG to prune and rewrite.
    scratch_table: Preallocated scratch table, which must be a 1D array of
      integers. Can be allocated using `np.full((2**25,), -1, dtype=np.int32)`.

  Returns:
    Equivalent DAG to `dag`, but only including reachable states, and with
    consecutive integer states.
  """
  # We want to assign consecutive IDs to states, but do not want to have to
  # spend a lot of time accessing memory. We accomplish this by augmenting
  # a dictionary with a hand-written hash table, and preferentially inserting
  # things into the hash table, backed by the scratch space. If the scratch
  # space is large enough, we can ensure that almost every lookup maps directly
  # to the node ID of the desired state. When conflicts do arise, we evict the
  # older state and instead insert it into a fallback dictionary.

  state_lookup_table = scratch_table
  scratch_space_size = len(state_lookup_table)
  state_lookup_table[:] = -1

  states_by_index = numba.typed.List()
  fallback_lookup_table = numba.typed.Dict()

  # evictions = np.zeros((), dtype=np.int32)
  # fallback_lookups = np.zeros((), dtype=np.int32)
  # fallback_failures = np.zeros((), dtype=np.int32)
  # hotpath_lookups = np.zeros((), dtype=np.int32)

  def lookup_id_for_state(state):
    state_hash = hash(state)
    state_hash_trunc = state_hash % scratch_space_size
    looked_up_index = state_lookup_table[state_hash_trunc]
    if looked_up_index == -1:
      # Missing entry
      # hotpath_lookups[()] += 1
      return -1
    elif state == states_by_index[looked_up_index]:
      # hotpath_lookups[()] += 1
      return looked_up_index
    elif state in fallback_lookup_table:
      # fallback_lookups[()] += 1
      return fallback_lookup_table[state]
    else:
      # fallback_failures[()] += 1
      return -1

  def lookup_or_create_id_for_state(state):
    state_hash = hash(state)
    state_hash_trunc = state_hash % scratch_space_size
    looked_up_index = state_lookup_table[state_hash_trunc]
    if looked_up_index == -1:
      # Missing entry. Create.
      next_index = len(states_by_index)
      states_by_index.append(state)
      state_lookup_table[state_hash_trunc] = next_index
      return next_index
    else:
      looked_up_state = states_by_index[looked_up_index]
      if state == looked_up_state:
        # Already present.
        return looked_up_index
      else:
        # evictions[()] += 1
        # Conflict! Evict the old state since it's less likely to be active.
        fallback_lookup_table[looked_up_state] = looked_up_index
        next_index = len(states_by_index)
        states_by_index.append(state)
        state_lookup_table[state_hash_trunc] = next_index
        return next_index

  # First pass: Iterate through edges, and assign integer indices to states that
  # are reachable from the start node.
  lookup_or_create_id_for_state(dag.initial_state)
  rewritten_edge_triples = numba.typed.List()
  for original_edge_index, edge in enumerate(dag.edges):
    if edge.cost >= INFINITE_COST:
      continue
    source_id = lookup_id_for_state(edge.source)
    if source_id == -1:
      # The source was not reachable!
      continue
    # Allocate an ID for the dest if necessary.
    dest_id = lookup_or_create_id_for_state(edge.dest)
    rewritten_edge_triples.append((source_id, dest_id, original_edge_index))

  # Second pass: Iterate through rewritten edges in reverse, and determine which
  # states are reachable from the end.
  reachable_from_end = np.zeros((len(states_by_index),), dtype=np.bool_)
  reachable_from_end[lookup_or_create_id_for_state(dag.final_state)] = True
  for i in range(len(rewritten_edge_triples) - 1, -1, -1):
    (source_id, dest_id, original_edge_index) = rewritten_edge_triples[i]
    if reachable_from_end[dest_id]:
      reachable_from_end[source_id] = True

  if not reachable_from_end[lookup_id_for_state(dag.initial_state)]:
    raise ValueError("No path from start to end")

  # Third pass: Consolidate state IDs to skip any unreachable ones, so that they
  # remain consecutive after pruning.
  # We do this using `cumsum`, since `cumsum` counts the number of
  # previously-reachable states.
  compact_rewrite_map = np.cumsum(reachable_from_end) - 1
  final_edges = numba.typed.List()
  for source_id, dest_id, original_edge_index in rewritten_edge_triples:
    if reachable_from_end[dest_id]:
      edge = dag.edges[original_edge_index]
      final_edges.append(
          Edge(
              source=compact_rewrite_map[source_id],
              dest=compact_rewrite_map[dest_id],
              cost=edge.cost,
              required_assignment=edge.required_assignment,
              info=edge.info,
          )
      )

  # print("evictions", evictions[()])
  # print("hotpath_lookups", hotpath_lookups[()])
  # print("fallback_lookups", fallback_lookups[()])
  # print("fallback_failures", fallback_failures[()])
  return CompleteStateDAG(
      initial_state=compact_rewrite_map[lookup_id_for_state(dag.initial_state)],
      edges=final_edges,
      final_state=compact_rewrite_map[lookup_id_for_state(dag.final_state)],
  )
