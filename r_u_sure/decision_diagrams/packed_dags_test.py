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

"""Tests for packed state DAGs."""

from absl.testing import absltest
from absl.testing import parameterized
import numba
import numpy as np
from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.decision_diagrams import gated_state_dag_test_lib
from r_u_sure.decision_diagrams import packed_dags
from r_u_sure.testing import test_flags

Edge = gated_state_dag.Edge
SharedVariableAssignment = gated_state_dag.SharedVariableAssignment
States = gated_state_dag_test_lib.States


def make_packed_dag():
  """Helper to construct a packed example DAG."""
  dag = gated_state_dag_test_lib.build_example_dag()
  pruned = gated_state_dag.prune_to_reachable(dag)
  packed, packing_data = packed_dags.convert_dag_to_packed(
      pruned,
      missing_assignment_value='_UNSET',
      variable_value_ordering=None,
  )
  return packed, packing_data


class PackedDagTest(parameterized.TestCase):

  def test_consistent_state_boundaries(self):
    packed, packing_data = make_packed_dag()
    for i in range(len(packed.state_variable_boundary_indices) - 1):
      a = packed.state_variable_boundary_indices[i]
      b = packed.state_variable_boundary_indices[i + 1]
      for state in packing_data.tagged_states[a:b]:
        self.assertEqual(state.last_variable_index, (i - 1), (i, state))

    self.assertEqual(packed.state_variable_boundary_indices[0], 0)
    self.assertEqual(  # pylint: disable=g-generic-assert
        packed.state_variable_boundary_indices[-1],
        len(packing_data.tagged_states),
    )

  def test_consistent_edge_groups(self):
    packed, packing_data = make_packed_dag()
    for i in range(len(packed.edge_group_boundary_indices) - 1):
      a = packed.edge_group_boundary_indices[i]
      b = packed.edge_group_boundary_indices[i + 1]
      for edge in packing_data.tagged_edges[a:b]:
        if i % 2 == 0:
          self.assertEqual(edge.source.last_variable_index, i // 2 - 1)
          self.assertEqual(edge.dest.last_variable_index, i // 2 - 1)
          self.assertIsNone(edge.required_assignment)
        else:
          self.assertEqual(edge.source.last_variable_index, i // 2 - 1)
          self.assertEqual(edge.dest.last_variable_index, i // 2)
          self.assertEqual(
              edge.required_assignment.key,
              packing_data.variable_keys[i // 2],
          )

  def test_consistent_packed_edges(self):
    packed, packing_data = make_packed_dag()
    self.assertLen(
        packing_data.tagged_edges,
        packed.packed_edge_matrix.shape[0],
    )
    for i in range(len(packing_data.tagged_edges)):
      edge = packing_data.tagged_edges[i]
      packed_edge = packed.packed_edge_matrix[i]
      self.assertEqual(edge.source, packing_data.tagged_states[packed_edge[0]])
      self.assertEqual(edge.dest, packing_data.tagged_states[packed_edge[1]])
      self.assertEqual(edge.cost, packed_edge[2])
      if edge.required_assignment is None:
        self.assertEqual(packed_edge[3], packed_edge[4], -1)
      else:
        self.assertEqual(
            edge.required_assignment.key,
            packing_data.variable_keys[packed_edge[3]],
        )
        self.assertEqual(
            edge.required_assignment.value,
            packing_data.variable_values[packed_edge[4]],
        )

  def test_consistent_edge_tagging(self):
    _, packing_data = make_packed_dag()
    counts = {k: 0 for k in packing_data.original_edges}
    for edge in packing_data.tagged_edges:
      if edge.info == -1:
        self.assertEqual(edge.source.original_state, edge.dest.original_state)
        self.assertIsNotNone(edge.required_assignment)
      else:
        orig_edge = packing_data.original_edges[edge.info]
        counts[orig_edge] += 1
        self.assertEqual(edge.source.original_state, orig_edge.source)
        self.assertEqual(edge.dest.original_state, orig_edge.dest)
        self.assertEqual(edge.cost, orig_edge.cost)
        self.assertEqual(
            edge.required_assignment, orig_edge.required_assignment
        )

    self.assertEqual(counts, {k: 1 for k in packing_data.original_edges})

  @parameterized.named_parameters(
      dict(testcase_name='numba', use_numba=True),
      dict(testcase_name='pure_python', use_numba=False),
  )
  def test_min_marginals(self, use_numba):
    if use_numba and test_flags.SKIP_JIT_TESTS.value:
      self.skipTest('Skipping JIT test variant')

    packed, packing_data = make_packed_dag()
    key_index = 2
    self.assertEqual(packing_data.variable_keys[key_index], 'y')
    value_index = 6
    self.assertEqual(packing_data.variable_values[value_index], 'A')

    def go(packed):
      # Construct a table.
      memo = packed_dags.empty_unpenalized_table(packed)
      initial_tables = (
          packed_dags.masked_prefixes(memo),
          packed_dags.masked_suffixes(memo),
      )

      # Validate the table.
      packed_dags.make_fully_valid(memo)
      valid_tables = (
          packed_dags.masked_prefixes(memo),
          packed_dags.masked_suffixes(memo),
      )
      valid_bounds = (
          np.copy(memo.prefixes_valid_ending_at_tag),
          np.copy(memo.suffixes_valid_starting_from_tag),
      )

      # Compute cost and min marginals
      costs = (
          packed_dags.compute_minimal_cost(
              memo,
              strategy=packed_dags.MinimalCostComputationStrategy.FROM_MIDDLE,
          ),
          packed_dags.compute_minimal_cost(
              memo, strategy=packed_dags.MinimalCostComputationStrategy.AT_START
          ),
          packed_dags.compute_minimal_cost(
              memo, strategy=packed_dags.MinimalCostComputationStrategy.AT_END
          ),
      )
      min_marginals = packed_dags.compute_all_min_marginals(memo)

      # Change penalties. This invalidates part of our table.
      # We are penalizing the assignment `y`: 'A'.
      cur_penalties = np.zeros((9,), packed_dags.COST_DTYPE)
      cur_penalties[value_index] = 40
      packed_dags.set_penalties(memo, key_index, cur_penalties)

      perturbed_partial_tables = (
          packed_dags.masked_prefixes(memo),
          packed_dags.masked_suffixes(memo),
      )
      perturbed_bounds = (
          np.copy(memo.prefixes_valid_ending_at_tag),
          np.copy(memo.suffixes_valid_starting_from_tag),
      )

      # Recompute marginals. This recomputes the changed parts of the table.
      perturbed_costs = (
          packed_dags.compute_minimal_cost(
              memo,
              strategy=packed_dags.MinimalCostComputationStrategy.FROM_MIDDLE,
          ),
          packed_dags.compute_minimal_cost(
              memo, strategy=packed_dags.MinimalCostComputationStrategy.AT_START
          ),
          packed_dags.compute_minimal_cost(
              memo, strategy=packed_dags.MinimalCostComputationStrategy.AT_END
          ),
      )
      perturbed_min_marginals = packed_dags.compute_all_min_marginals(memo)
      perturbed_full_tables = (
          packed_dags.masked_prefixes(memo),
          packed_dags.masked_suffixes(memo),
      )

      return (
          initial_tables,
          valid_tables,
          valid_bounds,
          costs,
          min_marginals,
          perturbed_partial_tables,
          perturbed_bounds,
          perturbed_costs,
          perturbed_min_marginals,
          perturbed_full_tables,
      )

    if use_numba:
      go = numba.jit(go, nopython=True)

    (
        (initial_prefixes, initial_suffixes),
        (valid_prefixes, valid_suffixes),
        valid_bounds,
        costs,
        min_marginals,
        (perturbed_partial_prefixes, perturbed_partial_suffixes),
        perturbed_bounds,
        perturbed_costs,
        perturbed_min_marginals,
        (perturbed_full_prefixes, perturbed_full_suffixes),
    ) = go(packed)

    bad = packed_dags.INVALID_COST
    inf = packed_dags.INFINITE_COST

    np.testing.assert_array_equal(
        initial_prefixes, np.full((packed.num_tagged_states,), bad)
    )
    np.testing.assert_array_equal(
        initial_suffixes, np.full((packed.num_tagged_states,), bad)
    )
    np.testing.assert_array_equal(
        valid_prefixes,
        np.array(
            [0, 100, 101, 103, 103, 101, 101, 103, 101, 103, 101, 101, 103]
            + [113, 101, 101, 113, 116, 113, 113, 113]
        ),
    )
    np.testing.assert_array_equal(
        valid_suffixes,
        np.array(
            [113, 13, 12, 10, 10, 15, 15, 10, 15, 10, 15, 25, 20]
            + [0, 15, 25, 0, 0, 0, 0, 0]
        ),
    )
    self.assertEqual(valid_bounds, (4, -1))
    self.assertEqual(costs, (113, 113, 113))
    np.testing.assert_array_equal(
        min_marginals,
        np.array([
            [113, inf, inf, 114, inf, inf, inf, inf, inf],
            [113, inf, inf, 116, inf, inf, inf, inf, inf],
            [inf, inf, inf, inf, inf, inf, 113, 116, inf],
            [inf, 113, inf, inf, 123, inf, inf, inf, 116],
            [inf, inf, 116, inf, inf, 126, inf, inf, 113],
        ]),
    )
    np.testing.assert_array_equal(
        perturbed_partial_prefixes,
        np.array(
            [0, 100, 101, 103, 103, 101, 101, 103, 101, bad, bad, bad, bad]
            + [bad, bad, bad, bad, bad, bad, bad, bad]
        ),
    )
    np.testing.assert_array_equal(
        perturbed_partial_suffixes,
        np.array(
            [bad, bad, bad, bad, bad, bad, bad, bad, bad, 10, 15, 25, 20]
            + [0, 15, 25, 0, 0, 0, 0, 0]
        ),
    )
    self.assertEqual(perturbed_bounds, (key_index - 1, key_index))
    self.assertEqual(perturbed_costs, (116, 116, 116))
    np.testing.assert_array_equal(
        perturbed_min_marginals,
        np.array([
            [116, inf, inf, 117, inf, inf, inf, inf, inf],
            [153, inf, inf, 116, inf, inf, inf, inf, inf],
            [inf, inf, inf, inf, inf, inf, 153, 116, inf],
            [inf, 153, inf, inf, 163, inf, inf, inf, 116],
            [inf, inf, 116, inf, inf, 126, inf, inf, 153],
        ]),
    )
    np.testing.assert_array_equal(
        perturbed_full_prefixes,
        np.array(
            [0, 100, 101, 103, 103, 101, 101, 103, 101, 143, 101, 101, 143]
            + [153, 101, 101, 153, 116, 116, 116, 116]
        ),
    )
    np.testing.assert_array_equal(
        perturbed_full_suffixes,
        np.array(
            [116, 16, 15, 50, 50, 15, 15, 50, 15, 10, 15, 25, 20]
            + [0, 15, 25, 0, 0, 0, 0, 0]
        ),
    )

  def test_extract_path(self):
    packed, packing_data = make_packed_dag()
    memo = packed_dags.empty_unpenalized_table(packed)

    (
        unpenalized_path,
        assignments,
    ) = packed_dags.extract_minimal_cost_path_and_assignments(
        memo, packing_data
    )
    self.assertEqual(
        [
            Edge(
                source=States.START_1,
                dest=States.START_2,
                cost=100,
                required_assignment=None,
                info=None,
            ),
            Edge(
                source=States.START_2,
                dest=States.START_3,
                cost=1,
                required_assignment=SharedVariableAssignment(
                    key='w', value='1'
                ),
                info=None,
            ),
            Edge(
                source=States.START_3,
                dest=States.VARIANT_1_A,
                cost=2,
                required_assignment=SharedVariableAssignment(
                    key='x', value='1'
                ),
                info=None,
            ),
            Edge(
                source=States.VARIANT_1_A,
                dest=States.VARIANT_1_B,
                cost=0,
                required_assignment=None,
                info=None,
            ),
            Edge(
                source=States.VARIANT_1_B,
                dest=States.VARIANT_1_C,
                cost=0,
                required_assignment=None,
                info=None,
            ),
            Edge(
                source=States.VARIANT_1_C,
                dest=States.VARIANT_1_D,
                cost=0,
                required_assignment=SharedVariableAssignment(
                    key='y', value='A'
                ),
                info=None,
            ),
            Edge(
                source=States.VARIANT_1_D,
                dest=States.VARIANT_1_F,
                cost=10,
                required_assignment=SharedVariableAssignment(
                    key='z-v1', value='10'
                ),
                info=None,
            ),
            Edge(
                source=States.VARIANT_1_F,
                dest=States.END_1,
                cost=0,
                required_assignment=None,
                info=None,
            ),
            Edge(
                source=States.END_1,
                dest=States.END_2,
                cost=0,
                required_assignment=None,
                info=None,
            ),
            Edge(
                source=States.END_2,
                dest=States.END_3,
                cost=0,
                required_assignment=None,
                info=None,
            ),
        ],
        unpenalized_path,
    )
    self.assertEqual(
        {'w': '1', 'x': '1', 'y': 'A', 'z-v1': '10', 'z-v2': '_UNSET'},
        assignments,
    )

    # We are penalizing the assignment `y`: 'A'. This blocks the edge
    # VARIANT_1_C -> VARIANT_1_D, causing the optimal path to use VARIANT_2
    # instead.
    key_index = 2
    self.assertEqual(packing_data.variable_keys[key_index], 'y')
    value_index = 6
    self.assertEqual(packing_data.variable_values[value_index], 'A')
    cur_penalties = np.zeros((9,), packed_dags.COST_DTYPE)
    cur_penalties[value_index] = 40
    packed_dags.set_penalties(memo, key_index, cur_penalties)

    (
        penalized_path,
        penalized_assignments,
    ) = packed_dags.extract_minimal_cost_path_and_assignments(
        memo, packing_data
    )
    self.assertEqual(
        [
            Edge(
                source=States.START_1,
                dest=States.START_2,
                cost=100,
                required_assignment=None,
                info=None,
            ),
            Edge(
                source=States.START_2,
                dest=States.START_3,
                cost=1,
                required_assignment=SharedVariableAssignment(
                    key='w', value='1'
                ),
                info=None,
            ),
            Edge(
                source=States.START_3,
                dest=States.VARIANT_2_A,
                cost=0,
                required_assignment=SharedVariableAssignment(
                    key='x', value='2'
                ),
                info=None,
            ),
            Edge(
                source=States.VARIANT_2_A,
                dest=States.VARIANT_2_B,
                cost=0,
                required_assignment=None,
                info=None,
            ),
            Edge(
                source=States.VARIANT_2_B,
                dest=States.VARIANT_2_C,
                cost=0,
                required_assignment=None,
                info=None,
            ),
            Edge(
                source=States.VARIANT_2_C,
                dest=States.VARIANT_2_D,
                cost=0,
                required_assignment=SharedVariableAssignment(
                    key='y', value='B'
                ),
                info=None,
            ),
            Edge(
                source=States.VARIANT_2_D,
                dest=States.VARIANT_2_F,
                cost=15,
                required_assignment=SharedVariableAssignment(
                    key='z-v2', value='15'
                ),
                info=None,
            ),
            Edge(
                source=States.VARIANT_2_F,
                dest=States.END_1,
                cost=0,
                required_assignment=None,
                info=None,
            ),
            Edge(
                source=States.END_1,
                dest=States.END_2,
                cost=0,
                required_assignment=None,
                info=None,
            ),
            Edge(
                source=States.END_2,
                dest=States.END_3,
                cost=0,
                required_assignment=None,
                info=None,
            ),
        ],
        penalized_path,
    )
    self.assertEqual(
        {'w': '1', 'x': '2', 'y': 'B', 'z-v1': '_UNSET', 'z-v2': '15'},
        penalized_assignments,
    )

  def test_numba_packing(self):
    if test_flags.SKIP_JIT_TESTS.value:
      self.skipTest('Skipping JIT test variant')

    dag = gated_state_dag_test_lib.build_example_dag()
    pruned = gated_state_dag.prune_to_reachable(dag)

    py_packed, py_packing_data = packed_dags.convert_dag_to_packed(
        pruned,
        missing_assignment_value='_UNSET',
        variable_value_ordering=None,
    )

    (
        convert_dag_to_packed__specialized,
        conversion_fn,
    ) = packed_dags.make_specialized_fn__convert_dag_to_packed(
        with_numba=True,
        example_state=States.START_1,
        example_variable_key='y',
        example_variable_value='A',
        example_info=None,
    )
    typed_pruned = conversion_fn(pruned)
    numba_packed, numba_packing_data = convert_dag_to_packed__specialized(
        typed_pruned,
        missing_assignment_value='_UNSET',
        variable_value_ordering=None,
    )

    self.assertEqual(
        py_packed.num_tagged_states, numba_packed.num_tagged_states
    )
    self.assertEqual(
        py_packed.num_variable_keys, numba_packed.num_variable_keys
    )
    self.assertEqual(
        py_packed.num_variable_values, numba_packed.num_variable_values
    )
    np.testing.assert_array_equal(
        py_packed.packed_edge_matrix, numba_packed.packed_edge_matrix
    )
    np.testing.assert_array_equal(
        py_packed.edge_group_boundary_indices,
        numba_packed.edge_group_boundary_indices,
    )
    np.testing.assert_array_equal(
        py_packed.state_variable_boundary_indices,
        numba_packed.state_variable_boundary_indices,
    )

    self.assertListEqual(
        py_packing_data.variable_keys, list(numba_packing_data.variable_keys)
    )
    self.assertListEqual(
        py_packing_data.variable_values,
        list(numba_packing_data.variable_values),
    )
    self.assertListEqual(
        py_packing_data.original_states,
        list(numba_packing_data.original_states),
    )
    self.assertListEqual(
        py_packing_data.original_edges, list(numba_packing_data.original_edges)
    )
    self.assertListEqual(
        py_packing_data.tagged_states, list(numba_packing_data.tagged_states)
    )
    self.assertListEqual(
        py_packing_data.tagged_edges, list(numba_packing_data.tagged_edges)
    )
    numba_outgoing = (
        numba_packing_data.outgoing_edge_indices_for_tagged_state_index
    )
    self.assertListEqual(
        py_packing_data.outgoing_edge_indices_for_tagged_state_index,
        [list(indices) for indices in numba_outgoing],
    )


if __name__ == '__main__':
  absltest.main()
