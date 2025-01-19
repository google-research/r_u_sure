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

"""Tests for gated state DAG construction."""

from absl.testing import absltest
from absl.testing import parameterized
import numba
import numpy as np
from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.decision_diagrams import gated_state_dag_test_lib
from r_u_sure.numba_helpers import numba_type_util


class GatedStateDagTest(parameterized.TestCase):

  def test_construction_and_reachability(self):
    dag = gated_state_dag_test_lib.build_example_dag()

    reachable_states = gated_state_dag.compute_reachable_states(dag)
    expected_reachable = set()
    for item in gated_state_dag_test_lib.States:
      if item not in {
          gated_state_dag_test_lib.States.SPECIAL_UNREACHABLE_1,
          gated_state_dag_test_lib.States.SPECIAL_UNREACHABLE_2,
          gated_state_dag_test_lib.States.SPECIAL_DEADEND,
      }:
        expected_reachable.add(item)

    self.assertEqual(reachable_states, expected_reachable)

    pruned_graph = gated_state_dag.prune_to_reachable(dag)
    seen_states = set()
    for edge in pruned_graph.edges:
      seen_states.add(edge.source)
      seen_states.add(edge.dest)

    self.assertEqual(seen_states, expected_reachable)

  def test_numba_reachability(self):
    dag = gated_state_dag_test_lib.build_example_dag()
    edge_type = numba.typeof(
        gated_state_dag.Edge(
            source=gated_state_dag_test_lib.States.START_1,
            dest=gated_state_dag_test_lib.States.START_1,
            cost=0.0,
            required_assignment=numba_type_util.PretendOptional(
                gated_state_dag.SharedVariableAssignment(key="a", value="b")
            ),
            info=None,
        )
    )
    numba_dag = gated_state_dag.CompleteStateDAG(
        initial_state=dag.initial_state,
        edges=numba.typed.List.empty_list(item_type=edge_type),
        final_state=dag.final_state,
    )
    for edge in dag.edges:
      # Edges must be appended one-by-one so that Numba casts each one
      # correctly.
      numba_dag.edges.append(edge)

    expected_reachable = set()
    for item in gated_state_dag_test_lib.States:
      if item not in {
          gated_state_dag_test_lib.States.SPECIAL_UNREACHABLE_1,
          gated_state_dag_test_lib.States.SPECIAL_UNREACHABLE_2,
          gated_state_dag_test_lib.States.SPECIAL_DEADEND,
      }:
        expected_reachable.add(item)

    pruned_graph = gated_state_dag.prune_to_reachable_jit(numba_dag)
    seen_states = set()
    for edge in pruned_graph.edges:
      seen_states.add(edge.source)
      seen_states.add(edge.dest)

    self.assertEqual(seen_states, expected_reachable)

  @parameterized.named_parameters(
      dict(
          testcase_name="large_table",
          table_size=2**20,
      ),
      dict(
          testcase_name="medium_table",
          table_size=16,
      ),
      dict(
          testcase_name="small_table",
          table_size=1,
      ),
  )
  def test_reachability_with_rewrite(self, table_size):
    dag = gated_state_dag_test_lib.build_example_dag()
    edge_type = numba.typeof(
        gated_state_dag.Edge(
            source=gated_state_dag_test_lib.States.START_1,
            dest=gated_state_dag_test_lib.States.START_1,
            cost=0.0,
            required_assignment=numba_type_util.PretendOptional(
                gated_state_dag.SharedVariableAssignment(key="a", value="b")
            ),
            info=None,
        )
    )
    numba_dag = gated_state_dag.CompleteStateDAG(
        initial_state=dag.initial_state,
        edges=numba.typed.List.empty_list(item_type=edge_type),
        final_state=dag.final_state,
    )
    for edge in dag.edges:
      # Edges must be appended one-by-one so that Numba casts each one
      # correctly.
      numba_dag.edges.append(edge)

    pruned_graph = gated_state_dag.prune_to_reachable_jit(numba_dag)
    pruned_and_rewritten_graph = (
        gated_state_dag.prune_unreachable_and_rewrite_states(
            numba_dag, scratch_table=np.full((table_size,), -1, dtype=np.int32)
        )
    )

    # Make sure the new graph contains consecutive integers as states.
    all_states = set()
    for edge in pruned_and_rewritten_graph.edges:
      all_states.add(edge.source)
      all_states.add(edge.dest)

    self.assertSequenceEqual(sorted(all_states), range(len(all_states)))

    # Make sure there exists a mapping from new states to old states such that
    # the returned eges are identical.
    self.assertEqual(
        len(pruned_and_rewritten_graph.edges), len(pruned_graph.edges)
    )
    new_to_old_state_map = {}
    for new_edge, old_edge in zip(
        pruned_and_rewritten_graph.edges, pruned_graph.edges
    ):
      if new_edge.source in new_to_old_state_map:
        self.assertEqual(new_to_old_state_map[new_edge.source], old_edge.source)
      else:
        new_to_old_state_map[new_edge.source] = old_edge.source

      if new_edge.dest in new_to_old_state_map:
        self.assertEqual(new_to_old_state_map[new_edge.dest], old_edge.dest)
      else:
        new_to_old_state_map[new_edge.dest] = old_edge.dest


if __name__ == "__main__":
  absltest.main()
