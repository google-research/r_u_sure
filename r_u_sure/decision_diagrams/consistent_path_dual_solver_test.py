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

"""Tests for consistent_path_dual_solver."""

from absl.testing import absltest
import numpy as np
from r_u_sure.decision_diagrams import consistent_path_dual_solver
from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.decision_diagrams import gated_state_dag_test_lib
from r_u_sure.decision_diagrams import packed_dags


class ConsistentPathDualSolverTest(absltest.TestCase):

  def test_tight_dual_bound(self):
    """Checks behavior when the dual bound is tight.

    This problem also includes a subproblem-specific variable to ensure that
    the indexing is handled correctly.
    """

    # Problem A strongly prefers shared_0=0, and weakly prefers the two
    # variables to be the same.
    subproblem_a = gated_state_dag_test_lib.dag_from_paths([
        ({"a_only_choice": 0, "shared_0": 0, "shared_1": 0}, 0.0),
        ({"a_only_choice": 1, "shared_0": 0, "shared_1": 1}, 1.0),
        ({"a_only_choice": 2, "shared_0": 1, "shared_1": 0}, 101.0),
        ({"a_only_choice": 3, "shared_0": 1, "shared_1": 1}, 100.0),
    ])

    # Problem Z has a medium preference for shared_0=1, a medium preference
    # for (0,0) over (0,1), and a small preference for (1,1) over (1,0).
    # It also has a large fixed cost.
    # (We use "z" so that the variable key sorts after the shared variables.)
    subproblem_z = gated_state_dag_test_lib.dag_from_paths([
        ({"shared_0": 0, "shared_1": 0, "z_only_choice": 4}, 1020.0),
        ({"shared_0": 0, "shared_1": 1, "z_only_choice": 5}, 1010.0),
        ({"shared_0": 1, "shared_1": 0, "z_only_choice": 6}, 1000.0),
        ({"shared_0": 1, "shared_1": 1, "z_only_choice": 7}, 1001.0),
    ])

    # Build a system.
    packed_a, conversion_data_a = packed_dags.convert_dag_to_packed(
        dag=subproblem_a,
        missing_assignment_value=-1,
        variable_value_ordering=list(range(8)),
    )
    packed_z, conversion_data_z = packed_dags.convert_dag_to_packed(
        dag=subproblem_z,
        missing_assignment_value=-1,
        variable_value_ordering=list(range(8)),
    )
    system = consistent_path_dual_solver.make_system([
        (packed_a, conversion_data_a),
        (packed_z, conversion_data_z),
    ])

    # Variables should appear in sorted order.
    self.assertEqual(
        ["a_only_choice", "shared_0", "shared_1", "z_only_choice"],
        system.variable_keys,
    )
    self.assertEqual(list(range(8)), system.variable_values)
    np.testing.assert_array_equal(
        system.data.dag_variable_index_map,
        np.array([[0, -1], [1, 0], [2, 1], [-1, 2]]),
    )

    # Extract min marginals. Based on these alone, it looks like
    # {shared_0:0, shared_1:0} is the best assignment (but it isn't).
    min_marginals = np.stack(
        [
            consistent_path_dual_solver.system_min_marginals(system.data, i)
            for i in range(4)
        ]
    )
    np.testing.assert_array_equal(
        min_marginals,
        np.array([
            # a_only_choice (plus min cost from subproblem z)
            [1000.0, 1001.0, 1101.0, 1100.0, np.inf, np.inf, np.inf, np.inf],
            # shared_0
            [1010.0, 1100.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            # shared_1
            [1000.0, 1002.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            # z_only_choice (plus min cost from subproblem a)
            [np.inf, np.inf, np.inf, np.inf, 1020.0, 1010.0, 1000.0, 1001.0],
        ]),
    )

    # Greedily extract a solution. We either commit early to a's best choice
    # or z's best choice, then are forced to follow it.
    (
        greedy_forward_assignments,
        greedy_forward_cost,
    ) = consistent_path_dual_solver.greedy_extract(
        system.data,
        direction=consistent_path_dual_solver.SweepDirection.FORWARD,
    )
    greedy_forward_assignments_dict = {
        system.variable_keys[k]: system.variable_values[v]
        for k, v in enumerate(greedy_forward_assignments)
    }
    # Greedy sweep will commit to "a_only_choice":0, which has the best min
    # marginal, then be forced to make consistent choices afterward.
    self.assertEqual(
        greedy_forward_assignments_dict,
        {"a_only_choice": 0, "shared_0": 0, "shared_1": 0, "z_only_choice": 4},
    )
    self.assertEqual(greedy_forward_cost, 1020.0)

    (
        greedy_reverse_assignments,
        greedy_reverse_cost,
    ) = consistent_path_dual_solver.greedy_extract(
        system.data,
        direction=consistent_path_dual_solver.SweepDirection.REVERSE,
    )
    greedy_reverse_assignments_dict = {
        system.variable_keys[k]: system.variable_values[v]
        for k, v in enumerate(greedy_reverse_assignments)
    }
    # Greedy sweep will commit to "z_only_choice":6, which has the best min
    # marginal, then be forced to make consistent choices afterward.
    self.assertEqual(
        greedy_reverse_assignments_dict,
        {"a_only_choice": 2, "shared_0": 1, "shared_1": 0, "z_only_choice": 6},
    )
    self.assertEqual(greedy_reverse_cost, 1101.0)

    # Run min marginal averaging sweeps.
    sweep_results = consistent_path_dual_solver.solve_system_with_sweeps(
        system.data
    )

    # Dual bound is now tight, and greedy decoding successfully extracts the
    # true best assignment.
    self.assertEqual(sweep_results.objective_at_step[-1], 1011.0)
    (
        converged_assignments,
        converged_cost,
    ) = consistent_path_dual_solver.greedy_extract(
        system.data,
        direction=consistent_path_dual_solver.SweepDirection.FORWARD,
    )
    converged_assignments_dict = {
        system.variable_keys[k]: system.variable_values[v]
        for k, v in enumerate(converged_assignments)
    }
    self.assertEqual(converged_cost, 1011.0)
    self.assertEqual(
        converged_assignments_dict,
        {"a_only_choice": 1, "shared_0": 0, "shared_1": 1, "z_only_choice": 5},
    )

  def test_infeasible_min_marginal_guess(self):
    """Tests that greedy decoding avoids infeasible assignments.

    This is important when min-marginal averaging can't find a tight dual
    bound, and the min marginal choices by themselves lead to an infeasible
    solution.
    """

    # First problem: (0,0) is better than (1,1) which is much better than
    # others.
    subproblem_a = gated_state_dag_test_lib.dag_from_paths([
        ({"shared_0": 0, "shared_1": 0}, 0.0),
        ({"shared_0": 0, "shared_1": 1}, 10.0),
        ({"shared_0": 1, "shared_1": 0}, 10.0),
        ({"shared_0": 1, "shared_1": 1}, 1.0),
    ])
    # Second problem: (0,0) is impossible.
    subproblem_b = gated_state_dag_test_lib.dag_from_paths([
        ({"shared_0": 0, "shared_1": 1}, 0.0),
        ({"shared_0": 1, "shared_1": 0}, 0.0),
        ({"shared_0": 1, "shared_1": 1}, 0.0),
    ])

    # Build a system.
    packed_a, conversion_data_a = packed_dags.convert_dag_to_packed(
        dag=subproblem_a,
        missing_assignment_value=-1,
        variable_value_ordering=list(range(2)),
    )
    packed_b, conversion_data_b = packed_dags.convert_dag_to_packed(
        dag=subproblem_b,
        missing_assignment_value=-1,
        variable_value_ordering=list(range(2)),
    )
    system = consistent_path_dual_solver.make_system([
        (packed_a, conversion_data_a),
        (packed_b, conversion_data_b),
    ])

    # Extract min marginals. Based on these alone, it looks like
    # {shared_0:0, shared_1:0} is the best assignment (but it isn't, since it's
    # impossible).
    min_marginals = np.stack(
        [
            consistent_path_dual_solver.system_min_marginals(system.data, i)
            for i in range(2)
        ]
    )
    np.testing.assert_array_equal(
        min_marginals,
        np.array([
            [0.0, 1.0],  # shared_0
            [0.0, 1.0],  # shared_1
        ]),
    )

    # Run min marginal averaging sweeps.
    # The dual bound is not tight for this problem, and we get stuck in a local
    # optimum. Note that the output is actually fractional even though no
    # assignment has a fractional cost; this is due to the corresponding
    # linear program relaxation having a non-integer solution.
    sweep_results = consistent_path_dual_solver.solve_system_with_sweeps(
        system.data
    )
    self.assertEqual(sweep_results.objective_at_step[-1], 0.5)

    # Min marginals now express total ignorance about the solution.
    min_marginals = np.stack(
        [
            consistent_path_dual_solver.system_min_marginals(system.data, i)
            for i in range(2)
        ]
    )
    np.testing.assert_array_equal(
        min_marginals,
        np.array([
            [0.5, 0.5],  # shared_0
            [0.5, 0.5],  # shared_1
        ]),
    )

    # Greedy decoding arbitrarily rounds down (due to numpy argmax behavior),
    # choosing shared_0=0. At this point it is constrained and still produces
    # a valid (but suboptimal) assignment.
    (
        greedy_forward_assignments,
        greedy_forward_cost,
    ) = consistent_path_dual_solver.greedy_extract(
        system.data,
        direction=consistent_path_dual_solver.SweepDirection.FORWARD,
    )
    greedy_forward_assignments_dict = {
        system.variable_keys[k]: system.variable_values[v]
        for k, v in enumerate(greedy_forward_assignments)
    }
    # Greedy sweep will commit to "a_only_choice":0, which has the best min
    # marginal, then be forced to make consistent choices afterward.
    self.assertEqual(
        greedy_forward_assignments_dict,
        {"shared_0": 0, "shared_1": 1},
    )
    self.assertEqual(greedy_forward_cost, 10.0)

  def test_partially_commit_selection(self):
    """Checks that `partially_commit` identifies the right variables."""
    # Build a subproblem with some obviously bad assignments.
    subproblem_a = gated_state_dag_test_lib.dag_from_paths([
        ({"a": 0, "b": 0, "c": 0}, 0.0),
        # alternatives for "a"
        ({"a": 1, "b": 0, "c": 0}, 1.0),
        ({"a": 2, "b": 0, "c": 0}, 1.0),
        ({"a": 3, "b": 0, "c": 0}, np.inf),
        ({"a": 4, "b": 0, "c": 0}, 100.0),
        # alternatives for "b"
        ({"a": 0, "b": 1, "c": 0}, 2.0),
        ({"a": 0, "b": 2, "c": 0}, 79.0),
        ({"a": 0, "b": 3, "c": 0}, 80.0),
        ({"a": 0, "b": 4, "c": 0}, 100.0),
        # alternatives for "c"
        ({"a": 0, "b": 0, "c": 1}, 4.0),
        ({"a": 0, "b": 0, "c": 2}, 100.0),
        ({"a": 0, "b": 0, "c": 3}, 100.0),
        ({"a": 0, "b": 0, "c": 4}, 100.0),
    ])
    # Make a system. (This system is pretty trivial, as we can just read off
    # the solution for the min marginals, but it's enough for our purposes.)
    subproblem_a = gated_state_dag.prune_to_reachable(subproblem_a)
    packed_a, conversion_data_a = packed_dags.convert_dag_to_packed(
        dag=subproblem_a,
        missing_assignment_value=-1,
        variable_value_ordering=list(range(5)),
    )
    system = consistent_path_dual_solver.make_system(
        [(packed_a, conversion_data_a)]
    )
    # Forbid bad min marginals, targeting about two assignments per variable,
    # but not very strictly. This will cause us to forbid the 6 assignments
    # whose cost is 80 or above, and also the one of cost 79, since it is
    # within `gap_strictness` of 80.
    # We also forbid the assignment "a":3 which has infinite cost, but note that
    # this doesn't count as one of the 6 assignments, since it was already
    # forbidden.
    status = consistent_path_dual_solver.partially_commit(
        system.data,
        direction=consistent_path_dual_solver.SweepDirection.FORWARD,
        target_commitments_per_variable=1.9,
        gap_strictness=0.9,
    )
    self.assertEqual(
        status,
        consistent_path_dual_solver.PartialCommitmentStatus.MADE_PARTIAL_ASSIGNMENT,
    )
    np.testing.assert_array_equal(
        system.data.dag_tables[0].penalties,
        np.array([
            [0.0, 0.0, 0.0, np.inf, np.inf],  # a
            [0.0, 0.0, np.inf, np.inf, np.inf],  # b
            [0.0, 0.0, np.inf, np.inf, np.inf],  # c
        ]),
    )

    # Forbid another 6 assignments. At this point it should detect that there
    # aren't that many remaining alternatives to forbid, and redirect to
    # greedy decoding.
    status = consistent_path_dual_solver.partially_commit(
        system.data,
        direction=consistent_path_dual_solver.SweepDirection.FORWARD,
        target_commitments_per_variable=1.9,
        gap_strictness=0.9,
    )
    self.assertEqual(
        status,
        consistent_path_dual_solver.PartialCommitmentStatus.NEEDS_GREEDY_EXTRACT,
    )

  def test_partially_commit_feasibility(self):
    """Checks that `partially_commit` reacts to sudden changes in min marginals."""
    # First problem: More variables being assigned 1 is better.
    subproblem_a = gated_state_dag_test_lib.dag_from_paths([
        ({"a": 0, "b": 0, "c": 0}, 300.0),
        ({"a": 1, "b": 0, "c": 0}, 200.0),
        ({"a": 0, "b": 1, "c": 0}, 200.0),
        ({"a": 0, "b": 0, "c": 1}, 200.0),
        ({"a": 1, "b": 1, "c": 0}, 100.0),
        ({"a": 1, "b": 1, "c": 1}, 0.0),
    ])
    # Second problem: It's bad to assign more than one variable to 1.
    subproblem_b = gated_state_dag_test_lib.dag_from_paths([
        ({"a": 0, "b": 0, "c": 0}, 0.0),
        ({"a": 1, "b": 0, "c": 0}, 0.0),
        ({"a": 0, "b": 1, "c": 0}, 0.0),
        ({"a": 0, "b": 0, "c": 1}, 0.0),
        ({"a": 1, "b": 1, "c": 0}, 200.0),
        ({"a": 1, "b": 1, "c": 1}, 300.0),
    ])

    # Build a system.
    packed_a, conversion_data_a = packed_dags.convert_dag_to_packed(
        dag=subproblem_a,
        missing_assignment_value=-1,
        variable_value_ordering=list(range(2)),
    )
    packed_b, conversion_data_b = packed_dags.convert_dag_to_packed(
        dag=subproblem_b,
        missing_assignment_value=-1,
        variable_value_ordering=list(range(2)),
    )
    system = consistent_path_dual_solver.make_system([
        (packed_a, conversion_data_a),
        (packed_b, conversion_data_b),
    ])

    # Extract min marginals. It looks like assigning all of them to 1 is best.
    min_marginals = np.stack(
        [
            consistent_path_dual_solver.system_min_marginals(system.data, i)
            for i in range(3)
        ]
    )
    np.testing.assert_array_equal(
        min_marginals,
        np.array([
            [200.0, 0.0],  # a
            [200.0, 0.0],  # b
            [100.0, 0.0],  # c
        ]),
    )

    # Try to forbid 3 assignments. However, when we forbid a=1,
    # it changes the min marginals enough to not forbid anything for b, and
    # forbid a different choice for c.
    status = consistent_path_dual_solver.partially_commit(
        system.data,
        direction=consistent_path_dual_solver.SweepDirection.FORWARD,
        target_commitments_per_variable=1,
        gap_strictness=1.0,
    )
    self.assertEqual(
        status,
        consistent_path_dual_solver.PartialCommitmentStatus.MADE_PARTIAL_ASSIGNMENT,
    )
    np.testing.assert_array_equal(
        system.data.dag_tables[0].penalties,
        np.array([
            [np.inf, 0.0],  # a
            [0.0, 0.0],  # b
            [0.0, np.inf],  # c
        ]),
    )

  def test_weak_constraints(self):
    """Tests that we can apply constraints to our solutions."""
    subproblem_a = gated_state_dag_test_lib.dag_from_paths([
        ({"choice_0": 0, "choice_1": 0}, 0.0),
        ({"choice_0": 0, "choice_1": 1}, 1.0),
        ({"choice_0": 1, "choice_1": 0}, 101.0),
        ({"choice_0": 1, "choice_1": 1}, 100.0),
    ])
    # Make a system. (This system is pretty trivial, as we can just read off
    # the solution for the min marginals, but it's enough for our purposes.)
    subproblem_a = gated_state_dag.prune_to_reachable(subproblem_a)
    packed_a, conversion_data_a = packed_dags.convert_dag_to_packed(
        dag=subproblem_a,
        missing_assignment_value=-1,
        variable_value_ordering=[0, 1],
    )
    system = consistent_path_dual_solver.make_system(
        [(packed_a, conversion_data_a)]
    )
    # Constrain the system to weakly prefer {"choice_0": 1, "choice_1": 0}.
    # We use such a weak penalty that this doesn't change the optimal solution.
    consistent_path_dual_solver.constrain_system(
        system, {"choice_0": 1, "choice_1": 0}, penalty=0.01
    )
    # Solve. We expect the ordinary solution, but the cost should go up by 0.01,
    # since we violated one of the two constraints.
    sweep_results = consistent_path_dual_solver.solve_system_with_sweeps(
        system.data
    )
    with self.subTest("weak_penalty_dual_bound"):
      self.assertEqual(sweep_results.objective_at_step[-1], 0.01)
    (
        greedy_forward_assignments_vector,
        greedy_forward_cost,
    ) = consistent_path_dual_solver.greedy_extract(
        system.data,
        direction=consistent_path_dual_solver.SweepDirection.FORWARD,
    )
    greedy_forward_assignments = (
        consistent_path_dual_solver.assignments_from_assignment_vector(
            system, greedy_forward_assignments_vector
        )
    )
    with self.subTest("greedy_forward_cost"):
      self.assertEqual(greedy_forward_cost, 0.01)
    with self.subTest("greedy_forward_assignments"):
      self.assertEqual(
          greedy_forward_assignments, {"choice_0": 0, "choice_1": 0}
      )

  def test_strong_constraints(self):
    """Tests that we can apply constraints to our solutions."""
    subproblem_a = gated_state_dag_test_lib.dag_from_paths([
        ({"choice_0": 0, "choice_1": 0}, 0.0),
        ({"choice_0": 0, "choice_1": 1}, 1.0),
        ({"choice_0": 1, "choice_1": 0}, 101.0),
        ({"choice_0": 1, "choice_1": 1}, 100.0),
    ])
    # Make a system. (This system is pretty trivial, as we can just read off
    # the solution for the min marginals, but it's enough for our purposes.)
    subproblem_a = gated_state_dag.prune_to_reachable(subproblem_a)
    packed_a, conversion_data_a = packed_dags.convert_dag_to_packed(
        dag=subproblem_a,
        missing_assignment_value=-1,
        variable_value_ordering=[0, 1],
    )
    system = consistent_path_dual_solver.make_system(
        [(packed_a, conversion_data_a)]
    )
    # Constrain the system to strongly prefer {"choice_0": 1, "choice_1": 0}.
    consistent_path_dual_solver.constrain_system(
        system, {"choice_0": 1, "choice_1": 0}, penalty=100_000
    )
    # Solve. The choice we selected is now cheapest.
    sweep_results = consistent_path_dual_solver.solve_system_with_sweeps(
        system.data
    )
    with self.subTest("weak_penalty_dual_bound"):
      self.assertEqual(sweep_results.objective_at_step[-1], 101.0)
    (
        greedy_forward_assignments_vector,
        greedy_forward_cost,
    ) = consistent_path_dual_solver.greedy_extract(
        system.data,
        direction=consistent_path_dual_solver.SweepDirection.FORWARD,
    )
    greedy_forward_assignments = (
        consistent_path_dual_solver.assignments_from_assignment_vector(
            system, greedy_forward_assignments_vector
        )
    )
    with self.subTest("greedy_forward_cost"):
      self.assertEqual(greedy_forward_cost, 101.0)
    with self.subTest("greedy_forward_assignments"):
      self.assertEqual(
          greedy_forward_assignments, {"choice_0": 1, "choice_1": 0}
      )

  def test_break_symmetry_randomly(self):
    """Tests that we can break symmetry in random problems."""
    # We construct a pair of subproblems such that all min marginals are zero,
    # but if we choose the same value for choice_0 and choice_1, we get stuck
    # in a contradiction at choice_2.
    subproblem_a = gated_state_dag_test_lib.dag_from_paths([
        ({"choice_0": 0, "choice_1": 0, "choice_2": 0}, 0.0),
        ({"choice_0": 1, "choice_1": 1, "choice_2": 1}, 0.0),
        ({"choice_0": 0, "choice_1": 1, "choice_2": 2}, 0.0),
        ({"choice_0": 1, "choice_1": 0, "choice_2": 2}, 0.0),
    ])
    subproblem_b = gated_state_dag_test_lib.dag_from_paths([
        ({"choice_0": 0, "choice_1": 0, "choice_2": 1}, 0.0),
        ({"choice_0": 1, "choice_1": 1, "choice_2": 0}, 0.0),
        ({"choice_0": 0, "choice_1": 1, "choice_2": 2}, 0.0),
        ({"choice_0": 1, "choice_1": 0, "choice_2": 2}, 0.0),
    ])

    # Build a system.
    system_parts = []
    for subproblem in [subproblem_a, subproblem_b]:
      subproblem = gated_state_dag.prune_to_reachable(subproblem)
      packed, conversion_data = packed_dags.convert_dag_to_packed(
          dag=subproblem,
          missing_assignment_value=-1,
          variable_value_ordering=[0, 1, 2],
      )
      system_parts.append((packed, conversion_data))

    system = consistent_path_dual_solver.make_system(system_parts)

    # Greedy decopding gets stuck, even after sweeping min marginals.
    # (note: this fails because greedy decoding always chooses the same
    # assignment under ties)
    sweep_results = consistent_path_dual_solver.solve_system_with_sweeps(
        system.data
    )
    with self.subTest("symmetry_not_broken_contradiction"):
      # Sweep finds a cost of 0, which is actually optimal here.
      self.assertEqual(sweep_results.objective_at_step[-1], 0.0)
      # But greedy decoding can't extract this solution, and commits to the
      # same value for choice 0 and choice 1.
      (assignments, cost) = consistent_path_dual_solver.greedy_extract(
          system.data,
          direction=consistent_path_dual_solver.SweepDirection.FORWARD,
      )
      self.assertEqual(cost, np.inf)
      assignments_dict = (
          consistent_path_dual_solver.assignments_from_assignment_vector(
              system, assignments
          )
      )
      self.assertEqual(
          assignments_dict["choice_0"], assignments_dict["choice_1"]
      )

    # Break symmetry.
    consistent_path_dual_solver.break_symmetry_randomly(system, random_seed=42)

    # Solve again.
    sweep_results = consistent_path_dual_solver.solve_system_with_sweeps(
        system.data
    )
    with self.subTest("symmetry_broken"):
      self.assertLess(sweep_results.objective_at_step[-1], np.inf)
      (assignments, cost) = consistent_path_dual_solver.greedy_extract(
          system.data,
          direction=consistent_path_dual_solver.SweepDirection.FORWARD,
      )
      # Should be tight
      self.assertEqual(cost, sweep_results.objective_at_step[-1])
      assignments_dict = (
          consistent_path_dual_solver.assignments_from_assignment_vector(
              system, assignments
          )
      )
      # Should have avoided the contradiction by choosing different values for
      # the two choices.
      self.assertNotEqual(
          assignments_dict["choice_0"], assignments_dict["choice_1"]
      )


if __name__ == "__main__":
  absltest.main()
