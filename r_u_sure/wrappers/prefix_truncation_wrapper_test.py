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

"""Tests for prefix cost function wrappers."""
import itertools
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from r_u_sure.decision_diagrams import consistent_path_dual_solver
from r_u_sure.testing import test_flags
from r_u_sure.wrappers import parser_tools
from r_u_sure.wrappers import prefix_truncation_wrapper
from r_u_sure.wrappers import wrapper_test_util


class PrefixTruncationWrapperTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='with_numba', use_numba=True),
      dict(testcase_name='without_numba', use_numba=False),
      dict(
          testcase_name='including_uncertainty_with_numba',
          use_numba=True,
          also_insert_uncertainty_regions=True,
      ),
      dict(
          testcase_name='including_uncertainty_without_numba',
          use_numba=False,
          also_insert_uncertainty_regions=True,
      ),
  )
  def test_prefix_by_edit_distance_wrapper(
      self,
      use_numba,
      also_insert_uncertainty_regions=False,
  ):
    if use_numba and test_flags.SKIP_JIT_TESTS.value:
      self.skipTest('Skipping JIT test variant')

    # Generate some examples which differ in the middle but have parts that
    # align
    model_samples = [
        textwrap.dedent(
            """\
            def foo():
              shared_1()
              different_XXXX()
              shared_2()
              different_AAAA()

            def different_1234():
              pass
            """
        ),
        textwrap.dedent(
            """\
            def foo():
              shared_1()
              different_YYYY(1234)
              shared_2()
              if different_BBBB:
                pass

            print("!")
            """
        ),
        textwrap.dedent(
            """\
            def foo():
              shared_1()
              different_ZZZZ("a", "b", "c")
              shared_2()
              different_CCCC(2)

            # other stuff
            """
        ),
        textwrap.dedent(
            """\
            def foo():
              shared_1()
              different_WWWW(42)
              shared_2()
              return different_DDDD

            # other stuff
            """
        ),
    ]
    # pylint: enable=g-complex-comprehension

    # Use the first sample as the prototype.
    prototype_string = model_samples[0]

    # Ground truth is slightly different as well.
    ground_truth = textwrap.dedent(
        """\
        def foo():
          shared_1()
          different_ABCD(5, 4)
          shared_2()
          return 42

        # some other functions
        """
    )
    # Pretend cursor is after `foo()`
    cursor_position = ground_truth.find('foo()') + len('foo()')

    # Construct a parser helper.
    parser_helper = parser_tools.ParserHelper(language='python')

    # Parse all of our strings.
    prototype_as_nodes = parser_helper.parse_to_nodes(prototype_string)
    model_samples_as_nodes = [
        parser_helper.parse_to_nodes(hypothetical_target_string)
        for hypothetical_target_string in model_samples
    ]
    ground_truth_as_nodes = parser_helper.parse_to_nodes(ground_truth)

    # Construct our wrapper object.
    wrapper = prefix_truncation_wrapper.PrefixByEditDistanceWrapper(
        also_insert_uncertainty_regions=also_insert_uncertainty_regions,
        start_editing_cost=0.0,
        uncertainty_region_effective_precision=0.7,
        low_confidence_edit_sensitivity=0.5,
        use_numba=use_numba,
    )

    # Process prototype using wrapper (this will insert region nodes as
    # needed)
    packed_prototype, context_info = wrapper.process_prototype_and_context(
        prototype_as_nodes, cursor_position
    )
    # Process targets
    packed_samples = [
        wrapper.process_target(model_sample_as_nodes, cursor_position)
        for model_sample_as_nodes in model_samples_as_nodes
    ]
    packed_ground_truth = wrapper.process_target(
        ground_truth_as_nodes, cursor_position
    )

    # Build the sample-based system for the optimization problem: we will find
    # the best suggestion for our model samples.
    sample_system = wrapper.build_system(
        prototype=packed_prototype,
        context_info=context_info,
        targets=packed_samples,
    )

    # Solve the system.
    opt_results = consistent_path_dual_solver.solve_system_with_sweeps(
        system_data=sample_system.data, soft_timeout=10.0
    )
    cost_bound_from_system = opt_results.objective_at_step[-1]

    # Extract a solution.
    greedy_assignment_vector, cost_from_samples = (
        consistent_path_dual_solver.greedy_extract(
            sample_system.data,
            direction=consistent_path_dual_solver.SweepDirection.FORWARD,
        )
    )
    greedy_assignments = (
        consistent_path_dual_solver.assignments_from_assignment_vector(
            sample_system, greedy_assignment_vector
        )
    )

    # Build an evaluation system with just the ground truth.
    eval_system = wrapper.build_system(
        prototype=packed_prototype,
        context_info=context_info,
        targets=[packed_ground_truth],
    )

    # Evaluate our solution.
    solution_info = wrapper.solution_info(
        prototype=packed_prototype,
        evaluation_target=packed_ground_truth,
        context_info=context_info,
        system=eval_system,
        assignments=greedy_assignments,
    )

    # We should have found the optimal solution for these samples, with a tight
    # optimization bound.
    if also_insert_uncertainty_regions:
      expected_cost = -22.3
    else:
      expected_cost = -16.0
    with self.subTest(name='system_bound'):
      self.assertAlmostEqual(expected_cost, cost_bound_from_system, places=6)
    with self.subTest(name='solution_cost'):
      self.assertAlmostEqual(expected_cost, cost_from_samples, places=6)

    # We should have produced the following suggestion:
    parts_grouped_by_confidence = []
    for is_low_confidence, parts in itertools.groupby(
        solution_info['extracted_parts'], lambda v: v[1]
    ):
      combined_parts = ''.join(part for (part, _) in parts)
      parts_grouped_by_confidence.append(
          (combined_parts, 'low' if is_low_confidence else 'high')
      )

    with self.subTest(name='suggestion_from_samples'):
      if also_insert_uncertainty_regions:
        self.assertEqual(
            [
                (':\n  shared_1()\n  ', 'high'),
                ('different_XXXX', 'low'),
                ('()\n  shared_2()\n', 'high'),
            ],
            parts_grouped_by_confidence,
        )
      else:
        self.assertEqual(
            [
                (':\n  shared_1()\n  different_XXXX()\n  shared_2()\n', 'high'),
            ],
            parts_grouped_by_confidence,
        )

    # ... along with many relevant summary metrics:
    metrics = dict(solution_info)
    del metrics['extracted_parts']
    rounded_metrics = {k: round(v, 6) for k, v in metrics.items()}
    with self.subTest(name='metrics'):
      if also_insert_uncertainty_regions:
        self.assertEqual(
            {
                'DELETE_HIGH_CONFIDENCE_chars': 0,
                'DELETE_HIGH_CONFIDENCE_cost': 0.0,
                'DELETE_HIGH_CONFIDENCE_edges': 1,
                'DELETE_LOW_CONFIDENCE_chars': 14,
                'DELETE_LOW_CONFIDENCE_cost': 4.2,
                'DELETE_LOW_CONFIDENCE_edges': 1,
                'EARLY_EXIT_HIGH_CONFIDENCE_chars': 0,
                'EARLY_EXIT_HIGH_CONFIDENCE_cost': 0.0,
                'EARLY_EXIT_HIGH_CONFIDENCE_edges': 1,
                'INSERT_NOT_APPLICABLE_chars': 17,
                'INSERT_NOT_APPLICABLE_cost': 0.0,
                'INSERT_NOT_APPLICABLE_edges': 4,
                'KEEP_HIGH_CONFIDENCE_chars': 23,
                'KEEP_HIGH_CONFIDENCE_cost': -23.0,
                'KEEP_HIGH_CONFIDENCE_edges': 12,
                'PROTOTYPE_DECORATION_HIGH_CONFIDENCE_chars': 10,
                'PROTOTYPE_DECORATION_HIGH_CONFIDENCE_cost': 0,
                'PROTOTYPE_DECORATION_HIGH_CONFIDENCE_edges': 7,
                'START_EDITING_HIGH_CONFIDENCE_chars': 0,
                'START_EDITING_HIGH_CONFIDENCE_cost': 0.0,
                'START_EDITING_HIGH_CONFIDENCE_edges': 34,
                'START_LOW_CONFIDENCE_NOT_APPLICABLE_chars': 0,
                'START_LOW_CONFIDENCE_NOT_APPLICABLE_cost': 0.0,
                'START_LOW_CONFIDENCE_NOT_APPLICABLE_edges': 1,
                'TARGET_DECORATION_HIGH_CONFIDENCE_chars': 11,
                'TARGET_DECORATION_HIGH_CONFIDENCE_cost': 0,
                'TARGET_DECORATION_HIGH_CONFIDENCE_edges': 8,
                'total_cost': -18.8,
            },
            rounded_metrics,
        )
      else:
        self.assertEqual(
            {
                'DELETE_HIGH_CONFIDENCE_chars': 14,
                'DELETE_HIGH_CONFIDENCE_cost': 14.0,
                'DELETE_HIGH_CONFIDENCE_edges': 2,
                'EARLY_EXIT_HIGH_CONFIDENCE_chars': 0,
                'EARLY_EXIT_HIGH_CONFIDENCE_cost': 0.0,
                'EARLY_EXIT_HIGH_CONFIDENCE_edges': 1,
                'INSERT_NOT_APPLICABLE_chars': 17,
                'INSERT_NOT_APPLICABLE_cost': 0.0,
                'INSERT_NOT_APPLICABLE_edges': 4,
                'KEEP_HIGH_CONFIDENCE_chars': 23,
                'KEEP_HIGH_CONFIDENCE_cost': -23.0,
                'KEEP_HIGH_CONFIDENCE_edges': 12,
                'PROTOTYPE_DECORATION_HIGH_CONFIDENCE_chars': 10,
                'PROTOTYPE_DECORATION_HIGH_CONFIDENCE_cost': 0,
                'PROTOTYPE_DECORATION_HIGH_CONFIDENCE_edges': 7,
                'START_EDITING_HIGH_CONFIDENCE_chars': 0,
                'START_EDITING_HIGH_CONFIDENCE_cost': 0.0,
                'START_EDITING_HIGH_CONFIDENCE_edges': 34,
                'TARGET_DECORATION_HIGH_CONFIDENCE_chars': 11,
                'TARGET_DECORATION_HIGH_CONFIDENCE_cost': 0,
                'TARGET_DECORATION_HIGH_CONFIDENCE_edges': 8,
                'total_cost': -9.0,
            },
            rounded_metrics,
        )

  def test_prefix_length_baselines(self):
    prototype_string = textwrap.dedent(
        """\
        print("context")
        def my_function(xxxx, yyyy):
          print(prob_80)
          prob_70()
          prob_60()
          prob_40()
          print(prob_90)
          return prob_10

        print(prob_20)
        """
    )
    # Pretend cursor is after `def`
    cursor_position = prototype_string.find('def') + len('def')

    fake_tokens_with_log_probs = wrapper_test_util.split_with_fake_log_probs(
        prototype_string[cursor_position:]
    )

    # Construct a parser helper.
    parser_helper = parser_tools.ParserHelper(language='python')

    # Parse all of our strings.
    prototype_as_nodes = parser_helper.parse_to_nodes(prototype_string)

    wrapper = prefix_truncation_wrapper.PrefixByEditDistanceWrapper(
        also_insert_uncertainty_regions=False,
        use_numba=False,
        baseline_token_prob_thresholds=(0.0, 0.3, 0.5, 0.7, 0.9),
        baseline_intellicode=True,
        baseline_max_characters=(20, 50, 100, 200, 500),
        baseline_max_lines=(1, 2, 4, 8, 16),
    )

    packed_prototype, context_info = wrapper.process_prototype_and_context(
        prototype_as_nodes, cursor_position
    )
    packed_prototype_as_target = wrapper.process_target(
        prototype_as_nodes, cursor_position
    )

    baseline_assignments = wrapper.build_baseline_assignments(
        prototype=packed_prototype,
        prototype_suggestion_as_target=packed_prototype_as_target,
        context_info=context_info,
        model_tokens_and_log_probs=fake_tokens_with_log_probs,
    )

    baseline_part_groups = {}
    for key, assignments in baseline_assignments.items():
      solution_info = wrapper.solution_info(
          prototype=packed_prototype,
          evaluation_target=None,
          context_info=context_info,
          system=None,
          assignments=assignments,
      )
      baseline_part_groups[key] = wrapper_test_util.group_parts_by_confidence(
          solution_info['extracted_parts']
      )

    self.assertEqual(
        {
            'character_length_20': [(' my_function(xxxx, yyyy', 'high')],
            'character_length_50': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60'
                ),
                'high',
            )],
            'character_length_100': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60()\n  prob_40()\n  print(prob_90)\n  return'
                    ' prob_10\n\nprint(prob_20'
                ),
                'high',
            )],
            'character_length_200': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60()\n  prob_40()\n  print(prob_90)\n  return'
                    ' prob_10\n\nprint(prob_20)\n'
                ),
                'high',
            )],
            'character_length_500': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60()\n  prob_40()\n  print(prob_90)\n  return'
                    ' prob_10\n\nprint(prob_20)\n'
                ),
                'high',
            )],
            'intellicode': [(
                ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n',
                'high',
            )],
            'line_length_1': [(' my_function(xxxx, yyyy):\n', 'high')],
            'line_length_2': [(
                ' my_function(xxxx, yyyy):\n  print(prob_80)\n',
                'high',
            )],
            'line_length_4': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60()\n'
                ),
                'high',
            )],
            'line_length_8': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60()\n  prob_40()\n  print(prob_90)\n  return'
                    ' prob_10\n\n'
                ),
                'high',
            )],
            'line_length_16': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60()\n  prob_40()\n  print(prob_90)\n  return'
                    ' prob_10\n\nprint(prob_20)\n'
                ),
                'high',
            )],
            'prob_threshold_0.0': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60()\n  prob_40()\n  print(prob_90)\n  return'
                    ' prob_10\n\nprint(prob_20)\n'
                ),
                'high',
            )],
            'prob_threshold_0.3': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60()\n  prob_40()\n  print(prob_90)\n  return'
                ),
                'high',
            )],
            'prob_threshold_0.5': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60()\n'
                ),
                'high',
            )],
            'prob_threshold_0.7': [(
                ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n',
                'high',
            )],
            'prob_threshold_0.9': [(
                ' my_function(xxxx, yyyy):\n  print(',
                'high',
            )],
        },
        baseline_part_groups,
    )


if __name__ == '__main__':
  absltest.main()
