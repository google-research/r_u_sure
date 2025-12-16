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

"""Tests for uncertain regions cost function wrappers."""
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from r_u_sure.decision_diagrams import consistent_path_dual_solver
from r_u_sure.testing import test_flags
from r_u_sure.wrappers import parser_tools
from r_u_sure.wrappers import uncertainty_regions_wrapper
from r_u_sure.wrappers import wrapper_test_util


class UncertainRegionsWrapperTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='with_numba', use_numba=True),
      dict(testcase_name='without_numba', use_numba=False),
  )
  def test_uncertainty_regions_wrapper(self, use_numba):
    if use_numba and test_flags.SKIP_JIT_TESTS.value:
      self.skipTest('Skipping JIT test variant')

    # Generate some examples.
    # All examples have different function names.
    # 6/10 have "if not xxxx and not yyyy" and 4/10 have
    #   "if xxxx is None and yyyy is None"
    # 8/10 have `print("aaaa")` and 2/10 have `print("bbbb")`
    # pylint: disable=g-complex-comprehension
    model_samples = (
        [
            textwrap.dedent(
                f"""\
                print("context")
                def my_function_{i}(xxxx, yyyy):
                  if not xxxx and not yyyy:
                    print("aaaa")
                """
            )
            for i in [0, 1, 2, 3]
        ]
        + [
            textwrap.dedent(
                f"""\
                print("context")
                def my_function_{i}(xxxx, yyyy):
                  if xxxx is None and yyyy is None:
                    print("aaaa")
                """
            )
            for i in [4, 5, 6, 7]
        ]
        + [
            textwrap.dedent(
                f"""\
                print("context")
                def my_function_{i}(xxxx, yyyy):
                  if not xxxx and not yyyy:
                    print("bbbb")
                """
            )
            for i in [8, 9]
        ]
    )
    # pylint: enable=g-complex-comprehension

    # Use the first sample as the prototype.
    prototype_string = model_samples[0]

    # Ground truth happens to have the "is None" variant and also "bbbb".
    # (This could also be chosen as a leave-one-out sample from the model
    # samples.)
    ground_truth = textwrap.dedent(
        """\
        print("context")
        def my_function_ground_truth(xxxx, yyyy):
          if xxxx is None and yyyy is None:
            print("bbbb")
        """
    )

    # Pretend cursor is after `def`
    cursor_position = ground_truth.find('def') + len('def')

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
    # We use an effective precision of 0.7, which means that (in most cases)
    # edits that happen more than 30% of the time will be flagged with
    # uncertainty regions. In this case, we expect the function name and if
    # condition to be put in uncertainty regions, but the print statement
    # NOT to be put in an uncertainty region.
    wrapper = uncertainty_regions_wrapper.UncertaintyRegionsWrapper(
        effective_precision=0.7,
        high_confidence_start_inserting_cost=5.0,
        low_confidence_edit_sensitivity=0.1,
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
    with self.subTest(name='system_bound'):
      self.assertAlmostEqual(-34.32, cost_bound_from_system, places=6)
    with self.subTest(name='solution_cost'):
      self.assertAlmostEqual(-34.32, cost_from_samples, places=6)

    # We should have produced the following suggestion:
    parts_grouped_by_confidence = wrapper_test_util.group_parts_by_confidence(
        solution_info['extracted_parts']
    )
    with self.subTest(name='suggestion_from_samples'):
      self.assertEqual(
          [
              (' ', 'high'),
              ('my_function_0', 'low'),  # low-confidence region for changing
              ('(xxxx, yyyy):\n  if ', 'high'),
              ('not ', 'low'),  # low-confidence region for deleting this
              ('xxxx ', 'high'),
              ('', 'low'),  # empty low-conf region for inserting `is None`
              ('and ', 'high'),
              ('not ', 'low'),  # low-confidence region for deleting this
              ('yyyy', 'high'),
              ('', 'low'),  # empty low-conf region for inserting `is None`
              (':\n    print("aaaa")\n', 'high'),
          ],
          parts_grouped_by_confidence,
      )

    # ... along with many relevant summary metrics:
    metrics = dict(solution_info)
    del metrics['extracted_parts']
    with self.subTest(name='metrics'):
      self.assertEqual(
          {
              'DELETE_HIGH_CONFIDENCE_chars': 4,
              'DELETE_HIGH_CONFIDENCE_cost': 4.0,
              'DELETE_HIGH_CONFIDENCE_edges': 1,
              'DELETE_LOW_CONFIDENCE_chars': 19,
              'DELETE_LOW_CONFIDENCE_cost': -4.94,
              'DELETE_LOW_CONFIDENCE_edges': 3,
              'INSERT_NOT_APPLICABLE_chars': 40,
              'INSERT_NOT_APPLICABLE_cost': 0.0,
              'INSERT_NOT_APPLICABLE_edges': 6,
              'KEEP_HIGH_CONFIDENCE_chars': 35,
              'KEEP_HIGH_CONFIDENCE_cost': -35.0,
              'KEEP_HIGH_CONFIDENCE_edges': 20,
              'PROTOTYPE_DECORATION_HIGH_CONFIDENCE_chars': 14,
              'PROTOTYPE_DECORATION_HIGH_CONFIDENCE_cost': 0.0,
              'PROTOTYPE_DECORATION_HIGH_CONFIDENCE_edges': 10,
              'PROTOTYPE_DECORATION_LOW_CONFIDENCE_chars': 2,
              'PROTOTYPE_DECORATION_LOW_CONFIDENCE_cost': 0.0,
              'PROTOTYPE_DECORATION_LOW_CONFIDENCE_edges': 2,
              'START_EDITING_HIGH_CONFIDENCE_chars': 0,
              'START_EDITING_HIGH_CONFIDENCE_cost': 5.0,
              'START_EDITING_HIGH_CONFIDENCE_edges': 1,
              'START_EDITING_LOW_CONFIDENCE_chars': 0,
              'START_EDITING_LOW_CONFIDENCE_cost': 2.5,
              'START_EDITING_LOW_CONFIDENCE_edges': 5,
              'START_LOW_CONFIDENCE_NOT_APPLICABLE_chars': 0,
              'START_LOW_CONFIDENCE_NOT_APPLICABLE_cost': 6.75,
              'START_LOW_CONFIDENCE_NOT_APPLICABLE_edges': 5,
              'TARGET_DECORATION_HIGH_CONFIDENCE_chars': 18,
              'TARGET_DECORATION_HIGH_CONFIDENCE_cost': 0.0,
              'TARGET_DECORATION_HIGH_CONFIDENCE_edges': 14,
              'total_cost': -21.69,
          },
          {k: round(v, 6) for k, v in metrics.items()},
      )

  def test_uncertainty_regions_baselines(self):
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

    wrapper = uncertainty_regions_wrapper.UncertaintyRegionsWrapper(
        use_numba=False,
        baseline_token_prob_thresholds=(0.3, 0.5, 0.7, 0.9),
        baseline_example_fractions=(0.0, 0.5, 1.0),
        baseline_depth_cutoffs=((1, 1), (1, 3), (1, 4), (2, 1)),
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
            'example_fraction_0.0': [
                (' ', 'high'),
                ('my_function(xxxx, yyyy):\n', 'low'),
                ('', 'high'),
                (
                    (
                        '  print(prob_80)\n  prob_70()\n  prob_60()\n '
                        ' prob_40()\n  print(prob_90)\n  return prob_10\n'
                    ),
                    'low',
                ),
                ('\n', 'high'),
                ('print(prob_20)\n', 'low'),
            ],
            'example_fraction_0.5': [
                (
                    (
                        ' my_function(xxxx, yyyy):\n  print(prob_80)\n '
                        ' prob_70()\n  prob_60'
                    ),
                    'high',
                ),
                (
                    '()\n  prob_40()\n  print(prob_90)\n  return prob_10\n',
                    'low',
                ),
                ('\n', 'high'),
                ('print(prob_20)\n', 'low'),
            ],
            'example_fraction_1.0': [(
                (
                    ' my_function(xxxx, yyyy):\n  print(prob_80)\n  prob_70()\n'
                    '  prob_60()\n  prob_40()\n  print(prob_90)\n  return'
                    ' prob_10\n\nprint(prob_20)\n'
                ),
                'high',
            )],
            'prob_threshold_0.3_cumulative': [
                (
                    (
                        ' my_function(xxxx, yyyy):\n  print(prob_80)\n '
                        ' prob_70()\n  prob_60()\n'
                    ),
                    'high',
                ),
                ('  prob_40()\n  print(prob_90)\n  return prob_10\n', 'low'),
                ('\n', 'high'),
                ('print(prob_20)\n', 'low'),
            ],
            'prob_threshold_0.3_nonempty': [
                (
                    (
                        ' my_function(xxxx, yyyy):\n  print(prob_80)\n '
                        ' prob_70()\n  prob_60()\n  prob_40()\n '
                        ' print(prob_90)\n  return '
                    ),
                    'high',
                ),
                ('prob_10\n', 'low'),
                ('\nprint(', 'high'),
                ('prob_20', 'low'),
                (')\n', 'high'),
            ],
            'prob_threshold_0.5_cumulative': [
                (
                    (
                        ' my_function(xxxx, yyyy):\n  print(prob_80)\n '
                        ' prob_70()\n'
                    ),
                    'high',
                ),
                (
                    (
                        '  prob_60()\n  prob_40()\n  print(prob_90)\n  return'
                        ' prob_10\n'
                    ),
                    'low',
                ),
                ('\n', 'high'),
                ('print(prob_20)\n', 'low'),
            ],
            'prob_threshold_0.5_nonempty': [
                (
                    (
                        ' my_function(xxxx, yyyy):\n  print(prob_80)\n '
                        ' prob_70()\n  prob_60()\n  '
                    ),
                    'high',
                ),
                ('prob_40', 'low'),
                ('()\n  print(prob_90)\n  return ', 'high'),
                ('prob_10\n', 'low'),
                ('\nprint(', 'high'),
                ('prob_20', 'low'),
                (')\n', 'high'),
            ],
            'prob_threshold_0.7_cumulative': [
                (' my_function(xxxx, yyyy):\n  print(prob_80)\n', 'high'),
                (
                    (
                        '  prob_70()\n  prob_60()\n  prob_40()\n '
                        ' print(prob_90)\n  return prob_10\n'
                    ),
                    'low',
                ),
                ('\n', 'high'),
                ('print(prob_20)\n', 'low'),
            ],
            'prob_threshold_0.7_nonempty': [
                (
                    (
                        ' my_function(xxxx, yyyy):\n  print(prob_80)\n '
                        ' prob_70()\n  '
                    ),
                    'high',
                ),
                ('prob_60', 'low'),
                ('()\n  ', 'high'),
                ('prob_40', 'low'),
                ('()\n  print(prob_90)\n  return ', 'high'),
                ('prob_10\n', 'low'),
                ('\nprint(', 'high'),
                ('prob_20', 'low'),
                (')\n', 'high'),
            ],
            'prob_threshold_0.9_cumulative': [
                (' my_function(xxxx, yyyy):\n  print', 'high'),
                (
                    (
                        '(prob_80)\n  prob_70()\n  prob_60()\n  prob_40()\n '
                        ' print(prob_90)\n  return prob_10\n'
                    ),
                    'low',
                ),
                ('\n', 'high'),
                ('print(prob_20)\n', 'low'),
            ],
            'prob_threshold_0.9_nonempty': [
                (' my_function(xxxx, yyyy):\n  print(', 'high'),
                ('prob_80', 'low'),
                (')\n  ', 'high'),
                ('prob_70', 'low'),
                ('()\n  ', 'high'),
                ('prob_60', 'low'),
                ('()\n  ', 'high'),
                ('prob_40', 'low'),
                ('()\n  print(prob_90)\n  return ', 'high'),
                ('prob_10\n', 'low'),
                ('\nprint(', 'high'),
                ('prob_20', 'low'),
                (')\n', 'high'),
            ],
            'cutoff_anc_1_child_1': [
                (' my_function', 'high'),
                ('(xxxx, yyyy)', 'low'),
                (':\n', 'high'),
                (
                    (
                        '  print(prob_80)\n  prob_70()\n  prob_60()\n '
                        ' prob_40()\n  print(prob_90)\n  return'
                        ' prob_10\n\nprint(prob_20)\n'
                    ),
                    'low',
                ),
            ],
            'cutoff_anc_1_child_3': [
                (' my_function(xxxx, yyyy):\n', 'high'),
                (
                    (
                        '  print(prob_80)\n  prob_70()\n  prob_60()\n '
                        ' prob_40()\n  print(prob_90)\n  return prob_10\n'
                    ),
                    'low',
                ),
                ('\n', 'high'),
                ('print', 'low'),
                ('(prob_20)\n', 'high'),
            ],
            'cutoff_anc_1_child_4': [
                (' my_function(xxxx, yyyy):\n  print', 'high'),
                ('(prob_80)\n', 'low'),
                ('  prob_70', 'high'),
                ('()\n', 'low'),
                ('  prob_60', 'high'),
                ('()\n', 'low'),
                ('  prob_40', 'high'),
                ('()\n', 'low'),
                ('  print', 'high'),
                ('(prob_90)\n', 'low'),
                ('  return prob_10\n\n', 'high'),
                ('print', 'low'),
                ('(prob_20)\n', 'high'),
            ],
            'cutoff_anc_2_child_1': [
                (' my_function', 'high'),
                ('(xxxx, yyyy)', 'low'),
                (':\n', 'high'),
                (
                    (
                        '  print(prob_80)\n  prob_70()\n  prob_60()\n '
                        ' prob_40()\n  print(prob_90)\n  return prob_10\n\n'
                    ),
                    'low',
                ),
                ('print(', 'high'),
                ('prob_20', 'low'),
                (')\n', 'high'),
            ],
        },
        baseline_part_groups,
    )


if __name__ == '__main__':
  absltest.main()
