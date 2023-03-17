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

"""Tests for high-level cost function wrappers."""
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from r_u_sure.decision_diagrams import consistent_path_dual_solver
from r_u_sure.testing import test_flags
from r_u_sure.tree_structure import packed_sequence_nodes
from r_u_sure.tree_structure import sequence_node_helpers
from r_u_sure.wrappers import call_sequence_wrapper
from r_u_sure.wrappers import parser_tools
from r_u_sure.wrappers import wrapper_test_util


class CallSequenceWrapperTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='with_numba', use_numba=True),
      dict(testcase_name='without_numba', use_numba=False),
  )
  def test_call_sequence_wrapper(self, use_numba):
    if use_numba and test_flags.SKIP_JIT_TESTS.value:
      self.skipTest('Skipping JIT test variant')

    # Create some examples.
    model_samples = [
        textwrap.dedent(
            """\
            already_seen_tokens = [
                first_shared_call,
                shared_namespace,
                unshared_call_2,
            ]
            # completion starts here

            out_of_order_call()
            if True:
              first_shared_call()
              x = shared_namespace.second_shared_call(unshared_arg)
              unshared_call_1a(10)
              distinct_namespace_1.third_shared_call(shared_arg=10)
              shared_namespace.unshared_call_2(1234)

            distinct_namespace_1.fourth_shared_call("a", "b")
            foo += fifth_shared_call()

            class ClassNameNotACall():
              def not_actually_a_call():
                shared_result = shared_namespace.sixth_shared_call()
            """
        ),
        textwrap.dedent(
            """\
            already_seen_tokens = [
                first_shared_call,
                shared_namespace,
                unshared_call_2,
            ]
            # completion starts here

            for i in range(10):
              first_shared_call()
              # this example misses the second call
              try:
                distinct_namespace_2.third_shared_call(shared_arg=10)
              except BaseException:
                pass

              distinct_namespace_2.fourth_shared_call("a", "b")

            # this example misses the fifth call
            # this example doesn't assign the sixth call result the same way
            distinct_result_2 = shared_namespace.sixth_shared_call()
            out_of_order_call()

            class ClassNameNotACall():
              def not_actually_a_call():
                pass
            """
        ),
        textwrap.dedent(
            """\
            already_seen_tokens = [
                first_shared_call,
                shared_namespace,
                unshared_call_2,
            ]
            # completion starts here

            # this example misses the first call
            unshared_call_1c(shared_namespace.second_shared_call(10))

            while False:
              distinct_namespace_3.third_shared_call(shared_arg=10)

            # this example misses the fourth call
            bar = fifth_shared_call("x")
            shared_result = shared_namespace.sixth_shared_call()

            class ClassNameNotACall():
              def not_actually_a_call():
                pass
            """
        ),
        textwrap.dedent(
            """\
            already_seen_tokens = [
                first_shared_call,
                shared_namespace,
                unshared_call_2,
            ]
            # completion starts here

            if x:
              first_shared_call()
            else:
              unshared_call_1d(56)

            # this example misses calls 2-5
            shared_result = shared_namespace.sixth_shared_call()

            print(shared_result)

            class ClassNameNotACall():
              def not_actually_a_call():
                pass
            """
        ),
    ]

    # Use the first sample as the prototype.
    prototype_string = model_samples[0]

    # Ground truth
    ground_truth = textwrap.dedent(
        """\
        already_seen_tokens = [
            first_shared_call,
            shared_namespace,
            unshared_call_2,
        ]
        # completion starts here

        if foo:
          if bar:
            first_shared_call()
            # actually missing second call
            distinct_namespace_gt.third_shared_call(shared_arg=10)

        var = distinct_namespace_1.fourth_shared_call("a", "b")
        fifth_shared_call(var)
        # actually missing sixth call
        """
    )

    # Pretend cursor is after `completion starts here`
    start_string = 'completion starts here'
    cursor_position = ground_truth.find(start_string) + len(start_string)

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
    # Any calls that appear in two or more samples will be kept.
    wrapper = call_sequence_wrapper.ApiCallSequenceWrapper(
        effective_precision=0.3,
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

    # The prototype object will have been modified to extract only the calls.
    prototype_debug_render = sequence_node_helpers.render_debug(
        packed_sequence_nodes.unpack(packed_prototype)
    )
    expected_debug_render = textwrap.dedent(
        """\
        GROUP(API_CALL): 'out_of_order_call()'
          RegionStartNode()
          TOK(NOVEL): 'out_of_order_call'
          TOK(START_CALL): '('
          RegionEndNode()
          TOK(EMPTY_ARGS): ''
          TOK(END_CALL): ')'
          RegionEndNode()
        DEC: '\\n'
        GROUP(API_CALL): 'first_shared_call()'
          RegionStartNode()
          TOK(EXPECTED): 'first_shared_call'
          TOK(START_CALL): '('
          RegionEndNode()
          TOK(EMPTY_ARGS): ''
          TOK(END_CALL): ')'
          RegionEndNode()
        DEC: '\\n'
        GROUP(API_CALL): 'x = shared_namespace.second_shared_call(unshared_arg)'
          RegionStartNode()
          TOK(NOVEL): 'x'
          DEC: ' '
          RegionStartNode()
          TOK(EXPECTED): '='
          DEC: ' '
          RegionStartNode()
          TOK(EXPECTED): 'shared_namespace'
          RegionStartNode()
          TOK(ATTRIBUTE_DOT): '.'
          TOK(NOVEL): 'second_shared_call'
          TOK(START_CALL): '('
          RegionEndNode()
          TOK(ARGS): 'unshared_arg'
          TOK(END_CALL): ')'
          RegionEndNode()
        DEC: '\\n'
        GROUP(API_CALL): 'unshared_call_1a(10)'
          RegionStartNode()
          TOK(NOVEL): 'unshared_call_1a'
          TOK(START_CALL): '('
          RegionEndNode()
          TOK(ARGS): '10'
          TOK(END_CALL): ')'
          RegionEndNode()
        DEC: '\\n'
        GROUP(API_CALL): 'distinct_namespace_1.third_shared_call(shared_arg=10)'
          RegionStartNode()
          TOK(NOVEL): 'distinct_namespace_1'
          RegionStartNode()
          TOK(ATTRIBUTE_DOT): '.'
          TOK(NOVEL): 'third_shared_call'
          TOK(START_CALL): '('
          RegionEndNode()
          TOK(ARGS): 'shared_arg=10'
          TOK(END_CALL): ')'
          RegionEndNode()
        DEC: '\\n'
        GROUP(API_CALL): 'shared_namespace.unshared_call_2(1234)'
          RegionStartNode()
          TOK(EXPECTED): 'shared_namespace'
          RegionStartNode()
          TOK(ATTRIBUTE_DOT): '.'
          TOK(EXPECTED): 'unshared_call_2'
          TOK(START_CALL): '('
          RegionEndNode()
          TOK(ARGS): '1234'
          TOK(END_CALL): ')'
          RegionEndNode()
        DEC: '\\n'
        GROUP(API_CALL): 'distinct_namespace_1.fourth_shared_call("a", "b")'
          RegionStartNode()
          TOK(NOVEL): 'distinct_namespace_1'
          RegionStartNode()
          TOK(ATTRIBUTE_DOT): '.'
          TOK(NOVEL): 'fourth_shared_call'
          TOK(START_CALL): '('
          RegionEndNode()
          TOK(ARGS): '"a", "b"'
          TOK(END_CALL): ')'
          RegionEndNode()
        DEC: '\\n'
        GROUP(API_CALL): 'foo += fifth_shared_call()'
          RegionStartNode()
          TOK(NOVEL): 'foo'
          DEC: ' '
          RegionStartNode()
          TOK(EXPECTED): '+'
          TOK(EXPECTED): '='
          DEC: ' '
          RegionStartNode()
          TOK(NOVEL): 'fifth_shared_call'
          TOK(START_CALL): '('
          RegionEndNode()
          TOK(EMPTY_ARGS): ''
          TOK(END_CALL): ')'
          RegionEndNode()
        DEC: '\\n'
        GROUP(API_CALL): 'shared_result = shared_namespace.sixth_shared_call()'
          RegionStartNode()
          TOK(NOVEL): 'shared_result'
          DEC: ' '
          RegionStartNode()
          TOK(EXPECTED): '='
          DEC: ' '
          RegionStartNode()
          TOK(EXPECTED): 'shared_namespace'
          RegionStartNode()
          TOK(ATTRIBUTE_DOT): '.'
          TOK(NOVEL): 'sixth_shared_call'
          TOK(START_CALL): '('
          RegionEndNode()
          TOK(EMPTY_ARGS): ''
          TOK(END_CALL): ')'
          RegionEndNode()
        DEC: '\\n'"""
    )
    with self.subTest(name='extracted_prototype_calls'):
      self.assertEqual(expected_debug_render, prototype_debug_render)

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

    # We should have found the optimal solution for these samples.
    with self.subTest(name='system_bound'):
      self.assertAlmostEqual(-20.4, cost_bound_from_system, places=6)
    with self.subTest(name='solution_cost'):
      self.assertAlmostEqual(-20.4, cost_from_samples, places=6)

    # We should have produced the following suggestion:
    parts_grouped_by_confidence = wrapper_test_util.group_parts_by_confidence(
        solution_info['extracted_parts'],
        invert_confidence=True,
    )
    with self.subTest(name='suggestion_from_samples'):
      self.assertEqual(
          [
              ('out_of_order_call()\n', 'low'),
              ('first_shared_call()', 'high'),
              (
                  (
                      '\n'
                      'x = shared_namespace.second_shared_call(unshared_arg)\n'
                      'unshared_call_1a(10)\n'
                      'distinct_namespace_1'
                  ),
                  'low',
              ),
              ('.third_shared_call(shared_arg=10)', 'high'),
              (
                  '\nshared_namespace.unshared_call_2(1234)\ndistinct_namespace_1',
                  'low',
              ),
              ('.fourth_shared_call("a", "b")', 'high'),
              ('\nfoo += ', 'low'),
              ('fifth_shared_call(', 'high'),
              (')\n', 'low'),
              ('shared_result = shared_namespace.sixth_shared_call()', 'high'),
              ('\n', 'low'),
          ],
          parts_grouped_by_confidence,
      )

    # ... along with many relevant summary metrics:
    metrics = dict(solution_info)
    del metrics['extracted_parts']
    with self.subTest(name='metrics'):
      # Note: low confidence and high confidence are backwards below, because
      # the API call wrapper uses "low confidence" logic to extract the
      # selected expressions.
      self.assertEqual(
          {
              'correct_args': 2,
              'correct_not_novel': 1,
              'correct_novel': 3,
              'deleted_args': 0,
              'deleted_not_novel': 2,
              'deleted_novel': 2,
              'total_cost': -16.9,
          },
          {k: round(v, 6) for k, v in metrics.items()},
      )

  def test_api_call_baselines(self):
    prototype_string = textwrap.dedent(
        """\
        already_seen_tokens = [
            first_shared_call,
            shared_namespace,
            unshared_call_2,
        ]
        # completion starts here

        out_of_order_call()
        if True:
          first_shared_call()
          x = shared_namespace.second_shared_call(unshared_arg)
          unshared_call_1a(10)
          distinct_namespace_1.third_shared_call(shared_arg=10)
          shared_namespace.unshared_call_2(1234)

        distinct_namespace_1.fourth_shared_call("a", "b")
        foo += fifth_shared_call()
        shared_result = shared_namespace.sixth_shared_call()
        """
    )
    # Pretend cursor is after `completion starts here`
    start_string = 'completion starts here'
    cursor_position = prototype_string.find(start_string) + len(start_string)

    # Log probs are not used here.
    fake_tokens_with_log_probs = [(prototype_string[cursor_position:], 0.0)]

    # Construct a parser helper.
    parser_helper = parser_tools.ParserHelper(language='python')

    # Parse the prototype.
    prototype_as_nodes = parser_helper.parse_to_nodes(prototype_string)

    wrapper = call_sequence_wrapper.ApiCallSequenceWrapper(
        use_numba=False,
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
          solution_info['extracted_parts'], invert_confidence=True
      )

    self.assertEqual(
        {
            'all_calls_full': [
                ('out_of_order_call()', 'high'),
                ('\n', 'low'),
                ('first_shared_call()', 'high'),
                ('\n', 'low'),
                (
                    'x = shared_namespace.second_shared_call(unshared_arg)',
                    'high',
                ),
                ('\n', 'low'),
                ('unshared_call_1a(10)', 'high'),
                ('\n', 'low'),
                (
                    'distinct_namespace_1.third_shared_call(shared_arg=10)',
                    'high',
                ),
                ('\n', 'low'),
                ('shared_namespace.unshared_call_2(1234)', 'high'),
                ('\n', 'low'),
                ('distinct_namespace_1.fourth_shared_call("a", "b")', 'high'),
                ('\n', 'low'),
                ('foo += fifth_shared_call()', 'high'),
                ('\n', 'low'),
                (
                    'shared_result = shared_namespace.sixth_shared_call()',
                    'high',
                ),
                ('\n', 'low'),
            ],
            'all_calls_shorter': [
                ('out_of_order_call(', 'high'),
                (')\n', 'low'),
                ('first_shared_call(', 'high'),
                (')\nx = ', 'low'),
                ('shared_namespace.second_shared_call(', 'high'),
                ('unshared_arg)\n', 'low'),
                ('unshared_call_1a(', 'high'),
                ('10)\n', 'low'),
                ('distinct_namespace_1.third_shared_call(', 'high'),
                ('shared_arg=10)\n', 'low'),
                ('shared_namespace.unshared_call_2(', 'high'),
                ('1234)\n', 'low'),
                ('distinct_namespace_1.fourth_shared_call(', 'high'),
                ('"a", "b")\nfoo += ', 'low'),
                ('fifth_shared_call(', 'high'),
                (')\nshared_result = ', 'low'),
                ('shared_namespace.sixth_shared_call(', 'high'),
                (')\n', 'low'),
            ],
            'novel_calls_shorter': [
                ('out_of_order_call(', 'high'),
                (')\nfirst_shared_call()\nx = ', 'low'),
                ('shared_namespace.second_shared_call(', 'high'),
                ('unshared_arg)\n', 'low'),
                ('unshared_call_1a(', 'high'),
                ('10)\n', 'low'),
                ('distinct_namespace_1.third_shared_call(', 'high'),
                (
                    'shared_arg=10)\nshared_namespace.unshared_call_2(1234)\n',
                    'low',
                ),
                ('distinct_namespace_1.fourth_shared_call(', 'high'),
                ('"a", "b")\nfoo += ', 'low'),
                ('fifth_shared_call(', 'high'),
                (')\nshared_result = ', 'low'),
                ('shared_namespace.sixth_shared_call(', 'high'),
                (')\n', 'low'),
            ],
        },
        baseline_part_groups,
    )


if __name__ == '__main__':
  absltest.main()
