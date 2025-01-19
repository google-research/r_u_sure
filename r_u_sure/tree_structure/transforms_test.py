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

"""Tests for transforms."""

import textwrap

from absl.testing import absltest
from r_u_sure.tree_structure import sequence_node_helpers
from r_u_sure.tree_structure import sequence_nodes
from r_u_sure.tree_structure import transforms

SAMPLE_SEQUENCE = [
    sequence_nodes.TextTokenNode("a", "token a"),
    sequence_nodes.TextDecorationNode(" "),
    sequence_nodes.GroupNode(
        [
            sequence_nodes.GroupNode(
                [sequence_nodes.TextTokenNode("b")], "inner b group"
            ),
        ],
        "outer b group",
    ),
    sequence_nodes.TextDecorationNode(" "),
    sequence_nodes.GroupNode(
        [
            sequence_nodes.GroupNode(
                [
                    sequence_nodes.TextTokenNode("("),
                    sequence_nodes.TextTokenNode("c"),
                    sequence_nodes.TextDecorationNode(" "),
                    sequence_nodes.TextTokenNode("d"),
                    sequence_nodes.TextTokenNode(")"),
                    sequence_nodes.TextDecorationNode(" "),
                ],
                "inner big group",
            ),
        ],
        "outer big group",
    ),
    sequence_nodes.TextDecorationNode(" "),
]

SAMPLE_SEQUENCE_LONGER_NODES = [
    sequence_nodes.TextTokenNode("aaaaa", "token a"),
    sequence_nodes.TextDecorationNode("-----"),
    sequence_nodes.GroupNode(
        [
            sequence_nodes.GroupNode(
                [sequence_nodes.TextTokenNode("bbbbb")], "inner b group"
            ),
        ],
        "outer b group",
    ),
    sequence_nodes.TextDecorationNode("-----"),
    sequence_nodes.GroupNode(
        [
            sequence_nodes.TextTokenNode("((((("),
            sequence_nodes.TextTokenNode("ccccc"),
            sequence_nodes.GroupNode(
                [
                    sequence_nodes.TextTokenNode("((((("),
                    sequence_nodes.TextTokenNode("ddddd"),
                    sequence_nodes.RegionStartNode(),
                    sequence_nodes.TextDecorationNode("-----"),
                    sequence_nodes.TextTokenNode("eeeee"),
                    sequence_nodes.TextTokenNode(")))))"),
                ],
                "inner big group",
            ),
            sequence_nodes.TextDecorationNode("-----"),
            sequence_nodes.TextTokenNode("fffff"),
            sequence_nodes.GroupNode(
                [sequence_nodes.TextTokenNode("ggggg")], "inner g group"
            ),
            sequence_nodes.TextTokenNode(")))))"),
        ],
        "outer big group",
    ),
    sequence_nodes.TextDecorationNode("-----"),
]


class TransformsTest(absltest.TestCase):

  def test_expected_starting_sequence(self):
    """Visualizes the starting sequence for comparison with other tests.

    This "test" serves as a reference for comparing the rendering of the
    starting sequence with renderings of the transformed sequences. It does not
    test any new logic.
    """
    result = sequence_node_helpers.render_debug(SAMPLE_SEQUENCE)
    expected_debug_view = textwrap.dedent(
        """\
        TOK(token a): 'a'
        DEC: ' '
        GROUP(outer b group): 'b'
          GROUP(inner b group): 'b'
            TOK: 'b'
        DEC: ' '
        GROUP(outer big group): '(c d) '
          GROUP(inner big group): '(c d) '
            TOK: '('
            TOK: 'c'
            DEC: ' '
            TOK: 'd'
            TOK: ')'
            DEC: ' '
        DEC: ' '"""
    )
    self.assertEqual(result, expected_debug_view)

  def test_flatten_groups(self):
    result = sequence_node_helpers.render_debug(
        transforms.flatten_groups(SAMPLE_SEQUENCE)
    )
    expected_debug_view = textwrap.dedent(
        """\
        TOK(token a): 'a'
        DEC: ' '
        TOK: 'b'
        DEC: ' '
        TOK: '('
        TOK: 'c'
        DEC: ' '
        TOK: 'd'
        TOK: ')'
        DEC: ' '
        DEC: ' '"""
    )
    self.assertEqual(result, expected_debug_view)

  def test_flatten_singleton_groups(self):
    result = sequence_node_helpers.render_debug(
        transforms.flatten_singleton_groups(SAMPLE_SEQUENCE)
    )
    expected_debug_view = textwrap.dedent(
        """\
        TOK(('token a',)): 'a'
        DEC: ' '
        TOK(('outer b group', 'inner b group', None)): 'b'
        DEC: ' '
        GROUP(('outer big group', 'inner big group')): '(c d) '
          TOK((None,)): '('
          TOK((None,)): 'c'
          DEC: ' '
          TOK((None,)): 'd'
          TOK((None,)): ')'
          DEC: ' '
        DEC: ' '"""
    )
    self.assertEqual(result, expected_debug_view)

  def test_insert_early_exit_everywhere(self):
    result = sequence_node_helpers.render_debug(
        transforms.insert_early_exit(SAMPLE_SEQUENCE, allowed_within_group=True)
    )
    expected_debug_view = textwrap.dedent(
        """\
        EarlyExitNode()
        TOK(token a): 'a'
        EarlyExitNode()
        DEC: ' '
        GROUP(outer b group): 'b'
          GROUP(inner b group): 'b'
            TOK: 'b'
            EarlyExitNode()
          EarlyExitNode()
        EarlyExitNode()
        DEC: ' '
        GROUP(outer big group): '(c d) '
          GROUP(inner big group): '(c d) '
            TOK: '('
            EarlyExitNode()
            TOK: 'c'
            EarlyExitNode()
            DEC: ' '
            TOK: 'd'
            EarlyExitNode()
            TOK: ')'
            EarlyExitNode()
            DEC: ' '
          EarlyExitNode()
        EarlyExitNode()
        DEC: ' '"""
    )
    self.assertEqual(result, expected_debug_view)

  def test_insert_early_exit_top_level(self):
    result = sequence_node_helpers.render_debug(
        transforms.insert_early_exit(
            SAMPLE_SEQUENCE, allowed_within_group=False
        )
    )
    expected_debug_view = textwrap.dedent(
        """\
        EarlyExitNode()
        TOK(token a): 'a'
        EarlyExitNode()
        DEC: ' '
        GROUP(outer b group): 'b'
          GROUP(inner b group): 'b'
            TOK: 'b'
        EarlyExitNode()
        DEC: ' '
        GROUP(outer big group): '(c d) '
          GROUP(inner big group): '(c d) '
            TOK: '('
            TOK: 'c'
            DEC: ' '
            TOK: 'd'
            TOK: ')'
            DEC: ' '
        EarlyExitNode()
        DEC: ' '"""
    )
    self.assertEqual(result, expected_debug_view)

  def test_insert_region_options_around_subsequences(self):
    result = sequence_node_helpers.render_debug(
        transforms.insert_region_options_around_subsequences(
            SAMPLE_SEQUENCE, allow_empty_regions=True
        )
    )
    expected_debug_view = textwrap.dedent(
        """\
        RegionStartNode()
        RegionEndNode()
        TOK(token a): 'a'
        RegionEndNode()
        DEC: ' '
        RegionStartNode()
        RegionEndNode()
        GROUP(outer b group): 'b'
          RegionStartNode()
          RegionEndNode()
          GROUP(inner b group): 'b'
            RegionStartNode()
            RegionEndNode()
            TOK: 'b'
            RegionEndNode()
            RegionStartNode()
            RegionEndNode()
          RegionEndNode()
          RegionStartNode()
          RegionEndNode()
        RegionEndNode()
        DEC: ' '
        RegionStartNode()
        RegionEndNode()
        GROUP(outer big group): '(c d) '
          RegionStartNode()
          RegionEndNode()
          GROUP(inner big group): '(c d) '
            RegionStartNode()
            RegionEndNode()
            TOK: '('
            RegionEndNode()
            RegionStartNode()
            RegionEndNode()
            TOK: 'c'
            RegionEndNode()
            DEC: ' '
            RegionStartNode()
            RegionEndNode()
            TOK: 'd'
            RegionEndNode()
            RegionStartNode()
            RegionEndNode()
            TOK: ')'
            RegionEndNode()
            DEC: ' '
            RegionStartNode()
            RegionEndNode()
          RegionEndNode()
          RegionStartNode()
          RegionEndNode()
        RegionEndNode()
        DEC: ' '
        RegionStartNode()
        RegionEndNode()"""
    )
    self.assertEqual(result, expected_debug_view)

  def test_insert_region_options_around_subsequences_nonempty(self):
    result = sequence_node_helpers.render_debug(
        transforms.insert_region_options_around_subsequences(
            SAMPLE_SEQUENCE, allow_empty_regions=False
        )
    )
    expected_debug_view = textwrap.dedent(
        """\
        RegionStartNode()
        TOK(token a): 'a'
        RegionEndNode()
        DEC: ' '
        RegionStartNode()
        GROUP(outer b group): 'b'
          RegionStartNode()
          GROUP(inner b group): 'b'
            RegionStartNode()
            TOK: 'b'
            RegionEndNode()
          RegionEndNode()
        RegionEndNode()
        DEC: ' '
        RegionStartNode()
        GROUP(outer big group): '(c d) '
          RegionStartNode()
          GROUP(inner big group): '(c d) '
            RegionStartNode()
            TOK: '('
            RegionEndNode()
            RegionStartNode()
            TOK: 'c'
            RegionEndNode()
            DEC: ' '
            RegionStartNode()
            TOK: 'd'
            RegionEndNode()
            RegionStartNode()
            TOK: ')'
            RegionEndNode()
            DEC: ' '
          RegionEndNode()
        RegionEndNode()
        DEC: ' '"""
    )
    self.assertEqual(result, expected_debug_view)

  def test_insert_region_options_around_subsequences_filtered(self):
    filter_subset = {"token a", "inner b group"}
    result = sequence_node_helpers.render_debug(
        transforms.insert_region_options_around_subsequences(
            SAMPLE_SEQUENCE,
            allow_empty_regions=False,
            node_filter=lambda node: (node.match_type in filter_subset),
        )
    )
    expected_debug_view = textwrap.dedent(
        """\
        RegionStartNode()
        TOK(token a): 'a'
        RegionEndNode()
        DEC: ' '
        GROUP(outer b group): 'b'
          RegionStartNode()
          GROUP(inner b group): 'b'
            TOK: 'b'
          RegionEndNode()
        DEC: ' '
        GROUP(outer big group): '(c d) '
          GROUP(inner big group): '(c d) '
            TOK: '('
            TOK: 'c'
            DEC: ' '
            TOK: 'd'
            TOK: ')'
            DEC: ' '
        DEC: ' '"""
    )
    self.assertEqual(result, expected_debug_view)

  def test_insert_region_options_around_single_nodes(self):
    result = sequence_node_helpers.render_debug(
        transforms.insert_region_options_around_single_nodes(
            SAMPLE_SEQUENCE
        )
    )
    expected_debug_view = textwrap.dedent(
        """\
        GROUP(region_scope): 'a'
          RegionStartNode()
          TOK(token a): 'a'
          RegionEndNode()
        DEC: ' '
        GROUP(region_scope): 'b'
          RegionStartNode()
          GROUP(outer b group): 'b'
            GROUP(region_scope): 'b'
              RegionStartNode()
              GROUP(inner b group): 'b'
                GROUP(region_scope): 'b'
                  RegionStartNode()
                  TOK: 'b'
                  RegionEndNode()
              RegionEndNode()
          RegionEndNode()
        DEC: ' '
        GROUP(region_scope): '(c d) '
          RegionStartNode()
          GROUP(outer big group): '(c d) '
            GROUP(region_scope): '(c d) '
              RegionStartNode()
              GROUP(inner big group): '(c d) '
                GROUP(region_scope): '('
                  RegionStartNode()
                  TOK: '('
                  RegionEndNode()
                GROUP(region_scope): 'c'
                  RegionStartNode()
                  TOK: 'c'
                  RegionEndNode()
                DEC: ' '
                GROUP(region_scope): 'd'
                  RegionStartNode()
                  TOK: 'd'
                  RegionEndNode()
                GROUP(region_scope): ')'
                  RegionStartNode()
                  TOK: ')'
                  RegionEndNode()
                DEC: ' '
              RegionEndNode()
          RegionEndNode()
        DEC: ' '"""
    )
    self.assertEqual(result, expected_debug_view)

  def test_truncate_prefix_at_offset(self):
    result = sequence_node_helpers.render_debug(
        transforms.truncate_prefix_at_offset(
            SAMPLE_SEQUENCE_LONGER_NODES, start_offset=37
        )
    )
    expected_debug_view = textwrap.dedent(
        """\
        GROUP(outer big group): 'ddd-----eeeee)))))-----fffffggggg)))))'
          GROUP(inner big group): 'ddd-----eeeee)))))'
            TOK: 'ddd'
            RegionStartNode()
            DEC: '-----'
            TOK: 'eeeee'
            TOK: ')))))'
          DEC: '-----'
          TOK: 'fffff'
          GROUP(inner g group): 'ggggg'
            TOK: 'ggggg'
          TOK: ')))))'
        DEC: '-----'"""
    )
    self.assertEqual(result, expected_debug_view)

  def test_truncate_prefix_at_offset_end_of_token(self):
    result = sequence_node_helpers.render_debug(
        transforms.truncate_prefix_at_offset(
            SAMPLE_SEQUENCE_LONGER_NODES, start_offset=40
        )
    )
    expected_debug_view = textwrap.dedent(
        """\
        GROUP(outer big group): '-----eeeee)))))-----fffffggggg)))))'
          GROUP(inner big group): '-----eeeee)))))'
            RegionStartNode()
            DEC: '-----'
            TOK: 'eeeee'
            TOK: ')))))'
          DEC: '-----'
          TOK: 'fffff'
          GROUP(inner g group): 'ggggg'
            TOK: 'ggggg'
          TOK: ')))))'
        DEC: '-----'"""
    )
    self.assertEqual(result, expected_debug_view)


if __name__ == "__main__":
  absltest.main()
