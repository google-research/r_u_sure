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

"""Tests for sequence_node_helpers."""

import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from r_u_sure.tree_structure import sequence_node_helpers
from r_u_sure.tree_structure import sequence_nodes

SAMPLE_TREE = sequence_nodes.GroupNode([
    sequence_nodes.TextTokenNode("a", "token type"),
    sequence_nodes.TextDecorationNode(" "),
    sequence_nodes.TextTokenNode("b"),
    sequence_nodes.EarlyExitNode(),
    sequence_nodes.TextDecorationNode(" "),
    sequence_nodes.GroupNode([
        sequence_nodes.RegionStartNode(),
        sequence_nodes.TextTokenNode("("),
        sequence_nodes.TextTokenNode("c"),
        sequence_nodes.TextDecorationNode(" "),
        sequence_nodes.TextTokenNode("d"),
        sequence_nodes.TextTokenNode(")"),
        sequence_nodes.RegionEndNode(),
    ], "group type"),
])
DEEP_TREE = [
    sequence_nodes.TextTokenNode("a"),
    sequence_nodes.GroupNode([
        sequence_nodes.GroupNode([
            sequence_nodes.TextTokenNode("b"),
        ]),
        sequence_nodes.GroupNode([
            sequence_nodes.GroupNode([
                sequence_nodes.TextTokenNode("c"),
            ]),
        ]),
    ]),
]

WalkTag = sequence_node_helpers.WalkTag
NestedNodePath = sequence_nodes.NestedNodePath


class SequenceNodeHelpersTest(parameterized.TestCase):

  def test_render_text_contents(self):
    text_contents = sequence_node_helpers.render_text_contents(SAMPLE_TREE)
    expected_text_contents = "a b (c d)"
    self.assertEqual(text_contents, expected_text_contents)

  def test_render_debug(self):
    debug_view = sequence_node_helpers.render_debug(SAMPLE_TREE)
    expected_debug = textwrap.dedent("""\
        GROUP: 'a b (c d)'
          TOK(token type): 'a'
          DEC: ' '
          TOK: 'b'
          EarlyExitNode()
          DEC: ' '
          GROUP(group type): '(c d)'
            RegionStartNode()
            TOK: '('
            TOK: 'c'
            DEC: ' '
            TOK: 'd'
            TOK: ')'
            RegionEndNode()
        """).rstrip()
    self.assertEqual(debug_view, expected_debug)

  def test_compute_depth(self):
    depth = sequence_node_helpers.compute_depth(DEEP_TREE)
    self.assertEqual(depth, 4)

  @parameterized.parameters([
      (False, False),
      (False, True),
      (True, False),
      (True, True),
  ])
  def test_walk_with_paths(self, before: bool, after: bool):
    result = list(sequence_node_helpers.walk_with_paths(
        SAMPLE_TREE.children,
        visit_groups_before=before,
        visit_groups_after=after))

    expected = [
        (sequence_nodes.TextTokenNode("a", "token type"), NestedNodePath(
            (0,)), WalkTag.LEAF),
        (sequence_nodes.TextDecorationNode(" "), NestedNodePath(
            (1,)), WalkTag.LEAF),
        (sequence_nodes.TextTokenNode("b"), NestedNodePath((2,)), WalkTag.LEAF),
        (sequence_nodes.EarlyExitNode(), NestedNodePath((3,)), WalkTag.LEAF),
        (sequence_nodes.TextDecorationNode(" "), NestedNodePath(
            (4,)), WalkTag.LEAF),
    ]
    the_group_node = SAMPLE_TREE.children[-1]
    if before:
      expected.append((
          the_group_node,
          NestedNodePath((5,)),
          WalkTag.GROUP_BEFORE_CHILDREN,
      ))
    expected.extend([
        (sequence_nodes.RegionStartNode(), NestedNodePath(
            (5, 0)), WalkTag.LEAF),
        (sequence_nodes.TextTokenNode("("), NestedNodePath(
            (5, 1)), WalkTag.LEAF),
        (sequence_nodes.TextTokenNode("c"), NestedNodePath(
            (5, 2)), WalkTag.LEAF),
        (sequence_nodes.TextDecorationNode(" "), NestedNodePath(
            (5, 3)), WalkTag.LEAF),
        (sequence_nodes.TextTokenNode("d"), NestedNodePath(
            (5, 4)), WalkTag.LEAF),
        (sequence_nodes.TextTokenNode(")"), NestedNodePath(
            (5, 5)), WalkTag.LEAF),
        (sequence_nodes.RegionEndNode(), NestedNodePath(
            (5, 6)), WalkTag.LEAF),
    ])
    if after:
      expected.append((
          the_group_node,
          NestedNodePath((5,)),
          WalkTag.GROUP_AFTER_CHILDREN,
      ))

    self.assertEqual(result, expected)


if __name__ == "__main__":
  absltest.main()
