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

"""Tests for sequence_from_pseudo_parse_test."""

import textwrap

from absl.testing import absltest
from r_u_sure.parsing import sequence_from_pseudo_parse
from r_u_sure.parsing.pseudo_parser import stack_parser
from r_u_sure.tree_structure import sequence_node_helpers


def drop_trailing_whitespace(s: str) -> str:
  return '\n'.join([line.rstrip() for line in s.split()])


class SequenceFromPseudoParseTestTest(absltest.TestCase):

  def test_can_convert_pseudo_parse(self):
    """Ensures we can convert a pseudo parse with round-tripping."""

    source = textwrap.dedent("""\
        def foo(x):
          print("1, 2, 3")
          pass
          return x
        """)

    (pseudo_parsed,
     tokens_maybe_corrected,
     token_types_maybe_corrected,
     tokens_maybe_corrected_raw,
     _) = (
         stack_parser.PseudoParser.parse_and_maybe_preprocess(
             language='python', code=source, spaces_per_indent=2))
    actual = pseudo_parsed.human_readable_string(tokens_maybe_corrected)

    self.assertEqual(source, ''.join(tokens_maybe_corrected_raw))

    expected = r"""
| def foo(x):\n  print("1, 2, 3")\n  pass\n  return x\n | Node(0-24, root)
| def foo(x):\n  print("1, 2, 3")\n  pass\n  return x\n |    Node(0-24, split)
| def foo(x):\n________________________________________ |       Node(0-8, split)
| def__________________________________________________ |          Node(0-1, sibling)
| ___ _________________________________________________ |          Node(1-2, sibling)
| ____foo______________________________________________ |          Node(2-3, sibling)
| _______(x)___________________________________________ |          Node(3-6, paired)
| _______(_____________________________________________ |             Node(3-4, left)
| ________x____________________________________________ |             Node(4-5, middle)
| ________x____________________________________________ |                Node(4-5, sibling)
| _________)___________________________________________ |             Node(5-6, right)
| __________:__________________________________________ |          Node(6-7, sibling)
| ___________\n________________________________________ |          Node(7-8, sibling)
| _____________  print("1, 2, 3")\n  pass\n  return x\n |       Node(8-24, split)
| _____________  print("1, 2, 3")\n  pass\n  return x__ |          Node(8-23, paired)
| _____________________________________________________ |             Node(8-9, left)
| _____________  print("1, 2, 3")\n  pass\n  return x__ |             Node(9-22, middle)
| _____________  print("1, 2, 3")\n____________________ |                Node(9-15, split)
| _____________  ______________________________________ |                   Node(9-10, sibling)
| _______________print_________________________________ |                   Node(10-11, sibling)
| ____________________("1, 2, 3")______________________ |                   Node(11-14, paired)
| ____________________(________________________________ |                      Node(11-12, left)
| _____________________"1, 2, 3"_______________________ |                      Node(12-13, middle)
| _____________________"1, 2, 3"_______________________ |                         Node(12-13, sibling)
| ______________________________)______________________ |                      Node(13-14, right)
| _______________________________\n____________________ |                   Node(14-15, sibling)
| _________________________________  pass\n____________ |                Node(15-18, split)
| _________________________________  __________________ |                   Node(15-16, sibling)
| ___________________________________pass______________ |                   Node(16-17, sibling)
| _______________________________________\n____________ |                   Node(17-18, sibling)
| _________________________________________  return x__ |                Node(18-22, split)
| _________________________________________  __________ |                   Node(18-19, sibling)
| ___________________________________________return____ |                   Node(19-20, sibling)
| _________________________________________________ ___ |                   Node(20-21, sibling)
| __________________________________________________x__ |                   Node(21-22, sibling)
| _____________________________________________________ |             Node(22-23, right)
| ___________________________________________________\n |          Node(23-24, sibling)"""[1:]  # pylint: disable=line-too-long

    self.assertEqual(
        drop_trailing_whitespace(actual),
        drop_trailing_whitespace(expected))

    sequence_node = (
        sequence_from_pseudo_parse.pseudo_parse_node_to_nested_sequence_node(
            pseudo_parsed, tokens_maybe_corrected, token_types_maybe_corrected,
            decoration_node_types={
                stack_parser.NodeType.WHITE_SPACE_LEAF,
                stack_parser.NodeType.NON_CONTENT_LEAF},
            subtokenize_node_types={stack_parser.NodeType.STRING_LITERAL}))

    # Round-tripping back to source.
    self.assertEqual(source,
                     sequence_node_helpers.render_text_contents(sequence_node))

    # Full structure:
    expected_debug_view = r"""
GROUP(ROOT): 'def foo(x):\n  print("1, 2, 3")\n  pass\n  return x\n'
  GROUP(SPLIT_GROUP): 'def foo(x):\n  print("1, 2, 3")\n  pass\n  return x\n'
    GROUP(SPLIT_GROUP): 'def foo(x):\n'
      TOK(CONTENT_LEAF): 'def'
      DEC: ' '
      TOK(CONTENT_LEAF): 'foo'
      GROUP(MATCH): '(x)'
        TOK(MATCH_LEFT): '('
        GROUP(MATCH_INNER): 'x'
          TOK(CONTENT_LEAF): 'x'
        TOK(MATCH_RIGHT): ')'
      TOK(CONTENT_LEAF): ':'
      DEC: '\n'
    GROUP(SPLIT_GROUP): '  print("1, 2, 3")\n  pass\n  return x\n'
      GROUP(MATCH): '  print("1, 2, 3")\n  pass\n  return x'
        TOK(MATCH_LEFT): ''
        GROUP(MATCH_INNER): '  print("1, 2, 3")\n  pass\n  return x'
          GROUP(SPLIT_GROUP): '  print("1, 2, 3")\n'
            DEC: '  '
            TOK(CONTENT_LEAF): 'print'
            GROUP(MATCH): '("1, 2, 3")'
              TOK(MATCH_LEFT): '('
              GROUP(MATCH_INNER): '"1, 2, 3"'
                GROUP(STRING_LITERAL): '"1, 2, 3"'
                  TOK(subtoken): '"'
                  TOK(subtoken): '1'
                  TOK(subtoken): ','
                  TOK(subtoken): ' '
                  TOK(subtoken): '2'
                  TOK(subtoken): ','
                  TOK(subtoken): ' '
                  TOK(subtoken): '3'
                  TOK(subtoken): '"'
              TOK(MATCH_RIGHT): ')'
            DEC: '\n'
          GROUP(SPLIT_GROUP): '  pass\n'
            DEC: '  '
            TOK(CONTENT_LEAF): 'pass'
            DEC: '\n'
          GROUP(SPLIT_GROUP): '  return x'
            DEC: '  '
            TOK(CONTENT_LEAF): 'return'
            DEC: ' '
            TOK(CONTENT_LEAF): 'x'
        TOK(MATCH_RIGHT): ''
      DEC: '\n'"""[1:]

    actual_debug_view = sequence_node_helpers.render_debug(sequence_node)
    self.assertEqual(expected_debug_view, actual_debug_view)


if __name__ == '__main__':
  absltest.main()
