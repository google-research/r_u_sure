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

"""Tests for token probability alignments."""

from absl.testing import absltest
from r_u_sure.tree_structure import packed_sequence_nodes
from r_u_sure.tree_structure import sequence_nodes
from r_u_sure.tree_structure import token_prob_align

# Some example code.
EXAMPLE_CODE = """\
def foo(my_value):
  print(my_value is None)
"""

# Cursor is after `:`
EXAMPLE_CURSOR_LOCATION = EXAMPLE_CODE.find(':') + 1

# Hypothetical log probs, based on what SentencePiece tends to produce.
# Of note: whitespace tokens have probabilities, some whitespace is attached
# to the start of non-whitespace tokens, and some programming language tokens
# are split across multiple Python tokens.
EXAMPLE_MODEL_TOKENS_AND_LOG_PROBS = [
    ('\n', 0.0),
    (' ', -0.1),
    (' print', -0.6),
    ('(', -0.001),
    ('my', -2.0),
    ('_value', -1.0),
    (' is', -1.0),
    (' None', -0.1),
    (')', -0.5),
    ('\n', -0.15),
    ('', -1.0),
]

GroupNode = sequence_nodes.GroupNode
TextTokenNode = sequence_nodes.TextTokenNode
TextDecorationNode = sequence_nodes.TextDecorationNode

# Based on the output of the pseudoparser, but with match types removed.
# Of note: whitespace is tagged as decoration, there are zero-length
# indent/dedent characters, and the grouping of characters is based on Python
# syntax.
PARSED_TREE = [
    GroupNode([
        GroupNode(
            [
                # (simulating context nodes being removed by truncation)
                TextDecorationNode('\n'),
            ]
        ),
        GroupNode([
            GroupNode([
                TextTokenNode(''),  # (represents the indent)
                GroupNode([
                    TextDecorationNode('  '),
                    TextTokenNode('print'),
                    GroupNode([
                        TextTokenNode('('),
                        GroupNode([
                            TextTokenNode('my_value'),
                            TextDecorationNode(' '),
                            TextTokenNode('is'),
                            TextDecorationNode(' '),
                            TextTokenNode('None'),
                        ]),
                        TextTokenNode(')'),
                    ]),
                ]),
                TextTokenNode(''),  # (represents the dedent)
            ]),
            TextDecorationNode('\n'),
        ]),
    ])
]

EXPECTED_FLATTENED_AFTER_ALIGN = [
    ('\n', 0.0),
    ('', -0.1),  # (indent token gets its log-prob from the whitespace)
    ('  ', 0.0),  # (but whitespace itself is a decoration node)
    ('print', -0.6),
    ('(', -0.001),
    ('my_value', -3.0),  # -2.0 + -1.0 from the constituent tokens
    (' ', 0.0),
    ('is', -1.0),
    (' ', 0.0),
    ('None', -0.1),
    (')', -0.5),
    ('', -0.15),  # (dedent token gets its log-prob from the newline)
    ('\n', 0.0),  # (but newline itself is a decoration node)
]

# As above, but skipping decorations, and indexing based on tree position.
EXPECTED_PROBS_BY_PREORDER_ID = {
    5: -0.1,
    8: -0.6,
    10: -0.001,
    12: -3.0,
    14: -1.0,
    16: -0.1,
    17: -0.5,
    18: -0.15,
}


class TokenProbAlignTest(absltest.TestCase):

  def test_align_and_flatten(self):
    packed_sequence = packed_sequence_nodes.pack(PARSED_TREE)
    aligned_log_probs_by_preorder = token_prob_align.align_token_log_probs(
        EXAMPLE_MODEL_TOKENS_AND_LOG_PROBS,
        packed_sequence,
        must_consume_all_model_tokens=True,
    )

    with self.subTest('align_token_probs'):
      self.assertEqual(
          aligned_log_probs_by_preorder, EXPECTED_PROBS_BY_PREORDER_ID
      )

    flattened = list(
        token_prob_align.flatten_token_log_probs_from_parsed(
            packed_sequence, aligned_log_probs_by_preorder
        )
    )

    with self.subTest('flatten_token_log_probs_from_parsed'):
      self.assertEqual(flattened, EXPECTED_FLATTENED_AFTER_ALIGN)


if __name__ == '__main__':
  absltest.main()
