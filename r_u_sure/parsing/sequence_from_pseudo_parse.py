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

"""Helpers to build SequenceNode representations from pseudo_parser results."""

import functools
from typing import AbstractSet, List

from r_u_sure.parsing import subtokenizer
from r_u_sure.parsing.pseudo_parser import stack_parser
from r_u_sure.tree_structure import sequence_nodes


DEFAULT_DECORATION_NODE_TYPES = frozenset({
    stack_parser.NodeType.WHITE_SPACE_LEAF,
})

DEFAULT_SUBTOKENIZE_NODE_TYPES = frozenset({
    stack_parser.NodeType.STRING_LITERAL,
})


def pseudo_parse_node_to_nested_sequence_node(
    root: stack_parser.Node,
    tokens: List[str],
    token_types: List[stack_parser.TokenType],
    decoration_node_types: AbstractSet[stack_parser.NodeType] = (
        DEFAULT_DECORATION_NODE_TYPES
    ),
    subtokenize_node_types: AbstractSet[stack_parser.NodeType] = (
        DEFAULT_SUBTOKENIZE_NODE_TYPES
    ),
) -> sequence_nodes.SequenceNode:
  """Converts a stack_parser Node to a SequenceNode representation.

  Walks the tree starting at the root. If a given node's type is in either of
  `decoration_node_types` or `subtokenize_node_types`, that node is directly
  converted into either a decoration node or a group of subtoken nodes.
  Otherwise, group nodes are processed recursively, and leaf nodes are treated
  as tokens.

  Args:
    root: The pseudo parser node to convert.
    tokens: A list of text tokens indexed by root.
    token_types: A list of token types indexed by root.
    decoration_node_types: Node types to treat as decoration.
    subtokenize_node_types: Node types to subtokenize using a generic tokenizer,
      which produces a flat list of tokens that can be concatenated to
      reconstruct the original input. This can be used to allow uncertain
      regions inside single logical tokens, e.g. inside docstrings or long
      identifiers.

  Returns:
    A sequence node representation of the provided node.
  """
  node_type = root.type(tokens, token_types)
  if node_type in decoration_node_types:
    # Output this node as a decoration node (regardless of if it has children)
    return sequence_nodes.TextDecorationNode(root.text(tokens))
  elif node_type in subtokenize_node_types:
    # Output this node as a group of subtokens (regardless of if it has
    # children)
    return sequence_nodes.GroupNode(
        children=subtokenize_token(root.text(tokens)), match_type=node_type.name
    )
  elif not root.children:
    # This is a leaf node (no children). Output it as a text token.
    return sequence_nodes.TextTokenNode(
        root.text(tokens), match_type=node_type.name
    )
  else:
    # This node has children (and isn't a decoration or subtokenizeable).
    # We require that all of the text contained by this node is also contained
    # by one of its children. This is always true for stack_parser.Node.
    boundaries_1 = [root.lo] + [child.hi for child in root.children]
    boundaries_2 = [child.lo for child in root.children] + [root.hi]
    if boundaries_1 != boundaries_2:
      raise ValueError(
          f"stack_parser.Node {root} does not contain "
          "exactly its children's contents!"
      )

    rec = functools.partial(
        pseudo_parse_node_to_nested_sequence_node,
        tokens=tokens,
        token_types=token_types,
        decoration_node_types=decoration_node_types,
        subtokenize_node_types=subtokenize_node_types,
    )
    return sequence_nodes.GroupNode(
        children=[rec(child) for child in root.children],
        match_type=node_type.name,
    )


def subtokenize_token(token: str) -> list[sequence_nodes.SequenceNode]:
  """Subtokenizes an individual token by splitting at various points.

  Subtokenization is done using the CuBERT simple lossless tokenizer, which
  breaks up arbitrary strings at possibly-semantically-interesting points.
  All subtokens are treated as TextTokenNodes and not as TextDecorationNodes,
  even if they are entirely whitespace, since whitespace in arbitrary strings
  may be important.

  Args:
    token: The token string to subtokenize.

  Returns:
    A list of subtokens.
  """
  simple_subtokens = subtokenizer.code_to_tokens_simple_lossless(token)
  return [
      sequence_nodes.TextTokenNode(subtoken, "subtoken")
      for subtoken in simple_subtokens
  ]
