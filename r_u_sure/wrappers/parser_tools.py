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

"""High-level wrapper utilities for parsing using the pseudoparser."""

import numba
from r_u_sure.parsing import sequence_from_pseudo_parse
from r_u_sure.parsing.pseudo_parser import stack_parser
from r_u_sure.parsing.pseudo_parser import utilities as pseudo_parse_utilities
from r_u_sure.tree_structure import packed_sequence_nodes
from r_u_sure.tree_structure import sequence_nodes
from r_u_sure.tree_structure import transforms

PackedSequenceNodeStorage = packed_sequence_nodes.PackedSequenceNodeStorage


def pack_sequence_from_pseudoparser(
    sequence: list[sequence_nodes.SequenceNode], with_numba: bool
) -> PackedSequenceNodeStorage:
  """Helper function to pack a sequence."""
  if with_numba:
    return packed_sequence_nodes.pack(
        sequence,
        as_numba_typed=True,
        match_type_numba_type=numba.typeof("CONTENT_LEAF"),
    )
  else:
    return packed_sequence_nodes.pack(sequence)


def allow_regions_around_pseudoparse_node(node) -> bool:
  """Returns True if we should allow uncertainty regions around a given node.

  We don't allow regions in any MATCH nodes. Regions are still
  allowed around the children of the MATCH_INNER node.

  We also don't allow regions to start right before or end right after a
  single newline character.

  Args:
    node: The node to check.

  Returns:
    True if we should insert region markers around this node.
  """
  if node.match_type in ("MATCH_LEFT", "MATCH_INNER", "MATCH_RIGHT"):
    return False
  if (
      isinstance(node, sequence_nodes.TextTokenNode)
      and node.text_contents == "\n"
  ):
    # Note: this should be rare, since most newlines should be decoration nodes.
    return False
  return True


class ParserHelper:
  """Helper object for parsing."""

  def __init__(
      self,
      language: str,
      flatten: bool = False,
  ):
    """Sets up the parser helper.

    Args:
      language: Language to interpret `code_or_text` in. Must either be a
        language supported by `stack_parser.PseudoParser` or "raw_simple_tokens"
        to bypass the pseudoparser and just heuristically emit flat tokens.
      flatten: Whether to flatten the tree structure to a single sequence.

    Returns:
      A sequence of nodes for downstream use.
    """
    self._language = language
    self._flatten = flatten

  def parse_to_nodes(
      self, code_or_text: str
  ) -> list[sequence_nodes.SequenceNode]:
    """Builds a tree from code in a given language, by parsing or just tokenizing.

    Args:
      code_or_text: Sequence to transform into nodes.

    Returns:
      A sequence of nodes for downstream use.
    """
    if self._language == "raw_simple_tokens":
      # Just (sub)tokenize the raw string itself.
      return sequence_from_pseudo_parse.subtokenize_token(code_or_text)
    else:
      if self._language == "python":
        inferred_spaces_per_indent = (
            pseudo_parse_utilities.infer_python_spaces_per_indent(code_or_text)
        )
      else:
        inferred_spaces_per_indent = 1
      (
          pseudo_parsed,
          tokens_maybe_corrected,
          token_types_maybe_corrected,
          raw_strings,
          n_pre_and_post,
      ) = stack_parser.PseudoParser.parse_and_maybe_preprocess(
          language=self._language,
          code=code_or_text,
          spaces_per_indent=inferred_spaces_per_indent,
      )
      del tokens_maybe_corrected, n_pre_and_post  # Unused
      root_sequence_node = (
          sequence_from_pseudo_parse.pseudo_parse_node_to_nested_sequence_node(
              pseudo_parsed,
              raw_strings,
              token_types_maybe_corrected,
              decoration_node_types={
                  stack_parser.NodeType.WHITE_SPACE_LEAF,
                  stack_parser.NodeType.NON_CONTENT_LEAF,
              },
              subtokenize_node_types={stack_parser.NodeType.STRING_LITERAL},
          )
      )
      result = [root_sequence_node]
      if self._flatten:
        return transforms.flatten_groups(result)
      else:
        return result
