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

"""Utilities that use the pseudo parser."""

import re
from typing import Iterable, Tuple, Union, Optional, List

from r_u_sure.parsing.pseudo_parser import stack_parser


def infer_python_spaces_per_indent(
    python_code: str,
    candidate_spaces_per_indent: Iterable[int] = (1, 2, 3, 4)
    ) -> int:
  """Infers the largest number of spaces per indent consistent with python_code.

  Args:
    python_code: python code to analyse
    candidate_spaces_per_indent: number of spaces per indent to consider

  Returns:
    - the largest number of spaces per indent consistent with python_code or
      1 if there is no evidence.
  """

  if not python_code:
    return 1

  (_, _, token_types, tokens_maybe_corrected_raw, (_, _)) = (
      stack_parser.PseudoParser.parse_and_maybe_preprocess(
          'python', python_code, spaces_per_indent=1))

  indent_lengths = []
  for (token_type, previous_token_type, token_raw) in (
      zip(token_types[1:], token_types[:-1], tokens_maybe_corrected_raw[1:])):
    if (token_type == stack_parser.TokenType.WHITE_SPACE and
        previous_token_type in (
            stack_parser.TokenType.PYTHON_INDENT,
            stack_parser.TokenType.PYTHON_DEDENT) and
        len(token_raw)):
      indent_lengths.append(len(token_raw))

  if not indent_lengths:
    return 1

  inferred_indent_length = 1
  for candidate_indent_length in candidate_spaces_per_indent:
    if all([not this_indent_length % candidate_indent_length for
            this_indent_length in indent_lengths]):
      inferred_indent_length = candidate_indent_length

  return inferred_indent_length


def find_python_function_docstrings(python_code: str) -> Iterable[int]:
  """Infers the starting positions of the python function docstrings.

  Args:
    python_code: code to search for function docstrings

  Returns:
    - a list of indices at which docstrings begin (not including the three
      quotation marks at the start of a docstring).

  We search for triple quoted sting literals (single or double quotes) that
  occur as immediate rhs siblings of nodes that look like function signatures.
  The triple quote needs also to be indented at the level of the body
  corresponding to the function signature.
  """

  docstring_re = re.compile(r' *(\"\"\"|\'\'\')')
  function_re = re.compile(r' *def ')

  inferred_indent_length = infer_python_spaces_per_indent(python_code)
  (root, _, _, tokens_maybe_corrected_raw, (_, _)) = (
      stack_parser.PseudoParser.parse_and_maybe_preprocess(
          'python', python_code, spaces_per_indent=inferred_indent_length))

  # copy the tree to allow equality comparison of nodes
  root = stack_parser.Node.copy(root)

  def node_code_fragment(node: stack_parser.Node) -> str:
    """Returns the code corresponding to node."""
    return ''.join(tokens_maybe_corrected_raw[node.lo:node.hi])

  def range_code(hi: int) -> str:
    """Returns the code up to and including token number hi."""
    return ''.join(tokens_maybe_corrected_raw[:hi])

  def leading_indents(code_fragment: str) -> int:
    """Counts the number of inferred_indent_length sized leading whitespaces."""
    leading_indent_count = 0
    for i in range(
        inferred_indent_length, len(code_fragment), inferred_indent_length):
      if code_fragment[(i-inferred_indent_length):i].isspace():
        leading_indent_count = i // inferred_indent_length
      else:
        break
    return leading_indent_count

  docstring_indices = []
  for (node, node_index) in stack_parser.Node.depth_first_traversal(root):

    parent = node.parent
    if not parent:
      # if we are at the root node
      continue

    assert node in parent.children
    node_index = parent.children.index(node)
    if len(parent.children) <= (node_index + 1):
      # if node does not have a rhs sibling
      continue

    node_code = node_code_fragment(node)
    node_match = function_re.match(node_code)
    if not node_match:
      # if node does not look like a function specification
      continue

    rhs_sibling = parent.children[node_index+1]
    rhs_sibling_code = node_code_fragment(rhs_sibling)
    rhs_sibling_match = docstring_re.match(rhs_sibling_code)
    if not rhs_sibling_match:
      # if rhs_sibling does not look like a docstring
      continue

    if leading_indents(rhs_sibling_code) != (leading_indents(node_code) + 1):
      # if the would be docstring isn't indented like the function body
      continue

    n_chars_before_rhs_sibling = len(range_code(rhs_sibling.lo))
    n_chars_to_start_of_docstring_within_sibling = rhs_sibling_match.end()

    # start index includes the rhs_sibling code plus the characters within
    # rhs_sibling matched by docstring_re:

    docstring_indices.append(
        n_chars_before_rhs_sibling +
        n_chars_to_start_of_docstring_within_sibling)

  return docstring_indices


def infer_match_pair_truncation_index(
    root: stack_parser.Node,
    tokens_raw: List[str],
    token_types: List[stack_parser.TokenType],
    match_pair: Optional[Tuple[
        Union[str, stack_parser.TokenType],
        Union[str, stack_parser.TokenType]]],
    cursor_position: int,
    ) -> Union[int, None]:
  """Returns index into code=''.join(tokens_raw) at which the match_pair grouping which contains cursor_position is closed.

  Args:
    root: parse tree root
    tokens_raw: uncorrected string tokens for root
    token_types: token types for root
    match_pair: tuple of (left, right) of relevant match pair in parse tree
      to consider, or None to consider all match pairs. If these are of type
      stack_parser.TokenType, then both left and right "brackets" must match the
      type of match_pair[0] and match_pair[1], respectively. If instead
      match_pair are of type str, then if either the left or right bracket
      matches then we consider this a match, to facilitate processing of error
      corrected parse trees which may in such cases have e.g. an empty token in
      tokens_raw where the correction occurred.
    cursor_position: index into ''.join(tokens_raw) (i.e. the code) of the
      cursor.

  Returns:
    - the integer position or None if it does not exist
  """

  len_code = sum(map(len, tokens_raw))
  if cursor_position == len_code:
    return len_code
  assert cursor_position >= 0 and cursor_position < len_code
  assert match_pair is None or isinstance(match_pair[0], type(match_pair[1]))

  def _traverse(node: stack_parser.Node, match_pair_depth: int):
    # Yield (node, depth) pairs in depth first order, where depth is the number
    # of braces of type match_pair surrounding node.

    yield node, match_pair_depth
    for child in node.children:
      depth_increment = 0
      if child.annotation == 'middle':
        left, middle, right = node.children
        assert middle == child
        assert left.annotation == 'left'
        assert right.annotation == 'right'
        if match_pair is None:
          depth_increment = 1
        else:
          if isinstance(match_pair[0], stack_parser.TokenType):
            if token_types[left.lo] == match_pair[0]:
              assert token_types[right.lo] == match_pair[1]
              depth_increment = 1
          else:
            # we use an "or" to allow error corrected braces to match:
            if (tokens_raw[left.lo] == match_pair[0] or
                tokens_raw[right.lo] == match_pair[1]):
              depth_increment = 1
      yield from _traverse(child, match_pair_depth + depth_increment)

  code_index = 0
  cursor_match_pair_depth = None
  finished = False

  for node, match_pair_depth in _traverse(root, 0):
    if not node.children:
      # consider leaves
      if code_index > cursor_position and cursor_match_pair_depth is None:
        # we found the cursor
        cursor_match_pair_depth = match_pair_depth
        if match_pair_depth == 0:
          # if the cursor is not within match_pair braces, return None
          return None
      if node.hi != node.lo:
        # update the index into ''.join(tokens_raw), i.e. the code
        code_index += len(tokens_raw[node.lo])
      if finished:
        return code_index
      if (
          (cursor_match_pair_depth is not None) and
          match_pair_depth == (cursor_match_pair_depth - 1)
          ):
        # we came out of the surrounding brace at the level of the cursor so we
        # are essentially done, but we use finished to process one more leaf
        # and thereby include the closing match pair, for consistency with
        # infer_split_truncation_index
        finished = True

  return None


def infer_split_truncation_index(
    root: stack_parser.Node,
    tokens_raw: List[str],
    splitters: Optional[Tuple[str, ...]],
    cursor_position: int,
    ) -> int:
  """Returns the index into ''.join(tokens_raw) of the termination of the next split demarcated by splitters which occurs after cursor_position, ignoring parse depth.

  Args:
    root: parse tree root
    tokens_raw: uncorrected string tokens for root
    splitters: tuple of splitters to consider, or None to consider all splitters
    cursor_position: index into ''.join(tokens_raw) (i.e. the code) of the
      cursor.

  Returns:
    - the integer position or None if it does not exist
  """

  len_code = sum(map(len, tokens_raw))
  if cursor_position == len_code:
    return len_code
  assert cursor_position >= 0 and cursor_position < len_code

  code_index = 0
  after_cursor = False

  for node, _ in stack_parser.Node.depth_first_traversal(root):
    if not node.children:
      # consider leaves
      if code_index > cursor_position and not after_cursor:
        # we found the cursor
        after_cursor = True
      if node.hi != node.lo:
        # update the index into ''.join(tokens_raw), i.e. the code
        code_index += len(tokens_raw[node.lo])
      if after_cursor:
        if (node.parent.annotation == 'split' and
            node.parent.children[-1] == node and
            tokens_raw[node.lo] in splitters):
          # we found the split immediately proceeding the cursor
          return code_index

  # shouldn't happen, but to future proof we fall back to returning the final
  # cursor position:
  return code_index


def infer_truncation_with_fallbacks(
    code: str,
    cursor_position: int,
    language: str,
    spaces_per_indent: Optional[int] = None,
) -> int:
  """Infers a truncation index into code using infer_match_pair_truncation_index, falling back to infer_split_truncation_index if necessary.

  Args:
    code: source code
    cursor_position: index into source code where the prefix ends
    language: e.g. 'python' or 'java'
    spaces_per_indent: python only; if None, inferred automatically.

  Returns:
    - index into code at which to truncate.
  """

  if language == 'python':
    splitters = ('\n',)
    match_pair = (
        stack_parser.TokenType.PYTHON_INDENT,
        stack_parser.TokenType.PYTHON_DEDENT,
        )
    if spaces_per_indent is None:
      spaces_per_indent = infer_python_spaces_per_indent(code)
  else:
    splitters = (';',)
    match_pair = ('{', '}')

  (root, _, token_types, tokens_raw, (_, _)) = (
      stack_parser.PseudoParser.parse_and_maybe_preprocess(
          language=language,
          code=code,
          spaces_per_indent=spaces_per_indent,
          )
      )

  truncation_index = infer_match_pair_truncation_index(
      root,
      tokens_raw,
      token_types,
      match_pair,
      cursor_position,
      )

  if truncation_index is not None:
    return truncation_index
  else:
    return infer_split_truncation_index(
        root,
        tokens_raw,
        splitters,
        cursor_position
        )


def infer_truncation_pydocstring(
    code: str,
    cursor_position: int,
) -> Union[int, None]:
  """Returns the truncation index that immediately follows the next triple quote after that assumed to immediately precede cursor_position.

  Args:
    code: source code
    cursor_position: index into source code where the prefix ends. the type of
      triple quote is taken from the last three characters (singe or double
      quotation marks).

  Returns:
    - index into code at which to truncate or None if there is no matching
      triple quote.
  """

  triple_quotes = code[:cursor_position][-3:]
  assert triple_quotes in ['"""', "'''"], f'{triple_quotes} not a triple quote'
  if triple_quotes not in code[cursor_position:]:
    return None
  truncation_index = (
      cursor_position +
      code[cursor_position:].find(triple_quotes) +
      len(triple_quotes))

  return truncation_index
