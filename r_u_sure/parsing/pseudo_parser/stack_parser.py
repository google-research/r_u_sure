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

"""Utility to pseudo parse code from multiple languages."""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Iterable, List, Optional, Tuple, Union

from absl import logging


LEADING_WHITE_SPACE_REGEX = re.compile(r'(\s*)')
ALL_WHITE_SPACE_REGEX = re.compile(r'^(\s+)$')


class ParseError(Exception):
  """Parse error occurs if the brackets do not match and correction is not used.
  """


def _unzip_pairs(list_of_pairs) -> ...:
  """Unzips a list of pairs.

  Args:
    list_of_pairs: [(a1,b1), (a2, b2), ...]
  Returns:
    (a1, a2, ...), (b1, b2, ...)
  """
  return (
      tuple(a for a, b in list_of_pairs),
      tuple(b for a, b in list_of_pairs),
  )


@enum.unique
class TokenType(enum.IntEnum):
  """Types of nodes."""
  NUMBER = 0
  ID = 1
  OP = 2
  PUNC = 3
  BRACE = 4
  NEWLINE = 5
  TAB = 6
  WHITE_SPACE = 7
  OTHER = 8
  STRING = 9
  COMMENT_HASH = 10
  COMMENT_SLASHSLASH = 11
  COMMENT_SLASHSTAR = 12
  C_PREPROCESSOR = 13
  NON_CONTENT = 14
  PYTHON_INDENT = 15
  PYTHON_DEDENT = 16


class RegexCodec:
  """Regular expression based token encoder / decoder."""

  def __init__(self, token_specification: List[Tuple[str, str]]):
    """Constructor.

    Args:
      token_specification: order dependent list of (token_type, regex) pairs
    """
    self.token_specification = token_specification
    self.tok_regex = '|'.join(
        '(?P<%s>%s)' % pair for pair in token_specification)
    self.token_to_index = dict()
    self.index_to_token = dict()
    self.index_to_token_type = dict()

  def add_token(self, token: str, token_type: TokenType, non_unique=False
                ) -> int:
    """Includes a new token in the set of tokens.

    Args:
      token: the raw string (e.g. a subsequence of source code) to represent.
      token_type: the type of the token.
      non_unique: iff False, assert that token has not been added previously.

    Returns:
      index: a newly added index that maps to token.
    """
    if not non_unique:
      assert token not in self.token_to_index
    index = len(self.token_to_index) + 1
    self.token_to_index[token] = index
    self.index_to_token[index] = token
    self.index_to_token_type[index] = token_type
    return index

  def encode(self, code: str) -> Tuple[List[int], List[TokenType]]:
    """Encode a string to tokens.

    Args:
      code: string (e.g. of source code) to encode of length M

    Returns:
      - token_list: list of N <= M integer tokens.
      - token_types: list of M <= M  corresponding token types.
    """
    tokens = []
    token_types = []
    for match in re.finditer(self.tok_regex, code, re.DOTALL):
      type_str = match.lastgroup
      token = match.group()
      token_type = TokenType[type_str]
      if token not in self.token_to_index:
        self.add_token(token, token_type)
      encoded_token = self.token_to_index[token]
      assert token_type == self.index_to_token_type[encoded_token], (
          code, self.token_specification)
      tokens.append(encoded_token)
      token_types.append(token_type)
    return tokens, token_types

  def decode_token(self,
                   indices: List[Union[int, str]]) -> List[Union[int, str]]:
    return list(self.index_to_token[index] for index in indices)

  def decode_type(self,
                  indices: List[Union[int, str]]) -> List[TokenType]:
    return list(self.index_to_token_type[index] for index in indices)

  def __len__(self) -> int:
    return len(self.token_to_index)

  @classmethod
  def preset(cls, language: str) -> RegexCodec:
    """Preset instances of class.

    Args:
      language: case insensitive preset name

    Returns:
      RegexCodec object

    Raises:
      Exception: if the preset is unknown.
    """

    basic = [
        ('NUMBER', r'\d+(\.\d*)?'),
        ('ID', r'[A-Za-z0-9_]+'),
        ('OP', r'[+\-*/]'),
        ('PUNC', r'[,.;:=><]'),
        ('BRACE', r'[()\[\]{}]'),
        ('NEWLINE', r'\n'),
        ('TAB', r'\t'),
        ('WHITE_SPACE', r' +'),
        ('OTHER', r'.'),
    ]
    python_strings_and_comments = [
        ('STRING', r'''(\"\"\"|\'\'\'|\"|\')((?<!\\)\\\2|.)*?\2'''),
        ('COMMENT_HASH', r'(#)[^\n]*'),
    ]
    c_comments = [
        ('COMMENT_SLASHSLASH', r'(//)[^\n]*'),
        ('COMMENT_SLASHSTAR',
         r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"'),
    ]
    c_strings = [
        ('STRING', r'''(\")((?<!\\)\\\2|.)*?\2'''),
    ]
    c_macros = [
        ('C_PREPROCESSOR',
         (r'(#include|#define|#undef|#ifdef|#ifndef|#line|#if|#endif|#error)'
          r'(\\\n|[^(\n|//|/\*)])*')),
    ]
    specifications = {
        'python_preprocessor':
            python_strings_and_comments + basic,
        'cpp':
            c_macros + c_comments + c_strings + basic,
        'java':
            c_comments + c_strings + basic,
        'javascript':
            c_comments + c_strings + basic,
        'simple':
            basic,
    }

    specification = specifications[language.lower()]

    for name, _ in specification:
      assert name in TokenType.__members__, name
    assert len(specification) == len(set([name for name, _ in specification]))

    return cls(specification)


@enum.unique
class NodeType(enum.IntEnum):
  """Types of nodes."""
  ROOT = 0
  MATCH = 1
  MATCH_LEFT = 2
  MATCH_RIGHT = 3
  MATCH_INNER = 4
  CONTENT_LEAF = 5
  NON_CONTENT_LEAF = 6
  WHITE_SPACE_LEAF = 7
  SPLIT_GROUP = 8
  STRING_LITERAL = 9


@dataclasses.dataclass
class Node:
  """Node in a pseudo parse tree.

  Attributes:
    lo: start of range of tokens spanned (inclusive)
    hi: end of range of tokens spanned (non-inclusive)
    parent: parent in the doubly linked tree
    children: list of type Node of children in the tree
    annotation: an arbitrary string to associate with the node as a label
  """
  lo: int
  hi: Optional[int] = None
  parent: Optional[Node] = None
  annotation: Optional[str] = ''
  children: List[Node] = dataclasses.field(default_factory=list)

  def type(self, tokens: List[str], token_types: List[TokenType]) -> NodeType:
    """The type of Node."""
    if self.annotation == 'root':
      return NodeType.ROOT
    elif self.annotation == 'sibling':
      content_str = self.text(tokens)
      assert not self.children
      if not content_str:
        return NodeType.NON_CONTENT_LEAF
      else:
        if ALL_WHITE_SPACE_REGEX.match(content_str):
          return NodeType.WHITE_SPACE_LEAF
        elif all(token_type == NodeType.STRING_LITERAL
                 for token_type in token_types[self.lo:self.hi]):
          return NodeType.STRING_LITERAL
        else:
          return NodeType.CONTENT_LEAF
    elif self.annotation == 'paired':
      return NodeType.MATCH
    elif self.annotation == 'left':
      return NodeType.MATCH_LEFT
    elif self.annotation == 'right':
      return NodeType.MATCH_RIGHT
    elif self.annotation == 'middle':
      return NodeType.MATCH_INNER
    elif self.annotation == 'split':
      return NodeType.SPLIT_GROUP
    assert False, 'bad node annotation {}'.format(self.annotation)

  def __repr__(self) -> str:

    return 'Node({}-{}, {})'.format(self.lo, self.hi, self.annotation)

  def human_readable_string(self,
                            tokens: Optional[List[Union[int, str]]] = None
                            ) -> str:
    """Pretty formatted string.

    Args:
      tokens: list of tokens

    Returns:
      the string
    """

    columns = [[], []]
    i_token_mask, i_node = 0, 1

    def unquoted_string_repr(x: Union[int, str]):
      if isinstance(x, str):
        return repr(x)[1:-1].replace('\\\\', '\\')
      elif isinstance(x, int):
        return str(x)

    def mask_view(node):
      if not tokens: return ''
      rval = ''
      lo = node.lo if node.lo else 0
      hi = node.hi if node.hi else lo+1
      for i, token in enumerate(tokens):
        if lo <= i < hi:
          rval += unquoted_string_repr(token)
        else:
          rval += len(unquoted_string_repr(token)) * '_'
      return unquoted_string_repr(rval)

    def _f(node, depth):
      columns[i_token_mask].append(mask_view(node))
      columns[i_node].append('   ' * depth + str(node))
      for child in node.children:
        _f(child, depth + 1)

    _f(self, 0)

    maxlens = [max([len(item) for item in lines]) for lines in columns]

    rval = ''
    for row in zip(*columns):
      rval += '| ' + ' | '.join(
          [c.ljust(ll) for c, ll in zip(row, maxlens)]) + '\n'

    return rval.strip()

  def text(self, tokens: List[Union[int, str]]) -> str:
    return ''.join(map(str, tokens[self.lo:self.hi]))

  def __len__(self) -> int:
    return self.hi - self.lo

  @classmethod
  def copy(cls, root: Node) -> Node:
    """Copy a tree.

    Args:
      root: node to copy

    Returns:
      a node with a deep copy of root
    """

    def _copy(node):
      n = Node(
          lo=node.lo,
          hi=node.hi,
          parent=node.parent,
          annotation=node.annotation)
      n.children = [_copy(child) for child in node.children]
      for child in n.children:
        child.parent = n
      return n

    return _copy(root)

  @classmethod
  def depth_first_traversal(cls, root: Node) -> Iterable[Tuple[Node, int]]:
    """Traverses a tree depth first.

    Args:
      root: starting node

    Yields:
      - a sequence of (node, index) starting with root, in depth first order.
      For reasons of efficiency we return also index which has the value of
      node.parent.children.index(node) if node.parent is not None, otherwise
      None (if node.parent is None).
    """

    def _traverse(node, index):
      yield (node, index)
      for index, child in enumerate(node.children):
        yield from _traverse(child, index)

    if root.parent is None:
      first_index = None
    else:
      first_index = root.parent.children.index(root)

    yield from _traverse(root, first_index)


def split_parse(
    root: Node,
    tokens: List[Union[int, str]],
    split_tokens: List[Union[int, str]],
    un_splittable_match_pairs: List[Tuple[str, str]],
    sub_match_pair: Optional[Tuple[Union[int, str], Union[int, str]]] = None,
    ) -> Node:
  """Split siblings into groups demarcated by the tokens in split_tokens.

  Args:
    root: parse tree to split
    tokens: text or indices of tokens
    split_tokens: text or indices of tokens that demarcate split points
    un_splittable_match_pairs: don't split when inside these parentheses
    sub_match_pair: a tuple (left, right) denoting a match pair to be
      nested under the preceding split_tokens demarcated chunk; for example if
      sub_match_pair = (pyindent, pydedent) then each indented block will be
      nested under node with two children, the first of which is the function
      signature (if the block is a function body) and the second of which is
      the function body.

  Returns:
    new parse tree root node
  """

  root = Node.copy(root)

  if not split_tokens:
    return root

  un_splittable_openers, un_splittable_closers = _unzip_pairs(
      un_splittable_match_pairs)

  def _split(node: Node):
    # we will terminate splitting if node represents matched braces that are
    # unsplittable. this is the case if node has three children, the first
    # and last of which represent matched unsplittable braces:
    if node.annotation == 'paired':
      assert len(node.children) == 3
      left = tokens[node.children[0].lo]
      if left in un_splittable_openers:
        right = tokens[node.children[-1].hi-1]
        if right == un_splittable_closers[un_splittable_openers.index(left)]:
          return
    # otherwise group children by splitters and put the groups under common
    # parents:
    groups = [[]]
    for child in node.children:
      groups[-1].append(child)
      if len(child) == 1 and tokens[child.lo] in split_tokens:
        groups.append([])
    original_node_children = node.children
    if len(groups) > 1 and node.annotation != 'paired':
      new_children = []
      for group in groups:
        if not group: continue
        n = Node(
            lo=group[0].lo, hi=group[-1].hi, parent=node, annotation='split')
        n.children = group
        for child in group:
          child.parent = n
        new_children.append(n)
      node.children = new_children
    for child in original_node_children:
      if len(child) > 1:
        _split(child)

  def _sub_match(node: Node):
    # yield (new_child, new_parent) pairs, where new_child should be grouped
    # under new_parent, and the old children of new_parent should be placed
    # under a new first child of new_parent. for example, new_child could be
    # an indented/dedented body of a python function and new_parent could be the
    # function signature that precedes that block.
    if node.annotation == 'paired':
      assert len(node.children) == 3
      left, right = tokens[node.children[0].lo], tokens[node.children[-1].hi-1]
      if (left == sub_match_pair[0] and
          right == sub_match_pair[1] and
          node.parent is not None and
          node.parent.parent is not None):
        grand_parent = node.parent.parent
        new_child = node.parent
        new_parent = grand_parent.children[
            grand_parent.children.index(new_child)-1]
        if (new_child.annotation == 'split' and
            new_parent.annotation == 'split' and
            new_parent.hi == new_child.lo):
          yield new_child, new_parent
    for child in node.children:
      yield from _sub_match(child)

  def _sub_match_transplant(new_child, new_parent):
    # make new_parent have two children, the first of which has children that
    # are all of the old children of new_parent, and the second of which is
    # new_child.
    new_first_child = Node(
        lo=new_parent.lo,
        hi=new_parent.hi,
        parent=new_parent,
        annotation='split')
    for old_child in new_parent.children:
      old_child.parent = new_first_child
    new_first_child.children = new_parent.children

    new_parent.children = [new_first_child, new_child]
    new_parent.parent.children.remove(new_child)
    new_child.parent = new_parent
    new_parent.hi = new_child.hi

  _split(root)

  if sub_match_pair is not None:
    for new_child, new_parent in list(_sub_match(root)):
      _sub_match_transplant(new_child, new_parent)

  return root


def valid_parse_ranges(root: Node, n: int) -> bool:
  """Returns true if root is a valid parse of range(0, n).

  It is valid if
  - the root has range lo = 0, hi = n
  - all nodes under root nodes have ranges (lo to hi) that are subdivided by
    the ranges of their children
  - no ranges include None
  - all children of each node have that node as their parent.

  Args:
    root: root of parse to check
    n: length of parsed tokens

  Returns:
    true iff it is root is a valid parse tree.
  """

  def _check(node):
    if node.children:
      if node.lo != node.children[0].lo:
        return False
      if node.hi != node.children[-1].hi:
        return False
      if node.lo is None or node.hi is None:
        return False
      lows = [n.lo for n in node.children[1:]]
      his = [n.hi for n in node.children[:-1]]
      if lows != his:
        return False
      for child in node.children:
        if child.parent != node:
          return False
        if not _check(child):
          return False
    return True

  return root.lo == 0 and root.hi == n and _check(root)


def stack_based_bracket_match(tokens: List[Union[int, str]],
                              match_pairs: List[Tuple[str, str]],
                              error_tolerant: bool = False) -> Node:
  """Stack based parsing algorithm.

  Args:
    tokens: list of tokens
    match_pairs: list of (left, right) text or indices of tokens to match
    error_tolerant: whether to recover from parse errors of unmatched closers
      such as
        ''.join(tokens) = '(])' with match_pairs = [tuple('()'), tuple('[]')]
      where the ']' does not match the open '(' and will raise a ParseError
      unless error_tolerant is True, in which case the ']' will be treated like
      a regular (non match_pair) token.

  Returns:
    root node of parse tree, wherein bracket matching leads to nodes that have
    exactly three children, (call them left, middle and right) where left and
    right match a single token each, for e.g. the opening and closing
    parentheses, and middle is a parent node that recursively matches the
    contents of the parentheses.

  Raises:
    ParseError: if not error tolerant and there is a parsing error as described
    above.
  """

  openers, closers = _unzip_pairs(match_pairs)
  root = Node(lo=0, hi=len(tokens), parent=None, annotation='root')
  opener_stack = []
  parent = root
  closer_to_opener = dict(zip(closers, openers))

  for i, token in enumerate(tokens):
    # is this token a simple non bracket opener / closer (e.g. an identifier) ?
    plain_token = (token not in openers) and (token not in closers)
    # is this token a well formed (matched) parenthesis closer ?
    matched_closer = (
        (token in closers) and
        opener_stack and
        (opener_stack[-1] == closer_to_opener[token]))
    # is this token an ill formed parenthesis closer that could not have been
    # fixed by prefixing the entire code block with a matching opener ?
    unprefixable_unmatched_closer = (
        (token in closers) and
        opener_stack and
        (opener_stack[-1] != closer_to_opener[token]))
    if token in openers:
      # n will have three children: left, middle, and right. left and right
      # have no children, and will represent the opening and closing braces.
      # the middle child will usually have children that represent the contents
      # of the brace.
      n = Node(lo=i, hi=None, parent=parent, annotation='paired')
      n.children.append(Node(lo=i, hi=i+1, parent=n, annotation='left'))
      opener_stack.append(token)
      parent.children.append(n)
      contents = Node(lo=i+1, hi=None, parent=n, annotation='middle')
      n.children.append(contents)
      parent = contents
    elif matched_closer:
      # parent.parent is an opener node, i.e. as explained in the comment
      # above it will ultimately have three children. At present it two
      # children, left and middle. we will add the third (right) one below.
      assert len(parent.parent.children) == 2
      middle_node = parent.parent.children[1]
      # we now know the rhs of the range:
      middle_node.hi = i
      right_node = Node(lo=i, hi=i + 1, parent=parent.parent,
                        annotation='right')
      parent.parent.children.append(right_node)
      # this rhs is one greater as it includes the closing token
      parent.parent.hi = i + 1
      opener_stack.pop()
      parent = parent.parent.parent
    elif plain_token or unprefixable_unmatched_closer:
      if unprefixable_unmatched_closer and not error_tolerant:
        raise ParseError('empty stack')
      n = Node(lo=i, hi=i+1, parent=parent, annotation='sibling')
      parent.children.append(n)

  if opener_stack and not error_tolerant:
    raise ParseError('unclosed openers')

  return root


def error_correcting_prefix_and_suffix(
    tokens: List[Union[int, str]],
    match_pairs: List[Tuple[Union[int, str], Union[int, str]]],
    verbose: bool = False,
) -> Tuple[List[Union[int, str]], List[Union[int, str]]]:
  """Determines the prefix and suffix that would allow parsing to work.

  Args:
    tokens: list of text or indices of tokens
    match_pairs: list of (left, right) text or indices of tokens to match
    verbose: output more via logging.info ?

  Returns:
    (prefix, suffix) such that prefix + tokens + suffix is error corrected
  """
  openers, closers = _unzip_pairs(match_pairs)
  closer_to_opener = dict(zip(closers, openers))
  opener_to_closer = dict(zip(openers, closers))
  opener_stack = []
  error_correcting_prefix = []

  if verbose:
    logging.info('tokens %s', repr(tokens))
    logging.info('match_pairs %s', repr(match_pairs))

  for i, token in enumerate(tokens):
    plain_token = (token not in openers) and (token not in closers)
    matched_closer = ((token in closers) and opener_stack and
                      (opener_stack[-1] == closer_to_opener[token]))
    prefixable_unmatched_closer = (token in closers) and (not opener_stack)
    unprefixable_unmatched_closer = (
        (token in closers) and (opener_stack) and
        (opener_stack[-1] != closer_to_opener[token]))
    if token in openers:
      # opener
      opener_stack.append(token)
      info = 'LEFT : opener_stack {}'.format(repr(opener_stack))
    elif matched_closer:
      # matched closer
      opener_stack.pop()
      info = 'RIGHT : opener_stack {}'.format(repr(opener_stack))
    elif prefixable_unmatched_closer:
      # unmatched closer that can be corrected with a prefix
      error_correcting_prefix.append(closer_to_opener[token])
      info = 'PREFIXABLE RIGHT : opener_stack {}'.format(repr(opener_stack))
    elif plain_token or unprefixable_unmatched_closer:
      # either a non opener/closer (which is not an error), or an
      # unmatched closer that cannot be corrected with a prefix (which we will
      # correct by treating it as a non-closer)
      if plain_token:
        info = 'PLAIN : opener_stack {}'.format(repr(opener_stack))
      elif unprefixable_unmatched_closer:
        info = 'NON-PREFIXABLE RIGHT : opener_stack {}'.format(
            repr(opener_stack))
    if verbose:
      logging.info('i = %i token = %s : %s', i, str(token), info)

  error_correcting_suffix = [opener_to_closer[token] for token in opener_stack]

  if verbose:
    logging.info('error_correcting_suffix = %s error_correcting_prefix = %s',
                 str(error_correcting_suffix), str(error_correcting_prefix))

  return list(reversed(error_correcting_prefix)), list(
      reversed(error_correcting_suffix))


def python_indent_dedent_desugar(root: Node, tokens: List[str],
                                 encoded_tokens: List[int],
                                 indent_token: int, dedent_token: int,
                                 newline_token: int,
                                 spaces_per_indent: int) -> List[int]:
  """Insert indent and dedent tokens that can be matched to group python blocks.

  Args:
    root: root of pseudo parse tree using basic bracket matching.
    tokens: list of text of tokens
    encoded_tokens: list of indices of tokens
    indent_token: index of python indent token to insert
    dedent_token: index of python dedent token to insert
    newline_token: index of newline token
    spaces_per_indent: spaces per python indent

  Returns:
    list of indices of tokens with indents and dedents inserted. note that
    dedents are guaranteed to be inserted before the final newline at which
    they occur, in order to allow downstream splitting with split_parse to
    meaningfully split indent/dedent blocks based on newlines.
  """

  def _top_level_line_slices():
    """Yield slices that index into tokens.

    The slices represent lines of code that are not split up in the tree, i.e.
    they are represented by contiguous ranges of siblings that are at the
    highest level, i.e. children of root.
    In this way, if we have already bracket matched to obtain root, we will
    avoid errant desugaring e.g. by processing newlines within a function
    argument list or list definition.

    Yields:
      slices that index into tokens.
    """
    first_node_of_line = None
    for depth_one_node in root.children:
      if not first_node_of_line:
        first_node_of_line = depth_one_node
      if encoded_tokens[depth_one_node.hi - 1] == newline_token:
        yield slice(first_node_of_line.lo, depth_one_node.hi)
        first_node_of_line = None
    if first_node_of_line:
      yield slice(first_node_of_line.lo, first_node_of_line.hi)

  def _line_indent(line_slice):
    """Count the indents in a line slice."""
    line_string = ''.join(tokens[line_slice])
    if ALL_WHITE_SPACE_REGEX.match(line_string):
      return None
    leading_white_space = LEADING_WHITE_SPACE_REGEX.match(line_string).group(0)  # pytype: disable=attribute-error  # re-none
    n_spaces = len(leading_white_space)
    return n_spaces // spaces_per_indent

  current_indent = 0
  output_tokens = []
  for line_slice in _top_level_line_slices():
    line_indent = _line_indent(line_slice)
    if line_indent is not None:
      indent_change = line_indent - current_indent
      if indent_change > 0:
        output_tokens += indent_change * [indent_token]
      elif indent_change < 0:
        # dedents go before the newline, to make the subsequent splitting
        # on newlines (by the function split_parse) easy:
        popped = output_tokens.pop()
        assert popped == newline_token
        output_tokens.extend((-indent_change) * [dedent_token])
        output_tokens.append(popped)
      current_indent = line_indent
    output_tokens += encoded_tokens[line_slice]
  indent_change = 0 - current_indent
  if indent_change > 0:
    output_tokens += indent_change * [indent_token]
  elif indent_change < 0:
    popped = output_tokens.pop()
    assert popped == newline_token
    output_tokens.extend((-indent_change) * [dedent_token])
    output_tokens.append(popped)

  return output_tokens


class PseudoParser:
  """Pseudo parser class.

  Attributes:
    match_pairs: list of pairs to match e.g. left and right parentheses
    split_tokens: list of tokens to split siblings based on
    split_types: list of token types to split siblings based on
    un_splittable_match_pairs: matches within which not to split with splitters
    codec: token encoder / decoder object
    encoded_match_pairs: match_pairs (see constructor) encoded by codec
    encoded_un_splittable_match_pairs: un_splittable_match_pairs (see
      constructor) encoded by codec
    encoded_splitters: splitters (see constructor) encoded by codec
    newline_terminate_correction: add a final newline if the bracket matching
      correction leaves a non-newline at the end of the code snippet
  """

  LANGUAGES = ('python_preprocessor', 'simple', 'cpp', 'java', 'javascript')
  LANGUAGE_TO_CODEC_PRESET = {
      'simple': 'simple',
      'python_preprocessor': 'python_preprocessor',
      'cpp': 'cpp',
      'java': 'java',
      'javascript': 'javascript'
  }

  def __init__(self, match_pairs: List[Tuple[Union[int, str], Union[int, str]]],
               split_tokens: List[Union[int, str]],
               un_splittable_match_pairs: List[Tuple[Union[int, str],
                                                     Union[int, str]]],
               split_types: List[str],
               codec: RegexCodec, newline_terminate_correction: bool):
    self.match_pairs = match_pairs
    self.split_tokens = split_tokens
    self.un_splittable_match_pairs = un_splittable_match_pairs
    self.split_types = split_types
    self.codec = codec
    self.newline_terminate_correction = newline_terminate_correction

    def encode_one(s):
      encoded, _ = self.codec.encode(s)
      assert len(encoded) == 1
      encoded = encoded[0]
      return encoded

    self.encoded_match_pairs = [
        tuple(map(encode_one, pair)) for pair in self.match_pairs
    ]

    self.encoded_un_splittable_match_pairs = [
        tuple(map(encode_one, pair)) for pair in self.un_splittable_match_pairs
    ]

    self.encoded_splitters = list(map(encode_one, self.split_tokens))

  def pseudo_parse(self, code: str) -> Tuple[
      Node,
      List[Union[int, str]],
      List[TokenType],
      List[Union[int, str]],
      Tuple[int, int]]:
    """Pseudo parse code.

    Args:
      code: snippet of code

    Returns:
      tuple containing:
        - root of the parse tree
        - list of N tokens with corrections, that root indexes into
        - list of N token types
        - list of N strings of the raw code for each token, with empty strings
          in place of tokens that were added during error correction.
        - a tuple of (n_pre, n_post) where n_pre (respectively n_post) is the
          number of error correcting tokens prepended (respectively appended)
    """

    encoded_tokens, token_types = self.codec.encode(code)

    encoded_error_correcting_prefix, encoded_error_correcting_suffix = (
        error_correcting_prefix_and_suffix(
            encoded_tokens, self.encoded_match_pairs))

    if encoded_error_correcting_suffix and self.newline_terminate_correction:
      encoded_error_correcting_suffix.append(self.codec.encode('\n')[0][0])

    encoded_tokens_maybe_corrected = (
        encoded_error_correcting_prefix +
        encoded_tokens +
        encoded_error_correcting_suffix)
    tokens_maybe_corrected = self.codec.decode_token(
        encoded_tokens_maybe_corrected)
    tokens_maybe_corrected_raw = (
        [''] * len(encoded_error_correcting_prefix) +
        self.codec.decode_token(encoded_tokens) +
        [''] * len(encoded_error_correcting_suffix)
    )
    tokens_types_maybe_corrected = self.codec.decode_type(
        encoded_tokens_maybe_corrected)

    encoded_parsed_unsplit = stack_based_bracket_match(
        encoded_tokens_maybe_corrected,
        self.encoded_match_pairs,
        error_tolerant=True)
    assert valid_parse_ranges(encoded_parsed_unsplit,
                              len(encoded_tokens_maybe_corrected))

    encoded_token_type_splitters = [
        encoded_token for encoded_token, token_type
        in zip(encoded_tokens, token_types)
        if token_type in self.split_types]
    encoded_splitters = self.encoded_splitters + encoded_token_type_splitters

    encoded_parsed_split = split_parse(
        encoded_parsed_unsplit,
        encoded_tokens_maybe_corrected,
        encoded_splitters,
        self.encoded_un_splittable_match_pairs,
        None)
    assert valid_parse_ranges(encoded_parsed_split,
                              len(encoded_tokens_maybe_corrected))

    return (encoded_parsed_split,
            tokens_maybe_corrected,
            tokens_types_maybe_corrected,
            tokens_maybe_corrected_raw,
            (len(encoded_error_correcting_prefix),
             len(encoded_error_correcting_suffix)))

  @classmethod
  def preset(cls, language: str) -> 'PseudoParser':
    """Preset instance of class.

    Args:
      language: string

    Returns:
      PseudoParser

    Raises:
      Exception: if language is unknown
    """
    if language == 'python_preprocessor':
      # these are the parameters for the python *preprocessor*, which parses
      # enough to determine which newlines separate statements (as opposed to
      # e.g. those occurring within function signatures, that do not separate
      # statements), to facilitate indent / dedent desugaring.
      # hence, no split_tokens or split_token_types are required here, but are
      # rather defined within the python specific branch of
      # parse_and_maybe_preprocess(), for use in the second pass of parsing
      # after indent / dedent desugaring.
      match_pairs = [tuple('()'), tuple('[]'), tuple('{}')]
      split_tokens = []
      un_splittable_match_pairs = [tuple('()'), tuple('[]'), tuple('[]')]
      split_types = []
      codec = RegexCodec.preset(cls.LANGUAGE_TO_CODEC_PRESET[language])
      newline_terminate_correction = True
    elif language == 'simple':
      match_pairs = [tuple('()'), tuple('[]'), tuple('{}')]
      split_tokens = [';']
      un_splittable_match_pairs = [tuple('()'), tuple('[]')]
      split_types = []
      codec = RegexCodec.preset(cls.LANGUAGE_TO_CODEC_PRESET[language])
      newline_terminate_correction = False
    elif language == 'cpp':
      match_pairs = [tuple('()'), tuple('[]'), tuple('{}')]
      split_tokens = [';']
      un_splittable_match_pairs = []
      # for cpp (and e.g. java and javascript), we split based on both
      # semi-colons and additional split_types defined here.
      # that way, e.g. comments are are not grouped with whatever else is in
      # between the surrounding semi-colons. This is necessary because e.g.
      # c-style comments are not demarcated by semi-colons, unlike c++
      # statements.
      split_types = [
          TokenType.C_PREPROCESSOR,
          TokenType.COMMENT_SLASHSLASH,
          TokenType.COMMENT_SLASHSTAR]
      codec = RegexCodec.preset(cls.LANGUAGE_TO_CODEC_PRESET[language])
      newline_terminate_correction = False
    elif language == 'java':
      match_pairs = [tuple('()'), tuple('[]'), tuple('{}')]
      split_tokens = [';']
      un_splittable_match_pairs = []
      split_types = [TokenType.COMMENT_SLASHSLASH, TokenType.COMMENT_SLASHSTAR]
      codec = RegexCodec.preset(cls.LANGUAGE_TO_CODEC_PRESET[language])
      newline_terminate_correction = False
    elif language == 'javascript':
      match_pairs = [tuple('()'), tuple('[]'), tuple('{}')]
      split_tokens = [';']
      un_splittable_match_pairs = []
      split_types = [TokenType.COMMENT_SLASHSLASH, TokenType.COMMENT_SLASHSTAR]
      codec = RegexCodec.preset(cls.LANGUAGE_TO_CODEC_PRESET[language])
      newline_terminate_correction = False
    else:
      raise Exception('unknown language {}'.format(language))

    return PseudoParser(match_pairs,
                        split_tokens,
                        un_splittable_match_pairs,
                        split_types,
                        codec,
                        newline_terminate_correction)

  @classmethod
  def parse_and_maybe_preprocess(
      cls,
      language: str,
      code: str,
      spaces_per_indent: int = 1) -> Tuple[
          Node,
          List[Union[int, str]],
          List[TokenType],
          List[Union[int, str]],
          Tuple[int, int]]:
    """Entry point for preprocessing, error correction and parsing.

    Args:
      language: preset language string e.g. 'python' or 'simple'.
      code: code to pseudo parse
      spaces_per_indent: only affects the 'python' language option

    Returns:
      tuple containing:
        - parse root
        - list of N tokens with corrections, that root indexes into
        - list of N token types
        - list of N strings of the raw code for each token, with empty strings
          in place of tokens that were added during error correction.
        - a tuple of (n_pre, n_post) where n_pre (respectively n_post) is the
          number of error correcting tokens prepended (respectively appended)
    """

    if language == 'python':

      if code[-1] != '\n':
        code = code + '\n'
        appended_newline = True
      else:
        appended_newline = False

      # parse enough to determine if newlines and indents represent python
      # indent / dedents (as opposed to newlines within e.g. lists which
      # do not), with error correction:

      pre_pseudo_parser = cls.preset('python_preprocessor')
      (pre_pseudo_parsed,
       pre_tokens_maybe_corrected,
       _,
       pre_tokens_maybe_corrected_raw,
       (n_pre, n_post)) = pre_pseudo_parser.pseudo_parse(code)

      if appended_newline:
        # we added n_post error correcting tokens, so the newline that was
        # was appended is now at position (-npost-1):
        assert pre_tokens_maybe_corrected_raw[-n_post-1] == '\n'
        pre_tokens_maybe_corrected_raw[-n_post-1] = ''

      assert (len(pre_tokens_maybe_corrected) ==
              len(pre_tokens_maybe_corrected_raw))

      assert valid_parse_ranges(pre_pseudo_parsed,
                                len(pre_tokens_maybe_corrected))

      ((newline_token,), (_,)) = pre_pseudo_parser.codec.encode('\n')
      # add tokens to represent the python indent (resp. dedent) which we treat
      # like opening (resp. closing) parentheses. these tokens are associated
      # with empty strings and so don't affect the code string, but do affect
      # the pseudo parse tree construction.
      indent_token = pre_pseudo_parser.codec.add_token(
          '', TokenType.PYTHON_INDENT, non_unique=True)
      dedent_token = pre_pseudo_parser.codec.add_token(
          '', TokenType.PYTHON_DEDENT, non_unique=True)

      encoded_pre_tokens_maybe_corrected, _ = (
          pre_pseudo_parser.codec.encode(''.join(pre_tokens_maybe_corrected)))

      encoded_desugared_tokens_maybe_corrected = python_indent_dedent_desugar(
          pre_pseudo_parsed,
          pre_tokens_maybe_corrected,
          encoded_pre_tokens_maybe_corrected,
          indent_token,
          dedent_token,
          newline_token,
          spaces_per_indent)
      desugared_tokens_maybe_corrected = pre_pseudo_parser.codec.decode_token(
          encoded_desugared_tokens_maybe_corrected)
      desugared_tokens_types_maybe_corrected = (
          pre_pseudo_parser.codec.decode_type(
              encoded_desugared_tokens_maybe_corrected))

      def _get_raw_desugared_tokens(
          raw,
          encoded_desugared,
          special_tokens,
      ):
        n1a = len(raw)
        n1b = sum((t in special_tokens) for t in encoded_desugared)
        n1 = n1a + n1b
        n2 = len(encoded_desugared)
        assert n1 == n2
        rval = []
        raw_iter = iter(raw)
        for token in encoded_desugared:
          if token in special_tokens:
            rval.append('')
          else:
            rval.append(next(raw_iter))
        assert len(rval) == n1
        return rval

      desugared_tokens_maybe_corrected_raw = _get_raw_desugared_tokens(
          pre_tokens_maybe_corrected_raw,
          encoded_desugared_tokens_maybe_corrected,
          [indent_token, dedent_token]
      )

      encoded_match_pairs = pre_pseudo_parser.encoded_match_pairs + [
          (indent_token, dedent_token)
      ]
      encoded_splitters = [newline_token]
      encoded_un_splittable_match_pairs = (
          pre_pseudo_parser.encoded_un_splittable_match_pairs)

      encoded_parsed_unsplit = stack_based_bracket_match(
          encoded_desugared_tokens_maybe_corrected,
          encoded_match_pairs,
          error_tolerant=True)

      assert valid_parse_ranges(encoded_parsed_unsplit,
                                len(encoded_desugared_tokens_maybe_corrected))

      encoded_parsed_split = split_parse(
          encoded_parsed_unsplit, encoded_desugared_tokens_maybe_corrected,
          encoded_splitters, encoded_un_splittable_match_pairs,
          (indent_token, dedent_token))

      assert valid_parse_ranges(encoded_parsed_split,
                                len(encoded_desugared_tokens_maybe_corrected))

      return (encoded_parsed_split,
              desugared_tokens_maybe_corrected,
              desugared_tokens_types_maybe_corrected,
              desugared_tokens_maybe_corrected_raw,
              (n_pre, n_post))

    else:

      pseudo_parser = PseudoParser.preset(language)
      (pseudo_parsed,
       tokens_maybe_corrected,
       token_types_maybe_corrected,
       tokens_maybe_corrected_raw,
       (n_pre, n_post)) = pseudo_parser.pseudo_parse(code)

      return (
          pseudo_parsed,
          tokens_maybe_corrected,
          token_types_maybe_corrected,
          tokens_maybe_corrected_raw,
          (n_pre, n_post),
          )

