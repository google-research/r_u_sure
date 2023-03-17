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

"""Small helper functions for working with SequenceNodes."""

import enum
from typing import Iterable, Union

from r_u_sure.tree_structure import sequence_nodes

SequenceNode = sequence_nodes.SequenceNode
NestedNodePath = sequence_nodes.NestedNodePath


def compute_depth(node_sequence: list[SequenceNode]) -> int:
  """Computes the depth of a node sequence.

  Args:
    node_sequence: Sequence of nodes.

  Returns:
    The depth of the sequence, equal to 1 + the number of nested GroupNodes. A
    sequence without group nodes has depth 1.
  """
  depth = 1
  for node in node_sequence:
    if isinstance(node, sequence_nodes.GroupNode):
      depth = max(depth, 1 + compute_depth(node.children))
  return depth


def render_text_contents(node: Union[SequenceNode, list[SequenceNode]]) -> str:
  """Recursively renders all text contained in a node and its descendants.

  Args:
    node: The node or list of nodes to render.

  Returns:
    Concatenated text contents of each descendant of the node, including both
    TextTokenNode instances and TextDecorationNode instances.
  """
  if isinstance(node, list):
    return "".join(render_text_contents(child) for child in node)
  elif isinstance(
      node, (sequence_nodes.TextDecorationNode, sequence_nodes.TextTokenNode)):
    return node.text_contents
  elif isinstance(node, sequence_nodes.GroupNode):
    return "".join(render_text_contents(child) for child in node.children)
  else:
    return ""


def render_debug(node: Union[SequenceNode, list[SequenceNode]]) -> str:
  """Renders a nested view of a node and its contents, for debugging and tests.

  Args:
    node: The node or list of nodes to render.

  Returns:
    String representation of the node and all of its children. Each line
    corresponds to a single node, and has information about that node's type
    and text contents. (See `sequence_node_helpers_test.py` for example output.)
  """
  if isinstance(node, list):
    debug_lines = []
    for child in node:
      _, cur_debug_lines = _render_debug_helper(child)
      debug_lines.extend(cur_debug_lines)
  else:
    _, debug_lines = _render_debug_helper(node)
  return "\n".join(debug_lines)


def _render_debug_helper(node: SequenceNode) -> tuple[str, list[str]]:
  """Helper function for `render_debug`.

  Args:
    node: Node to render.

  Returns:
    Tuple containing text contents of node along with list of debug lines.
  """
  if isinstance(node, sequence_nodes.TextDecorationNode):
    return (
        node.text_contents,
        ["DEC: " + _summarize_contents(node.text_contents)],
    )

  elif isinstance(node, sequence_nodes.TextTokenNode):
    header = "TOK: " if node.match_type is None else f"TOK({node.match_type}): "
    return (
        node.text_contents,
        [header + _summarize_contents(node.text_contents)],
    )

  elif isinstance(node, sequence_nodes.GroupNode):
    texts = []
    debugviews = []

    for child in node.children:
      text, debugview = _render_debug_helper(child)
      texts.append(text)
      debugviews.extend(debugview)

    header = ("GROUP: "
              if node.match_type is None else f"GROUP({node.match_type}): ")
    combined_text = "".join(texts)
    combined_debug = ([header + _summarize_contents(combined_text)] +
                      [f"  {debugline}" for debugline in debugviews])
    return combined_text, combined_debug

  else:
    return ("", [str(node)])


def _summarize_contents(contents: str) -> str:
  if len(contents) > 80:
    return repr(contents[:77]) + "..."
  else:
    return repr(contents)


class WalkTag(enum.Enum):
  """Enum of tags for nodes visited during a walk."""
  LEAF = enum.auto()
  GROUP_BEFORE_CHILDREN = enum.auto()
  GROUP_AFTER_CHILDREN = enum.auto()


def walk_with_paths(
    node_sequence: list[SequenceNode],
    visit_groups_before: bool = False,
    visit_groups_after: bool = False,
    prefix: NestedNodePath = NestedNodePath.root()
) -> Iterable[tuple[SequenceNode, NestedNodePath, WalkTag]]:
  """Recursively walks a nested sequence, emitting nodes and their paths.

  By default, only outputs leaf nodes. However, can also be configured to output
  group nodes, either before the children, after the children, or both. When
  both are used, the tag output can be used to determine whether we are starting
  or ending a group.

  Args:
    node_sequence: Sequence of nodes to walk over.
    visit_groups_before: Whether to output group nodes before their children.
    visit_groups_after: Whether to output group nodes after their children.
    prefix: Prefix to prepend to every output path.

  Yields:
    Tuples (node, path, tag) of nodes, paths to those nodes, and tags that
    identify whether group nodes are being emitted before or after children.
    Tag can be ignored unless both visit_groups_before and visit_groups_after
    are True.
  """
  for i, node in enumerate(node_sequence):
    path = prefix.extended_with(i)
    if isinstance(node, sequence_nodes.GroupNode):
      if visit_groups_before:
        yield node, path, WalkTag.GROUP_BEFORE_CHILDREN

      yield from walk_with_paths(
          node.children,
          visit_groups_before=visit_groups_before,
          visit_groups_after=visit_groups_after,
          prefix=path)

      if visit_groups_after:
        yield node, path, WalkTag.GROUP_AFTER_CHILDREN

    else:
      yield node, path, WalkTag.LEAF
