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

"""Numba-compatible versions of sequence nodes."""

import enum
from typing import Any, NamedTuple, Optional

import numba

from r_u_sure.numba_helpers import numba_type_util
from r_u_sure.numba_helpers import register_enum_hash  # pylint: disable=unused-import

from r_u_sure.tree_structure import sequence_nodes


class PackedSequenceNodeCategory(enum.Enum):
  """Categories of packed nodes."""

  TEXT_DECORATION_NODE = enum.auto()
  TEXT_TOKEN_NODE = enum.auto()
  REGION_START_NODE = enum.auto()
  REGION_END_NODE = enum.auto()
  EARLY_EXIT_NODE = enum.auto()
  GROUP_NODE = enum.auto()

  # INVALID is intended to be used as a sentinel value by logic dealing with
  # node categories, to make its use in Numba easier. It should NOT appear in
  # an actual packed sequence node ID.
  INVALID = enum.auto()

  # Numba-compatible __hash__ implementation.
  __hash__ = register_enum_hash.jitable_enum_hash


ROOT_SEQUENCE_PREORDER_INDEX = -1
NO_PREORDER_INDEX = -2


class PackedSequenceNodeID(NamedTuple):
  """Identifier for a node relative to its storage.

  A node is represented by a category and an index; the storage maintains
  separate typed lists for each category.

  Attributes:
    preorder_index: Index of this node in a pre-order traversal of the tree.
      Group nodes are immediately followed by their children in the pre-order
      traversal.
    category: Type of node that this is.
    index_in_category: The index into the category-specific storage that has the
      data for this node, if applicable. -1 if this node does not have any data
      that needs to be stored.
  """

  preorder_index: int
  category: PackedSequenceNodeCategory
  index_in_category: int


INVALID_NODE_ID = PackedSequenceNodeID(
    preorder_index=NO_PREORDER_INDEX,
    category=PackedSequenceNodeCategory.INVALID,
    index_in_category=-1,
)

PACKED_SEQUENCE_NODE_ID_NUMBA_TYPE = numba.typeof(INVALID_NODE_ID)


class PositionInParent(NamedTuple):
  """Information about the position of this node in its parent.

  Attributes:
    parent_preorder_index: Preorder index of the parent of a given node, or -1
      if this node is a node in the root sequence.
    index_of_child_in_parent: Index of this child in its parent GroupNode (or in
      the root sequence).
  """

  parent_preorder_index: int
  index_of_child_in_parent: int


POSITION_IN_PARENT_NUMBA_TYPE = numba.typeof(
    PositionInParent(parent_preorder_index=-1, index_of_child_in_parent=0)
)


class PackedTextDecorationNode(NamedTuple):
  """Packed version of sequence_nodes.TextDecorationNode."""

  text_contents: str


class PackedTextTokenNode(NamedTuple):
  """Packed version of sequence_nodes.TextDecorationNode."""

  text_contents: str
  match_type: Any = None


class PackedGroupNode(NamedTuple):
  """Packed version of sequence_nodes.GroupNode."""

  children_ids: list[PackedSequenceNodeID]
  match_type: Any = None


class PackedSequenceNodeStorage(NamedTuple):
  """Data structure holding all of the nodes for a given sequence.

  The main purpose of this data structure is to remove two features that
  numba doesn't support:
  - isinstance checks to determine which type of node we have
  - nodes that contain other nodes recursively.

  We do this by separating the nodes by category, then storing them in flat
  lists, and passing around references to those lists instead of nodes
  themselves. By checking the `category` of an ID, we can determine the type
  of the node without running `isinstance`, then look up its statically-typed
  contents by indexing into the appropriate array if required. Recursive calls
  can then just pass around the storage itself, using the ids inside a group
  node to restrict to a subset of children.

  Note that the node ID contains a bit of redundant information, in that you
  could look up the node ID in `preorder_traversal` given only the
  `preorder_index` field, but the root sequence and PackedGroupNode contain all
  of the fields. This is intentional to simplify processing in the common case
  where you want to iterate over the children of a particular sequence of nodes.

  Attributes:
    root_sequence: Sequence of IDs in the root sequence.
    preorder_traversal: Sequence of IDs in a preorder traversal of the tree.
    text_decoration_nodes: Storage for text decoration nodes. Given an ID whose
      category is TEXT_DECORATION_NODE, the value at
      `text_decoration_nodes[index_in_category]` will be the content of that
      decoration node.
    text_token_nodes: Storage for text token nodes, as above.
    group_nodes: Storage for group nodes, as above.
    parents_and_offsets_from_preorder: List that maps from a node's preorder
      index to its position in its parent. Used to "unwind" the path to a given
      node from its ID.
    level_from_preorder: List that maps from a node's preorder index to its
      level. Nodes in the root sequence have level 0, and children of group
      nodes have level 1 more than their parent.
    depth: The total depth of the tree (equal to one more than the maximum level
      in level_from_preorder.)
  """

  root_sequence: list[PackedSequenceNodeID]
  preorder_traversal: list[PackedSequenceNodeID]
  text_decoration_nodes: list[PackedTextDecorationNode]
  text_token_nodes: list[PackedTextTokenNode]
  group_nodes: list[PackedGroupNode]
  parents_and_offsets_from_preorder: list[PositionInParent]
  level_from_preorder: list[int]
  depth: int

  def node_id_from_node_path(
      self, node_path: sequence_nodes.NestedNodePath
  ) -> PackedSequenceNodeID:
    """Looks up a node ID from a path."""
    if not node_path.path:
      raise ValueError("Cannot look up a node for the empty path.")
    the_sequence = self.subsequence_from_node_path(
        sequence_nodes.NestedNodePath(path=node_path.path[:-1])
    )
    return the_sequence[node_path.path[-1]]

  def subsequence_from_node_path(
      self, sequence_path: sequence_nodes.NestedNodePath
  ) -> list[PackedSequenceNodeID]:
    """Looks up a node ID sequence from a path."""
    the_sequence = self.root_sequence
    for position in sequence_path.path:
      if position < 0 or position >= len(the_sequence):
        raise ValueError(
            f"Invalid path position {position} for sequence of length"
            f" {len(the_sequence)}"
        )
      the_id = the_sequence[position]
      if the_id.category == PackedSequenceNodeCategory.GROUP_NODE:
        the_sequence = self.group_nodes[the_id.index_in_category].children_ids
      else:
        raise ValueError(
            f"Encountered a non-group-node {the_id} along a subsequence path"
            f" {sequence_path}"
        )
    return the_sequence

  def path_from_preorder_index(
      self, preorder_index: int
  ) -> sequence_nodes.NestedNodePath:
    """Looks up the path to a particular packed node."""
    reversed_path = []
    while preorder_index != -1:
      parent_info = self.parents_and_offsets_from_preorder[preorder_index]
      reversed_path.append(parent_info.index_of_child_in_parent)
      preorder_index = parent_info.parent_preorder_index
    return sequence_nodes.NestedNodePath(path=tuple(reversed(reversed_path)))


def pack(
    nodes: list[sequence_nodes.SequenceNode],
    as_numba_typed: bool = False,
    match_type_numba_type: Optional[numba.types.Type] = None,
) -> PackedSequenceNodeStorage:
  """Packs a node sequence into an efficient typed storage.

  Args:
    nodes: List of sequence nodes to pack.
    as_numba_typed: Whether to produce outputs as `numba.typed.List` objects,
      for passing into a Numba function.
    match_type_numba_type: The Numba type for the `match_type` field on token
      and group nodes. Only required if `as_numba_typed == True`, in which case
      all `match_type` values will be cast to this numba type during conversion
      (with special care to support optional types).

  Returns:
    A packed version of `nodes`, using either Python lists or numba.typed.Lists.
  """
  # pylint: disable=invalid-name
  if as_numba_typed:
    assert match_type_numba_type is not None
    fake_match_type = numba_type_util.PretendType(match_type_numba_type)
    result = PackedSequenceNodeStorage(
        text_decoration_nodes=numba.typed.List.empty_list(
            numba.typeof(PackedTextDecorationNode("foo"))
        ),
        text_token_nodes=numba.typed.List.empty_list(
            numba.typeof(PackedTextTokenNode("foo", fake_match_type))
        ),
        group_nodes=numba.typed.List.empty_list(
            numba.typeof(
                PackedGroupNode(
                    children_ids=numba.typed.List.empty_list(
                        PACKED_SEQUENCE_NODE_ID_NUMBA_TYPE
                    ),
                    match_type=fake_match_type,
                )
            )
        ),
        root_sequence=numba.typed.List.empty_list(
            PACKED_SEQUENCE_NODE_ID_NUMBA_TYPE
        ),
        preorder_traversal=numba.typed.List.empty_list(
            PACKED_SEQUENCE_NODE_ID_NUMBA_TYPE
        ),
        parents_and_offsets_from_preorder=numba.typed.List.empty_list(
            POSITION_IN_PARENT_NUMBA_TYPE
        ),
        level_from_preorder=numba.typed.List.empty_list(numba.int64),
        depth=-1,  # temporary placeholder value
    )

  else:
    result = PackedSequenceNodeStorage(
        text_decoration_nodes=[],
        text_token_nodes=[],
        group_nodes=[],
        root_sequence=[],
        preorder_traversal=[],
        parents_and_offsets_from_preorder=[],
        level_from_preorder=[],
        depth=-1,  # temporary placeholder value
    )
  max_level = 0

  def preorder_traversal(
      node: sequence_nodes.SequenceNode,
      position_in_parent: PositionInParent,
      level: int,
  ) -> PackedSequenceNodeID:
    """Inserts a node and all of its children in a pre-order traversal."""
    nonlocal max_level
    max_level = max(max_level, level)
    next_preorder_index = len(result.preorder_traversal)

    if isinstance(node, sequence_nodes.TextDecorationNode):
      next_index_in_category = len(result.text_decoration_nodes)
      node_id = PackedSequenceNodeID(
          preorder_index=next_preorder_index,
          category=PackedSequenceNodeCategory.TEXT_DECORATION_NODE,
          index_in_category=next_index_in_category,
      )
      result.text_decoration_nodes.append(
          PackedTextDecorationNode(node.text_contents)
      )
      result.preorder_traversal.append(node_id)
      result.parents_and_offsets_from_preorder.append(position_in_parent)
      result.level_from_preorder.append(level)
      return node_id

    elif isinstance(node, sequence_nodes.TextTokenNode):
      next_index_in_category = len(result.text_token_nodes)
      node_id = PackedSequenceNodeID(
          preorder_index=next_preorder_index,
          category=PackedSequenceNodeCategory.TEXT_TOKEN_NODE,
          index_in_category=next_index_in_category,
      )
      result.text_token_nodes.append(
          PackedTextTokenNode(node.text_contents, node.match_type)
      )
      result.preorder_traversal.append(node_id)
      result.parents_and_offsets_from_preorder.append(position_in_parent)
      result.level_from_preorder.append(level)
      return node_id

    elif isinstance(node, sequence_nodes.GroupNode):
      next_index_in_category = len(result.group_nodes)
      if as_numba_typed:
        children_ids = numba.typed.List.empty_list(
            PACKED_SEQUENCE_NODE_ID_NUMBA_TYPE
        )
      else:
        children_ids = []
      node_id = PackedSequenceNodeID(
          preorder_index=next_preorder_index,
          category=PackedSequenceNodeCategory.GROUP_NODE,
          index_in_category=next_index_in_category,
      )
      the_group = PackedGroupNode(
          children_ids=children_ids, match_type=node.match_type
      )
      result.group_nodes.append(the_group)
      result.preorder_traversal.append(node_id)
      result.parents_and_offsets_from_preorder.append(position_in_parent)
      result.level_from_preorder.append(level)

      # Process children
      for i, child in enumerate(node.children):
        pos_in_parent = PositionInParent(
            parent_preorder_index=node_id.preorder_index,
            index_of_child_in_parent=i,
        )
        child_id = preorder_traversal(child, pos_in_parent, level=level + 1)
        the_group.children_ids.append(child_id)

      return node_id

    elif isinstance(node, sequence_nodes.RegionStartNode):
      node_id = PackedSequenceNodeID(
          preorder_index=next_preorder_index,
          category=PackedSequenceNodeCategory.REGION_START_NODE,
          index_in_category=-1,
      )
      result.preorder_traversal.append(node_id)
      result.parents_and_offsets_from_preorder.append(position_in_parent)
      result.level_from_preorder.append(level)
      return node_id

    elif isinstance(node, sequence_nodes.RegionEndNode):
      node_id = PackedSequenceNodeID(
          preorder_index=next_preorder_index,
          category=PackedSequenceNodeCategory.REGION_END_NODE,
          index_in_category=-1,
      )
      result.preorder_traversal.append(node_id)
      result.parents_and_offsets_from_preorder.append(position_in_parent)
      result.level_from_preorder.append(level)
      return node_id

    elif isinstance(node, sequence_nodes.EarlyExitNode):
      node_id = PackedSequenceNodeID(
          preorder_index=next_preorder_index,
          category=PackedSequenceNodeCategory.EARLY_EXIT_NODE,
          index_in_category=-1,
      )
      result.preorder_traversal.append(node_id)
      result.parents_and_offsets_from_preorder.append(position_in_parent)
      result.level_from_preorder.append(level)
      return node_id

    else:
      raise ValueError(f"Invalid node {node}")

  for i, node in enumerate(nodes):
    pos_in_parent = PositionInParent(
        parent_preorder_index=-1, index_of_child_in_parent=i
    )
    node_id = preorder_traversal(node, pos_in_parent, level=0)
    result.root_sequence.append(node_id)

  return PackedSequenceNodeStorage(
      text_decoration_nodes=result.text_decoration_nodes,
      text_token_nodes=result.text_token_nodes,
      group_nodes=result.group_nodes,
      root_sequence=result.root_sequence,
      preorder_traversal=result.preorder_traversal,
      parents_and_offsets_from_preorder=result.parents_and_offsets_from_preorder,
      level_from_preorder=result.level_from_preorder,
      depth=max_level + 1,
  )


def unpack(
    storage: PackedSequenceNodeStorage,
) -> list[sequence_nodes.SequenceNode]:
  """Unpacks a node sequence from storage."""

  def build(node_id: PackedSequenceNodeID) -> sequence_nodes.SequenceNode:
    if node_id.category == PackedSequenceNodeCategory.TEXT_DECORATION_NODE:
      return sequence_nodes.TextDecorationNode(
          text_contents=storage.text_decoration_nodes[
              node_id.index_in_category
          ].text_contents
      )
    elif node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
      stored_node = storage.text_token_nodes[node_id.index_in_category]
      return sequence_nodes.TextTokenNode(
          text_contents=stored_node.text_contents,
          match_type=stored_node.match_type,
      )
    elif node_id.category == PackedSequenceNodeCategory.GROUP_NODE:
      stored_node = storage.group_nodes[node_id.index_in_category]
      return sequence_nodes.GroupNode(
          children=[build(child_id) for child_id in stored_node.children_ids],
          match_type=stored_node.match_type,
      )
    elif node_id.category == PackedSequenceNodeCategory.REGION_START_NODE:
      return sequence_nodes.RegionStartNode()
    elif node_id.category == PackedSequenceNodeCategory.REGION_END_NODE:
      return sequence_nodes.RegionEndNode()
    elif node_id.category == PackedSequenceNodeCategory.EARLY_EXIT_NODE:
      return sequence_nodes.EarlyExitNode()
    else:
      raise ValueError(f"Invalid node id {node_id}")

  return [build(node_id) for node_id in storage.root_sequence]


MAX_NESTED_NODE_DEPTH = 10
INVALID_PATH_ENTRY = -100


def render_text_contents_of_node(
    node_id: PackedSequenceNodeID, storage: PackedSequenceNodeStorage
) -> str:
  """Recursively renders all text contained in a node and its descendants.

  Args:
    node_id: The node ID to render.
    storage: Packed storage for the node IDs.

  Returns:
    Concatenated text contents of each descendant of the node, including both
    TextTokenNode instances and TextDecorationNode instances.
  """
  if node_id.category == PackedSequenceNodeCategory.TEXT_DECORATION_NODE:
    return storage.text_decoration_nodes[
        node_id.index_in_category
    ].text_contents
  elif node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
    return storage.text_token_nodes[node_id.index_in_category].text_contents
  elif node_id.category == PackedSequenceNodeCategory.GROUP_NODE:
    group_node = storage.group_nodes[node_id.index_in_category]
    return render_text_contents_of_list(group_node.children_ids, storage)
  elif not isinstance(node_id.category, PackedSequenceNodeCategory):
    raise ValueError(f"Invalid node id category {node_id.category}")
  else:
    return ""


def render_text_contents_of_list(
    node_ids: list[PackedSequenceNodeID], storage: PackedSequenceNodeStorage
) -> str:
  """Recursively renders all text contained in a node and its descendants.

  Args:
    node_ids: The list of node IDs to render.
    storage: Packed storage for the node IDs.

  Returns:
    Concatenated text contents of each descendant of the node, including both
    TextTokenNode instances and TextDecorationNode instances.
  """
  parts = []
  for node_id in node_ids:
    parts.append(render_text_contents_of_node(node_id, storage))
  return "".join(parts)


class InOrderTraversalCategory(enum.Enum):
  """Categories of elements in an in-order traversal."""

  LEAF = enum.auto()
  BEFORE_GROUP = enum.auto()
  AFTER_GROUP = enum.auto()

  # Numba-compatible __hash__ implementation.
  __hash__ = register_enum_hash.jitable_enum_hash


class InOrderTraversalItem(NamedTuple):
  """Element of an in-order traversal.

  Attributes:
    category: Type of item that this is. Group nodes are visited twice; once
      before and once after the children.
    node_id: ID of the node we are visiting.
  """

  category: InOrderTraversalCategory
  node_id: PackedSequenceNodeID


IN_ORDER_TRAVERSAL_ITEM_NUMBA_TYPE = numba.typeof(
    InOrderTraversalItem(
        category=InOrderTraversalCategory.LEAF,
        node_id=INVALID_NODE_ID,
    )
)


def _make_recursive_in_order_traversal(jit: bool = False):
  """Helper function to build a recursive in order traversal function."""

  # This is necessary to get proper recursion support under numba.
  def do_in_order_traversal(
      storage: PackedSequenceNodeStorage,
      node_ids: list[PackedSequenceNodeID],
      dest: list[InOrderTraversalItem],
      strip_decorations: bool,
  ) -> None:
    """Helper function to build an in order traversal."""
    for node_id in node_ids:
      if node_id.category == PackedSequenceNodeCategory.GROUP_NODE:
        node = storage.group_nodes[node_id.index_in_category]
        dest.append(
            InOrderTraversalItem(InOrderTraversalCategory.BEFORE_GROUP, node_id)
        )
        do_in_order_traversal(
            storage=storage,
            node_ids=node.children_ids,
            dest=dest,
            strip_decorations=strip_decorations,
        )
        dest.append(
            InOrderTraversalItem(InOrderTraversalCategory.AFTER_GROUP, node_id)
        )
      else:
        if (
            strip_decorations
            and node_id.category
            == PackedSequenceNodeCategory.TEXT_DECORATION_NODE
        ):
          # Skip this node.
          pass
        else:
          dest.append(
              InOrderTraversalItem(InOrderTraversalCategory.LEAF, node_id)
          )

  if jit:
    do_in_order_traversal = numba.njit(do_in_order_traversal)

  return do_in_order_traversal


def in_order_traversal(
    storage: PackedSequenceNodeStorage, strip_decorations: bool
) -> list[InOrderTraversalItem]:
  """Constructs an in order traversal.

  Args:
    storage: Storage to construct a traversal for.
    strip_decorations: Whether to remove decoration nodes.

  Returns:
    List of traversal items, visiting each group node twice and each leaf node
      once.
  """
  result = []
  _make_recursive_in_order_traversal(jit=False)(
      storage=storage,
      node_ids=storage.root_sequence,
      dest=result,
      strip_decorations=strip_decorations,
  )
  return result


@numba.extending.overload(in_order_traversal)
def _in_order_traversal_numba_overload(storage, strip_decorations):
  """Numba overload for in_order_traversal."""
  del storage, strip_decorations
  fn = _make_recursive_in_order_traversal(jit=True)

  def _impl(storage, strip_decorations):
    result = numba.typed.List.empty_list(IN_ORDER_TRAVERSAL_ITEM_NUMBA_TYPE)
    fn(
        storage=storage,
        node_ids=storage.root_sequence,
        dest=result,
        strip_decorations=strip_decorations,
    )
    return result

  return _impl
