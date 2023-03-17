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

"""High-level interface for applying R-U-SURE to the call sequence extraction task."""


import functools
import html
import re
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Sequence
import numba
import numpy as np
from r_u_sure.decision_diagrams import consistent_path_dual_solver
from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.decision_diagrams import packed_dags
from r_u_sure.edit_distance_utility import constraint_dags
from r_u_sure.edit_distance_utility import edit_dags
from r_u_sure.edit_distance_utility import part_selection_dags
from r_u_sure.edit_distance_utility import region_decisions
from r_u_sure.parsing import python_builtin_tokens
from r_u_sure.tree_structure import packed_sequence_nodes
from r_u_sure.tree_structure import sequence_node_helpers
from r_u_sure.tree_structure import sequence_nodes
from r_u_sure.tree_structure import transforms
from r_u_sure.wrappers import parser_tools
from r_u_sure.wrappers import wrapper_base


PackedSequenceNodeStorage = packed_sequence_nodes.PackedSequenceNodeStorage
PackedSequenceNodeCategory = packed_sequence_nodes.PackedSequenceNodeCategory
DualDecompositionDiagramSystem = (
    consistent_path_dual_solver.DualDecompositionDiagramSystem
)

DecisionKey = Any
DecisionValue = Any

ContextInfo = Any


def extract_context_tokens(
    context_and_prototype: list[sequence_nodes.SequenceNode],
    prediction_location: int,
) -> set[str]:
  """Extracts a set of tokens that are seen in the context."""
  seen_tokens = set()
  position = 0
  for node, _, _ in sequence_node_helpers.walk_with_paths(
      context_and_prototype
  ):
    if isinstance(node, sequence_nodes.TextTokenNode):
      seen_tokens.add(node.text_contents)
    if isinstance(
        node, (sequence_nodes.TextTokenNode, sequence_nodes.TextDecorationNode)
    ):
      position += len(node.text_contents)
      if position >= prediction_location:
        break
  return seen_tokens


class ApiCallSequenceWrapper(wrapper_base.HighLevelUtilityWrapper):
  """Wrapper object that tries to identify API call sequences."""

  def _is_paren(self, node: sequence_nodes.SequenceNode) -> bool:
    """Checks if a pseudoparser node is an open parenthesis."""
    return (
        isinstance(node, sequence_nodes.GroupNode)
        and node.match_type == "MATCH"
        and len(node.children) == 3
        and node.children[0] == sequence_nodes.TextTokenNode("(", "MATCH_LEFT")
    )

  def _extract_all_from(
      self,
      node_sequence: Sequence[sequence_nodes.SequenceNode],
      expected_tokens: set[str],
  ) -> Iterator[list[sequence_nodes.SequenceNode]]:
    """Heuristically extracts call-like expressions from a sequence of nodes.

    Call-like expressions are identified by taking a consecutive sequence of
    tokens at the same depth (in the pseudoparser output, e.g. not containing
    any brackets), and searching for the first group of parentheses. If found,
    we take the content of those parentheses as optional args, and all previous
    tokens as possible call tokens to extract.

    We insert region start/end nodes so that regions can start
     - at the beginning of the subtree,
     - before any period,
     - after any whitespace
    and end
     - after the opening parenthesis
     - after the closing parenthesis

    Note that this heuristic sometimes includes other call-like expressions,
    e.g. function definitions.

    Args:
      node_sequence: Sequence of nodes to extract call expressions from.
      expected_tokens: Set of tokens that are "expected" in the sense that they
        appear in the context, and thus are worth less if predicted
        successfully.

    Yields:
      Lists of nodes, each corresponding to a single call-like expression.
    """
    # Look for the first group node. If this is a call, that group node will
    # be a pair of matched parentheses.
    started_candidate = False
    ended_candidate = False
    candidate = []
    after_candidate = []
    for node in node_sequence:
      if not ended_candidate:
        if not isinstance(node, sequence_nodes.TextDecorationNode):
          started_candidate = True
        if started_candidate:
          candidate.append(node)
        if isinstance(node, sequence_nodes.GroupNode):
          ended_candidate = True
      else:  # ended_candidate
        after_candidate.append(node)

    # Now process the possible candidate
    if (
        len(candidate) >= 2
        and candidate[0] != sequence_nodes.TextTokenNode("def", "CONTENT_LEAF")
        and (
            candidate[0]
            != sequence_nodes.TextTokenNode("class", "CONTENT_LEAF")
        )
        and self._is_paren(candidate[-1])
        and isinstance(candidate[-2], sequence_nodes.TextTokenNode)
    ):
      # Build it in reverse
      reverse_outs = []
      reverse_outs.append(sequence_nodes.RegionEndNode())
      reverse_outs.append(sequence_nodes.TextTokenNode(")", "END_CALL"))
      args_text = sequence_node_helpers.render_text_contents(
          candidate[-1].children[1]  # pytype: disable=attribute-error
      )
      if args_text:
        reverse_outs.append(sequence_nodes.TextTokenNode(args_text, "ARGS"))
      else:
        reverse_outs.append(sequence_nodes.TextTokenNode("", "EMPTY_ARGS"))
      reverse_outs.append(sequence_nodes.RegionEndNode())
      reverse_outs.append(sequence_nodes.TextTokenNode("(", "START_CALL"))
      for node in candidate[::-1]:
        if isinstance(node, sequence_nodes.TextTokenNode):
          if node == sequence_nodes.TextTokenNode(".", "CONTENT_LEAF"):
            reverse_outs.append(
                sequence_nodes.TextTokenNode(".", "ATTRIBUTE_DOT")
            )
            reverse_outs.append(sequence_nodes.RegionStartNode())
          elif node.text_contents in expected_tokens:
            reverse_outs.append(
                sequence_nodes.TextTokenNode(node.text_contents, "EXPECTED")
            )
          else:
            reverse_outs.append(
                sequence_nodes.TextTokenNode(node.text_contents, "NOVEL")
            )

        elif isinstance(node, sequence_nodes.TextDecorationNode):
          if re.fullmatch(r"\s+", node.text_contents):
            reverse_outs.append(sequence_nodes.RegionStartNode())
          reverse_outs.append(node)
      reverse_outs.append(sequence_nodes.RegionStartNode())
      yield reverse_outs[::-1]
    else:
      # Didn't pass the check. Process the nodes recursively.
      for node in candidate:  # the ones we didn't process already
        if isinstance(node, sequence_nodes.GroupNode):
          yield from self._extract_all_from(node.children, expected_tokens)

    for node in after_candidate:
      if isinstance(node, sequence_nodes.GroupNode):
        yield from self._extract_all_from(node.children, expected_tokens)

  def _make_callseq_tree(
      self,
      context_and_suggestion: list[sequence_nodes.SequenceNode],
      prediction_location: int,
  ) -> list[sequence_nodes.SequenceNode]:
    """Constructs a sequence of extracted calls from an original full program.

    Args:
      context_and_suggestion: Parsed representation of context and suggestion,
        concatenated together.
      prediction_location: Offset into `context_and_suggestion` representing the
        end of the context and the start of the suggestion. We will adjust this
        to start at the previous newline, to avoid truncating partial calls.

    Returns:
      A new, depth-2 node sequence containing calls extracted from the
      original prototype suggestion. Each call is in its own group node, and
      no region start/end nodes are allowed.
    """
    # Adjust prediction location to previous newline
    raw_text = sequence_node_helpers.render_text_contents(
        context_and_suggestion
    )
    adjusted_prediction_location = raw_text[:prediction_location].rfind("\n")
    if adjusted_prediction_location == -1:
      adjusted_prediction_location = 0
    expected_tokens = extract_context_tokens(
        context_and_suggestion, prediction_location
    ).union(python_builtin_tokens.PYTHON_BUILTIN_TOKENS)
    result = []
    truncated_prototype = transforms.truncate_prefix_at_offset(
        context_and_suggestion, adjusted_prediction_location
    )
    for seq in self._extract_all_from(truncated_prototype, expected_tokens):
      result.append(
          sequence_nodes.GroupNode(children=seq, match_type="API_CALL")
      )
      result.append(sequence_nodes.TextDecorationNode("\n"))
    return result

  def __init__(
      self,
      effective_precision: float = 0.3,
      use_numba: bool = True,
      rewrite_states: bool = True,
  ):
    """Constructs a UncertaintyRegionsWrapper.

    See `derive_uncertainty_region_costs` for details on the
    meaning of arguments.

    Args:
      effective_precision: The probability level that is high enough to not show
        an extracted region.
      use_numba: Whether to use Numba to accelerate computation.
      rewrite_states: If using Numba, whether to rewrite states to be integers,
        which is slightly faster.
    """

    @numba.extending.register_jitable
    def token_node_utility_fn(
        token: packed_sequence_nodes.PackedTextTokenNode,
    ) -> part_selection_dags.UtilitiesAndCostsForNode:
      if token.match_type == "ARGS":
        util = 1.0
      elif token.match_type == "EMPTY_ARGS":
        util = 1.0
      elif token.match_type == "EXPECTED":
        util = 1.0
      elif token.match_type == "NOVEL":
        util = 10.0
      else:
        util = 0.0

      # Note: We want expected utility to be 0 if we are correct with
      # probability `effective_precision`. So we want
      # effective_precision * match_utility
      # = (1-effective_precision) * delete_cost
      return part_selection_dags.UtilitiesAndCostsForNode(
          match_utility=util * (1 - effective_precision),
          delete_cost=util * effective_precision,
      )

    utility_config = part_selection_dags.PartSelectionUtilityParameters(
        token_node_utility_fn=token_node_utility_fn,
    )
    self._utility_config = utility_config
    self._use_numba = use_numba
    if use_numba:
      self._pack_selection_dag, self._selection_dag_converter = (
          part_selection_dags.make_specialized_dag_packer(
              states_rewritten_as_integers=rewrite_states
          )
      )
      self._pack_constraint_dag, self._constraint_dag_converter = (
          constraint_dags.make_specialized_dag_packer(
              states_rewritten_as_integers=rewrite_states
          )
      )
      if rewrite_states:
        self._scratch_table = np.full((2**25,), -1, dtype=np.int32)
        self._prune_to_reachable = functools.partial(
            gated_state_dag.prune_unreachable_and_rewrite_states,
            scratch_table=self._scratch_table,
        )
      else:
        self._prune_to_reachable = gated_state_dag.prune_to_reachable_jit

    else:
      self._prune_to_reachable = gated_state_dag.prune_to_reachable
      self._pack_selection_dag = part_selection_dags.pack_dag
      self._selection_dag_converter = lambda dag: dag
      self._pack_constraint_dag = constraint_dags.pack_dag
      self._constraint_dag_converter = lambda dag: dag

    self._construct_selection_dag = (
        part_selection_dags.make_selection_dag_builder(
            self._utility_config, with_numba=use_numba
        )
    )
    self._construct_constraint_dag = (
        constraint_dags.make_constraint_dag_builder(with_numba=use_numba)
    )

  def process_prototype_and_context(
      self,
      context_and_prototype: list[sequence_nodes.SequenceNode],
      prediction_location: int,
  ) -> tuple[PackedSequenceNodeStorage, ContextInfo]:
    """See HighLevelUtilityWrapper.process_prototype_and_context."""
    tree = self._make_callseq_tree(context_and_prototype, prediction_location)
    packed_tree = parser_tools.pack_sequence_from_pseudoparser(
        tree, with_numba=self._use_numba
    )
    return packed_tree, {}

  def process_target(
      self,
      context_and_target: list[sequence_nodes.SequenceNode],
      prediction_location: int,
  ) -> PackedSequenceNodeStorage:
    """See HighLevelUtilityWrapper.process_prototype_and_context."""
    tree = self._make_callseq_tree(context_and_target, prediction_location)
    tree = transforms.strip_decision_nodes(tree)
    packed_tree = parser_tools.pack_sequence_from_pseudoparser(
        tree, with_numba=self._use_numba
    )
    return packed_tree

  def build_baseline_assignments(
      self,
      prototype: PackedSequenceNodeStorage,
      prototype_suggestion_as_target: PackedSequenceNodeStorage,
      context_info: ContextInfo,
      model_tokens_and_log_probs: list[tuple[str, float]],
  ) -> dict[str, dict[DecisionKey, DecisionValue]]:
    """See HighLevelUtilityWrapper.build_baseline_assignments."""
    del prototype_suggestion_as_target, context_info, model_tokens_and_log_probs
    # All function calls with entire LHS, potentially including variable names,
    # and also including arguments.
    all_calls_full = []
    # All function calls, but only starting at the most recent whitespace
    # character, and ending at the left opening parenthesis.
    all_calls_shorter = []
    # Like `all_calls_shorter`, but only including calls that contain at least
    # one identifier not yet seen in the context.
    novel_calls_shorter = []

    # We start by iterating through calls, and adding preorder indices of
    # region start/end nodes to the lists above, based on which of the
    # corresponding decisions should be True.
    for toplevel_node_id in prototype.root_sequence:
      if toplevel_node_id.category == PackedSequenceNodeCategory.GROUP_NODE:
        saw_novel = False

        toplevel_node = prototype.group_nodes[
            toplevel_node_id.index_in_category
        ]
        last_end_node = toplevel_node.children_ids[-1]

        assert (
            last_end_node.category == PackedSequenceNodeCategory.REGION_END_NODE
        )
        all_calls_full.append(last_end_node.preorder_index)

        assert (
            toplevel_node.children_ids[-2].category
            == PackedSequenceNodeCategory.TEXT_TOKEN_NODE
        )
        assert (
            toplevel_node.children_ids[-3].category
            == PackedSequenceNodeCategory.TEXT_TOKEN_NODE
        )
        assert (
            toplevel_node.children_ids[-4].category
            == PackedSequenceNodeCategory.REGION_END_NODE
        )

        current_call_shorter_end = toplevel_node.children_ids[-4].preorder_index

        assert (
            last_end_node.category == PackedSequenceNodeCategory.REGION_END_NODE
        )
        assert (
            last_end_node.category == PackedSequenceNodeCategory.REGION_END_NODE
        )

        prev_start = None
        current_call_shorter_start = None
        for child_id in toplevel_node.children_ids[-5::-1]:
          if child_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
            child = prototype.text_token_nodes[child_id.index_in_category]
            if child.match_type == "NOVEL":
              saw_novel = True
          elif (
              child_id.category
              == PackedSequenceNodeCategory.TEXT_DECORATION_NODE
          ):
            assert prev_start is not None
            current_call_shorter_start = prev_start
            break
          elif (
              child_id.category == PackedSequenceNodeCategory.REGION_START_NODE
          ):
            prev_start = child_id.preorder_index
        if current_call_shorter_start is None:
          assert prev_start is not None
          current_call_shorter_start = prev_start

        first_start_node = toplevel_node.children_ids[0]
        assert (
            first_start_node.category
            == PackedSequenceNodeCategory.REGION_START_NODE
        )
        all_calls_full.append(first_start_node.preorder_index)

        all_calls_shorter.append(current_call_shorter_start)
        all_calls_shorter.append(current_call_shorter_end)

        if saw_novel:
          novel_calls_shorter.append(current_call_shorter_start)
          novel_calls_shorter.append(current_call_shorter_end)

    # To build actual assignments based on these, we iterate through the
    # region start and end, and assign to false except for the ones we
    # flagged as true. Note: we have to assign "not applicable" for the ones
    # that can't be set to true.
    def _make_suggestion(preorder_indices):
      assignment = {}
      seen_indices = set()
      is_in_low_conf = False
      for node_id in prototype.preorder_traversal:
        if node_id.category == PackedSequenceNodeCategory.REGION_START_NODE:
          seen_indices.add(node_id.preorder_index)
          if node_id.preorder_index in preorder_indices:
            assert not is_in_low_conf
            val = region_decisions.DecisionValue.TRUE
            is_in_low_conf = True
          elif is_in_low_conf:
            val = region_decisions.DecisionValue.NOT_APPLICABLE
          else:
            val = region_decisions.DecisionValue.FALSE

          assignment[
              region_decisions.DecisionKey(
                  prototype_preorder_index=node_id.preorder_index,
                  category=(
                      region_decisions.DecisionCategory.REGION_SHOULD_START
                  ),
              )
          ] = val
        elif node_id.category == PackedSequenceNodeCategory.REGION_END_NODE:
          seen_indices.add(node_id.preorder_index)
          if node_id.preorder_index in preorder_indices:
            assert is_in_low_conf
            val = region_decisions.DecisionValue.TRUE
            is_in_low_conf = False
          elif is_in_low_conf:
            val = region_decisions.DecisionValue.FALSE
          else:
            val = region_decisions.DecisionValue.NOT_APPLICABLE

          assignment[
              region_decisions.DecisionKey(
                  prototype_preorder_index=node_id.preorder_index,
                  category=(
                      region_decisions.DecisionCategory.REGION_SHOULD_END
                  ),
              )
          ] = val

      remaining = preorder_indices - seen_indices
      assert not remaining
      return assignment

    return {
        "all_calls_full": _make_suggestion(set(all_calls_full)),
        "all_calls_shorter": _make_suggestion(set(all_calls_shorter)),
        "novel_calls_shorter": _make_suggestion(set(novel_calls_shorter)),
    }

  def build_system(
      self,
      prototype: PackedSequenceNodeStorage,
      context_info: ContextInfo,
      targets: list[PackedSequenceNodeStorage],
      target_utility_scale_factors: Optional[list[float]] = None,
  ) -> DualDecompositionDiagramSystem:
    """See HighLevelUtilityWrapper.build_system."""
    del context_info

    if target_utility_scale_factors is None:
      uniform_weight = 1.0 / len(targets)
      target_utility_scale_factors = [uniform_weight for _ in targets]

    dags_and_conversion_data = []
    dag_update_proportions = []

    # Selection dags
    for target, scale_factor in zip(targets, target_utility_scale_factors):
      dag = self._construct_selection_dag(prototype=prototype, target=target)
      reachable_dag = self._prune_to_reachable(dag)
      packed_dag, conversion_data = self._pack_selection_dag(reachable_dag)
      packed_dag_scaled = packed_dags.scale_packed_dag_costs(
          packed_dag, scale_factor
      )
      dags_and_conversion_data.append((packed_dag_scaled, conversion_data))
      dag_update_proportions.append(scale_factor)

    # Constraint dag
    scale_factor = sum(target_utility_scale_factors)
    dag = self._construct_constraint_dag(prototype=prototype)
    reachable_dag = self._prune_to_reachable(dag)
    packed_dag, conversion_data = self._pack_constraint_dag(reachable_dag)
    packed_dag_scaled = packed_dags.scale_packed_dag_costs(
        packed_dag, scale_factor
    )
    dags_and_conversion_data.append((packed_dag_scaled, conversion_data))
    dag_update_proportions.append(scale_factor)

    system = consistent_path_dual_solver.make_system(
        dags_and_conversion_data,
        dag_update_proportions=np.array(dag_update_proportions),
    )
    return system

  def solution_info(
      self,
      prototype: PackedSequenceNodeStorage,
      evaluation_target: Optional[PackedSequenceNodeStorage],
      context_info: ContextInfo,
      system: Optional[DualDecompositionDiagramSystem],
      assignments: dict[
          region_decisions.DecisionKey,
          region_decisions.DecisionValue,
      ],
      sample_system: Optional[DualDecompositionDiagramSystem] = None,
  ) -> dict[Any, Any]:
    """See HighLevelUtilityWrapper.solution_info."""
    del context_info
    parts = []
    for region_info in region_decisions.extract_sequence_with_regions(
        prototype, assignments
    ):
      parts.append(
          (region_info.token_or_decoration, region_info.in_annotated_region)
      )

    result = {
        "extracted_parts": parts,
    }
    if system is not None and evaluation_target is not None:
      assert len(system.data.dag_tables) == 2  # selection dag, constraint dag
      packed_dag = system.data.dag_tables[0].dag
      conversion_data = system.conversion_data[0]
      path, _ = packed_dags.constrained_best_path(
          packed_dag, conversion_data, assignments
      )
      system_cost = sum(edge.cost for edge in path)
      result["total_cost"] = system_cost

      # Count novel identifiers predicted correctly and incorrectly
      correct_novel = 0
      correct_not_novel = 0
      correct_args = 0
      deleted_novel = 0
      deleted_not_novel = 0
      deleted_args = 0
      for edge in path:
        if edge.info is not None:
          assert isinstance(
              edge.info, part_selection_dags.PartSelectionDagEdgeInfo
          )
          if (
              edge.info.prototype_node_preorder_index
              != edit_dags.NO_PREORDER_INDEX
          ):
            node_id = prototype.preorder_traversal[
                edge.info.prototype_node_preorder_index
            ]
            if node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
              token_node = prototype.text_token_nodes[node_id.index_in_category]
              if (
                  edge.info.category
                  == part_selection_dags.DagStateCategory.SELECT_MATCHED
              ):
                if token_node.match_type == "NOVEL":
                  correct_novel += 1
                elif token_node.match_type == "EXPECTED":
                  correct_not_novel += 1
                elif token_node.match_type == "ARGS":
                  correct_args += 1
              elif (
                  edge.info.category
                  == part_selection_dags.DagStateCategory.SELECT_UNMATCHED
              ):
                if token_node.match_type == "NOVEL":
                  deleted_novel += 1
                elif token_node.match_type == "EXPECTED":
                  deleted_not_novel += 1
                elif token_node.match_type == "ARGS":
                  deleted_args += 1

      result.update({
          "correct_novel": correct_novel,
          "correct_not_novel": correct_not_novel,
          "correct_args": correct_args,
          "deleted_novel": deleted_novel,
          "deleted_not_novel": deleted_not_novel,
          "deleted_args": deleted_args,
      })

    if sample_system is not None:
      # Assumption (based on `build_system` above): the system contains all of
      # the edit dags in order, then one constraint dag. We just need the edit
      # dags.
      all_packed_dags = [
          table.dag for table in sample_system.data.dag_tables[:-1]
      ]
      result.update(
          wrapper_base.compute_sample_system_info(
              all_packed_dags=all_packed_dags,
              conversion_datas=sample_system.conversion_data[:-1],
              assignments=assignments,
              total_cost=result.get("total_cost"),
          )
      )

    return result

  def render_solution_html(
      self,
      prototype: PackedSequenceNodeStorage,
      context_info: ContextInfo,
      assignments: dict[
          region_decisions.DecisionKey,
          region_decisions.DecisionValue,
      ],
  ) -> str:
    """See HighLevelUtilityWrapper.render_solution_html."""
    del context_info
    return region_decisions.render_regions_to_html(prototype, assignments)

  def render_pairwise_matching_html(
      self,
      prototype: PackedSequenceNodeStorage,
      targets: list[PackedSequenceNodeStorage],
      context_info: ContextInfo,
      system: DualDecompositionDiagramSystem,
      assignments: dict[
          region_decisions.DecisionKey,
          region_decisions.DecisionValue,
      ],
  ) -> str:
    """See HighLevelUtilityWrapper.render_pairwise_matching_html."""
    source = []
    # Assumption (based on `build_system` above): the system contains all
    # of the selection dags in order, then one constraint dag. We just need the
    # selection dags.
    all_packed_dags = [table.dag for table in system.data.dag_tables[:-1]]
    for i, (packed_dag, conversion_data, target) in enumerate(
        zip(all_packed_dags, system.conversion_data, targets)
    ):
      path, _ = packed_dags.constrained_best_path(
          packed_dag, conversion_data, assignments
      )
      source.append(
          f"========== Target {i} ==========<br>"
          + '<span style="white-space: pre; font-family: monospace;">'
      )
      for part in part_selection_dags.extract_match_groups(
          path, prototype=prototype, target=target
      ):
        if part.category == part_selection_dags.DagStateCategory.SELECT_MATCHED:
          source.append(
              f"MATCHED: {html.escape(' '.join(part.content_tokens))}<br>"
          )
        elif (
            part.category
            == part_selection_dags.DagStateCategory.SELECT_UNMATCHED
        ):
          source.append(
              f"WRONG: {html.escape(' '.join(part.content_tokens))}<br>"
          )
      source.append("</span><br>")

    return "".join(source)
