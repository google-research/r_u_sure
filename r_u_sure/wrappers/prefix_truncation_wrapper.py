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

"""High-level interface for applying R-U-SURE to the prefix truncation task."""

import math
from typing import Any, Optional
from r_u_sure.edit_distance_utility import edit_dags
from r_u_sure.edit_distance_utility import region_decisions
from r_u_sure.tree_structure import packed_sequence_nodes
from r_u_sure.tree_structure import sequence_nodes
from r_u_sure.tree_structure import token_prob_align
from r_u_sure.wrappers import edit_distance_wrapper_base
from r_u_sure.wrappers import uncertainty_regions_wrapper

PackedSequenceNodeStorage = packed_sequence_nodes.PackedSequenceNodeStorage
PackedSequenceNodeCategory = packed_sequence_nodes.PackedSequenceNodeCategory

DecisionKey = Any
DecisionValue = Any

ContextInfo = Any


class PrefixByEditDistanceWrapper(
    edit_distance_wrapper_base.EditDistanceWrapperBase
):
  """Wrapper object that finds a length of a prefix to minimize edit distance.

  This is implemented in mostly the same way as the uncertainty regions wrapper,
  but with a different set of decisions in the prototype: instead of deciding
  whether to put nodes in low-cost regions, we decide whether to early exit.

  This could be optimized a bit to be more efficient by bypassing the constraint
  DAG entirely when uncertainty regions aren't allowed, but that DAG is
  relatively small so it may not be worthwhile. Similarly, some variable
  assignments are redundant.
  """
  show_start_editing_marker = False

  def __init__(
      self,
      start_editing_cost: float = 0.0,
      also_insert_uncertainty_regions: bool = False,
      uncertainty_region_effective_precision: float = 0.7,
      low_confidence_edit_sensitivity: float = 0.5,
      baseline_token_prob_thresholds: tuple[float, ...] = (
          0.0,
          0.3,
          0.5,
          0.7,
          0.9,
      ),
      baseline_prefix_prob_thresholds: tuple[float, ...] = (
          0.01,
          0.02,
          0.05,
          0.1,
          0.2,
          0.3,
          0.5,
          0.7,
          0.9,
      ),
      baseline_max_avg_log_prob: bool = True,
      baseline_intellicode: bool = True,
      baseline_max_characters: tuple[int, ...] = (20, 50, 100, 200, 500),
      baseline_max_lines: tuple[int, ...] = (1, 2, 4, 8, 16),
      use_numba: bool = True,
  ):
    """Constructs a PrefixByEditDistanceWrapper.


    Args:
      start_editing_cost: Cost for starting to insert or delete characters.
      also_insert_uncertainty_regions: Whether to also insert uncertainty
        regions. This has multiple effects: (1) it expands the prototype search
        space to allow low-confidence regions, (2) it disables all baselines,
        which don't use low-confidence regions.
      uncertainty_region_effective_precision: Effective precision for cost
        function. Should only change results if `also_insert_uncertainty_regions
        = True` is set, since it only affects low-confidence regions.
      low_confidence_edit_sensitivity: Sensitivity to edits in low-confidence
        regions. Should only change results if `also_insert_uncertainty_regions
        = True` is set, since it only affects low-confidence regions.
      baseline_token_prob_thresholds: Token probability thresholds to use for
        baselines.
      baseline_prefix_prob_thresholds: Prefix probability thresholds to use.
      baseline_max_avg_log_prob: Whether to use a maximum-average-log-prob
        heuristic, dividing log prob by sequence length and taking the maximum
        cutoff.
      baseline_intellicode: Whether to use IntelliCode Compose heuristic with
        their recommended parameters.
      baseline_max_characters: Character length thresholds to use for baselines.
      baseline_max_lines: Max-line thresholds to use for baselines.
      use_numba: Whether to use Numba to accelerate computation.
    """
    self._also_insert_uncertainty_regions = also_insert_uncertainty_regions
    self._baseline_token_prob_thresholds = baseline_token_prob_thresholds
    self._baseline_prefix_prob_thresholds = baseline_prefix_prob_thresholds
    self._baseline_max_avg_log_prob = baseline_max_avg_log_prob
    self._baseline_intellicode = baseline_intellicode
    self._baseline_max_characters = baseline_max_characters
    self._baseline_max_lines = baseline_max_lines

    # Note: This only matters if also_insert_uncertainty_regions is True.
    self._character_count_cost_config = (
        uncertainty_regions_wrapper.derive_uncertainty_region_costs(
            effective_precision=uncertainty_region_effective_precision,
            high_confidence_start_editing_cost=start_editing_cost,
            low_confidence_edit_sensitivity=low_confidence_edit_sensitivity,
        )
    )
    super().__init__(
        utility_config=edit_dags.make_character_count_cost_config(
            **self._character_count_cost_config
        ),
        use_numba=use_numba,
    )

  def process_prototype_and_context(
      self,
      context_and_prototype: list[sequence_nodes.SequenceNode],
      prediction_location: int,
  ) -> tuple[PackedSequenceNodeStorage, ContextInfo]:
    """See HighLevelUtilityWrapper.process_prototype_and_context."""
    # We always allow early exit, but might not allow low confidence.
    return self._process_prototype_and_context_maybe_exit(
        context_and_prototype,
        prediction_location,
        can_insert_low_confidence=self._also_insert_uncertainty_regions,
        can_early_exit=True,
    )

  def build_baseline_assignments(
      self,
      prototype: PackedSequenceNodeStorage,
      prototype_suggestion_as_target: PackedSequenceNodeStorage,
      context_info: ContextInfo,
      model_tokens_and_log_probs: list[tuple[str, float]],
  ) -> dict[str, dict[DecisionKey, DecisionValue]]:
    """See HighLevelUtilityWrapper.build_baseline_assignments."""
    if self._also_insert_uncertainty_regions:
      # It's awkward to try to combine baselines for uncertainty regions and
      # also prefix length. And the point we are trying to make is just that
      # our approach can do this, not that we are better than any specific
      # baseline. So  we just disable baselines if we are trying to insert
      # uncertainty regions as well as tune length.
      return {}

    baseline_builder = PrefixByEditDistanceBaselineBuilder(
        prototype=prototype,
        prototype_suggestion_as_target=prototype_suggestion_as_target,
        context_info=context_info,
        model_tokens_and_log_probs=model_tokens_and_log_probs,
    )

    results = {}

    for prob_threshold in self._baseline_token_prob_thresholds:
      results[f"prob_threshold_{prob_threshold}"] = (
          baseline_builder.baseline_assignment_from_token_prob(
              prob_threshold,
              cumulative=False,
          )
      )

    for prob_threshold in self._baseline_prefix_prob_thresholds:
      results[f"prefix_prob_threshold_{prob_threshold}"] = (
          baseline_builder.baseline_assignment_from_token_prob(
              prob_threshold,
              cumulative=True,
          )
      )

    if self._baseline_intellicode:
      results["intellicode"] = (
          baseline_builder.baseline_assignment_from_intellicode_compose()
      )

    if self._baseline_max_avg_log_prob:
      results["max_avg_log_prob"] = (
          baseline_builder.baseline_assignment_from_max_average_log_prob()
      )

    for character_length in self._baseline_max_characters:
      results[f"character_length_{character_length}"] = (
          baseline_builder.baseline_assignment_from_characters(character_length)
      )

    for line_length in self._baseline_max_lines:
      results[f"line_length_{line_length}"] = (
          baseline_builder.baseline_assignment_from_lines(line_length)
      )
    return results


class PrefixByEditDistanceBaselineBuilder:
  """Helper to compute baselines for prefix length tuning."""

  def __init__(
      self,
      prototype: PackedSequenceNodeStorage,
      prototype_suggestion_as_target: PackedSequenceNodeStorage,
      context_info: ContextInfo,
      model_tokens_and_log_probs: list[tuple[str, float]],
  ):
    """Constructs a baseline builder."""
    self.prototype = prototype
    self.prototype_suggestion_as_target = prototype_suggestion_as_target
    self.context_info = context_info
    self.model_tokens_and_log_probs = model_tokens_and_log_probs

    # Align the log probs.
    self.aligned_log_probs = token_prob_align.align_token_log_probs(
        model_tokens_and_log_probs, prototype
    )

  def _assignment_from_exit_point(
      self,
      early_exit_preorder_index: Optional[int],
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Builds an assignment based on constraints on low-confidence tokens.

    Args:
      early_exit_preorder_index: Preorder index of the node that we should early
        exit at, if any.

    Returns:
      A (possibly incomplete, but sufficient) set of assignments that ensure
      that the path that early-exits at that node, if provided, or doesn't
      early exit otherwise. Also ensures that no low-confidence regions start.
      This set of assignments is suitable for use as constraints for a
      system.
    """
    constraints = {}
    already_exited = False
    for node_id in self.prototype.preorder_traversal:
      if node_id.category == PackedSequenceNodeCategory.EARLY_EXIT_NODE:
        key = region_decisions.DecisionKey(
            prototype_preorder_index=node_id.preorder_index,
            category=region_decisions.DecisionCategory.SHOULD_EARLY_EXIT,
        )
        if node_id.preorder_index == early_exit_preorder_index:
          assert not already_exited
          already_exited = True
          constraints[key] = region_decisions.DecisionValue.TRUE
        elif already_exited:
          constraints[key] = region_decisions.DecisionValue.NOT_APPLICABLE
        else:
          constraints[key] = region_decisions.DecisionValue.FALSE
      elif node_id.category == PackedSequenceNodeCategory.REGION_START_NODE:
        # Never allow low-confidence regions to start.
        constraints[
            region_decisions.DecisionKey(
                prototype_preorder_index=node_id.preorder_index,
                category=(
                    region_decisions.DecisionCategory.REGION_SHOULD_START
                ),
            )
        ] = region_decisions.DecisionValue.FALSE

    if early_exit_preorder_index is not None and not already_exited:
      raise ValueError(
          "early_exit_preorder_index was provided (with value"
          f" {early_exit_preorder_index}) but no early exit node was found with"
          " that index."
      )

    return constraints

  def baseline_assignment_from_token_prob(
      self,
      probability_threshold: float,
      cumulative: bool,
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Truncates at the first token below a given log probability."""
    current_log_prob = 0.0
    last_exit_seen = None
    should_exit_at_next_opportunity = False
    for node_id in self.prototype.preorder_traversal:
      if node_id.category == PackedSequenceNodeCategory.EARLY_EXIT_NODE:
        last_exit_seen = node_id.preorder_index
        if should_exit_at_next_opportunity:
          return self._assignment_from_exit_point(node_id.preorder_index)
      elif node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
        token_node = self.prototype.text_token_nodes[node_id.index_in_category]
        if token_node.text_contents:
          node_log_prob = self.aligned_log_probs[node_id.preorder_index]
          if cumulative:
            current_log_prob += node_log_prob
          else:
            current_log_prob = node_log_prob
          if math.exp(current_log_prob) < probability_threshold:
            if last_exit_seen is not None:
              # Exit BEFORE seeing this node if possible!
              return self._assignment_from_exit_point(last_exit_seen)
            else:
              should_exit_at_next_opportunity = True

    # Didn't find any early exit point.
    return self._assignment_from_exit_point(None)

  def baseline_assignment_from_max_average_log_prob(
      self,
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Truncates at the position with the highest average log probability."""
    current_log_prob = 0.0
    current_token_count = 0

    best_average_log_prob_at_exit_point = -math.inf
    best_exit = None

    for node_id in self.prototype.preorder_traversal:
      if node_id.category == PackedSequenceNodeCategory.EARLY_EXIT_NODE:
        if current_token_count > 0:
          average_log_prob = current_log_prob / current_token_count
          if average_log_prob > best_average_log_prob_at_exit_point:
            best_average_log_prob_at_exit_point = average_log_prob
            best_exit = node_id.preorder_index

      elif node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
        token_node = self.prototype.text_token_nodes[node_id.index_in_category]
        if token_node.text_contents:
          node_log_prob = self.aligned_log_probs[node_id.preorder_index]
          current_log_prob += node_log_prob
          current_token_count += 1

    if best_exit is not None:
      return self._assignment_from_exit_point(best_exit)
    else:
      return self._assignment_from_exit_point(None)

  def baseline_assignment_from_intellicode_compose(
      self,
      alpha: float = 0.8,
      kappa: float = 10.0,
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Uses a version of the IntelliCode Compose heuristic to truncate.

    The IntelliCode Compose heuristic chooses to truncate a suggestion based on
    a trie of tokens generated from beam-search. They state: "we terminate the
    completion-tree traversal if none of the child nodes has a score that is
    equal to or larger than the score of its parent multiplied by a ratio R,
    defined as: R = alpha / (1 + exp(-L / kappa))"
    where "L is the position of the root node of the trie", and they recommend
    setting alpha=0.8, kappa=10. (See https://arxiv.org/pdf/2005.08025.pdf)

    We use sampling instead of beam search, so we do not exactly have a trie
    of suggestions. However, we can still apply a similar technique.
    Assumptions / interpretations:

    - The "score" above was likely total log probability, and the ratio of
      parent to child would then be the probability of each individual token.
      So we can equivalently check if the log prob of this token is < R.
    - The "position of the root node" is unclear; it sounds like it refers to
      the position of the root node in the file, but they later refer to it as
      if it is a curve over the length of the suggestion. Here I'm just using it
      to refer to the number of tokens seen.

    Args:
      alpha: Value of `alpha` parameter in equation for R.
      kappa: Value of `kappa` parameter in equation for R.

    Returns:
      An assignment dictionary.
    """
    last_exit_seen = None
    should_exit_at_next_opportunity = False
    tokens_seen = 0
    for node_id in self.prototype.preorder_traversal:
      if node_id.category == PackedSequenceNodeCategory.EARLY_EXIT_NODE:
        last_exit_seen = node_id.preorder_index
        if should_exit_at_next_opportunity:
          return self._assignment_from_exit_point(node_id.preorder_index)
      elif node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
        token_node = self.prototype.text_token_nodes[node_id.index_in_category]
        if token_node.text_contents:
          node_log_prob = self.aligned_log_probs[node_id.preorder_index]
          threshold_r = alpha / (1 + math.exp(-tokens_seen / kappa))
          if math.exp(node_log_prob) < threshold_r:
            if last_exit_seen is not None:
              # Exit BEFORE seeing this node if possible!
              return self._assignment_from_exit_point(last_exit_seen)
            else:
              should_exit_at_next_opportunity = True
          tokens_seen += 1

    # Didn't find any early exit point.
    return self._assignment_from_exit_point(None)

  def baseline_assignment_from_characters(
      self,
      max_token_characters: int,
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Truncates after outputting a certain number of characters."""
    characters_so_far = 0
    should_exit_at_next_opportunity = False
    for node_id in self.prototype.preorder_traversal:
      if node_id.category == PackedSequenceNodeCategory.EARLY_EXIT_NODE:
        if should_exit_at_next_opportunity:
          return self._assignment_from_exit_point(node_id.preorder_index)
      elif node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
        node = self.prototype.text_token_nodes[node_id.index_in_category]
        characters_so_far += len(node.text_contents)
        if characters_so_far >= max_token_characters:
          should_exit_at_next_opportunity = True

    # Didn't find any early exit point.
    return self._assignment_from_exit_point(None)

  def baseline_assignment_from_lines(
      self,
      max_lines: int,
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Truncates after outputting a certain number of lines."""
    # Note: Newline characters appear inside decoration nodes.
    # We assume newline characters will be followed by an early exit node
    # because they end a statement. We could alternatively try to early exit
    # right before the newline but they should be equivalent.
    newlines_so_far = 0
    should_exit_at_next_opportunity = False
    for node_id in self.prototype.preorder_traversal:
      if node_id.category == PackedSequenceNodeCategory.EARLY_EXIT_NODE:
        if should_exit_at_next_opportunity:
          return self._assignment_from_exit_point(node_id.preorder_index)
      elif node_id.category == PackedSequenceNodeCategory.TEXT_DECORATION_NODE:
        node = self.prototype.text_decoration_nodes[node_id.index_in_category]
        if "\n" in node.text_contents:
          newlines_so_far += 1
          if newlines_so_far >= max_lines:
            should_exit_at_next_opportunity = True

    # Didn't find any early exit point.
    return self._assignment_from_exit_point(None)
