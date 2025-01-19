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

"""High-level interface for applying R-U-SURE to the edit localization task."""

import math
from typing import Any
from r_u_sure.decision_diagrams import consistent_path_dual_solver
from r_u_sure.edit_distance_utility import edit_dags
from r_u_sure.edit_distance_utility import region_decisions
from r_u_sure.tree_structure import packed_sequence_nodes
from r_u_sure.tree_structure import sequence_nodes
from r_u_sure.tree_structure import token_prob_align
from r_u_sure.wrappers import edit_distance_wrapper_base

PackedSequenceNodeStorage = packed_sequence_nodes.PackedSequenceNodeStorage
PackedSequenceNodeCategory = packed_sequence_nodes.PackedSequenceNodeCategory
InOrderTraversalCategory = packed_sequence_nodes.InOrderTraversalCategory
DualDecompositionDiagramSystem = (
    consistent_path_dual_solver.DualDecompositionDiagramSystem
)

DecisionKey = Any
DecisionValue = Any

ContextInfo = Any


def derive_uncertainty_region_costs(
    effective_precision: float,
    high_confidence_start_editing_cost: float,
    low_confidence_edit_sensitivity: float,
) -> edit_dags.TrustRegionUtilityParameters:
  """Computes parameters for UncertaintyRegionsWrapper based on precision.

  The UncertaintyRegionsWrapper is configured with on a base penalty for
  low-confidence regions along with a separate penalty for edits. In general,
  the low confidence regions have a smaller penalty for edits, meaning that
  regions with many edits should be placed in low-confidence areas. However,
  it's not obvious what this penalty should be.

  This function identifies penalties such that, if a suggested token is the
  correct length, but has only probability `p` of being the correct actual
  token, then it is optimal to put it in a low-confidence region whenever `p <
  effective_precision`, and optimal to keep it in a high-confidence region
  whenever `p > effective_precision`.

  Args:
    effective_precision: The probability level that is high enough to not show a
      low-confidence region.
    high_confidence_start_editing_cost: The cost for having to start editing
      some characters into high confidence regions. (We don't have penalties for
      the number of characters inserted, to avoid annoying edge cases for rare
      insertions of large numbers of characters.)
    low_confidence_edit_sensitivity: Relative cost (between 0 and 1) of edits in
      low-confidence regions, compared to edits in high-confidence regions. If
      this is 0, edits are free in low-confidence regions, but those regions
      have large fixed penalties regardless of the ground truth. A value of
      exactly 0 may lead to spurious empty edits (e.g. we start an edit but
      don't edit anything). If this is close to 1, edits cost almost the same as
      in high-confidence regions, and the fixed penalties for low-confidence
      regions are smaller. When exactly 1, there is no difference between high
      and low confidence regions. Values too close to 1 may be unstable.

  Returns:
    Parameters for a character-count utility.
  """
  # Consider a single token of length L in a proposed suggestion, which we can
  # optionally place in a low-confidence region. We will assume that insertion
  # costs are NOT correlated with the length of the insertion, e.g. the per
  # character costs of insertions are zero.
  #
  # There are four relevant situations:
  # * High confidence, correct: We have utility
  #     L * high_conf_match_util
  # * Low confidence, correct: We pay a small penalty for the low-confidence
  #   region and also get a bit less utility:
  #     L * low_conf_match_util - low_conf_region_cost
  # * High confidence, incorrect: We must start editing, then delete the token.
  #     - L * high_conf_delete_cost - high_conf_start_edit_cost
  # * Low confidence, incorrect: We pay a penalty for the low-confidence region,
  #   and also pay for the edits. The total utility is
  #     - L * low_conf_delete_cost - low_conf_start_edit_cost
  #        - low_conf_region_cost
  #
  # To get an effective precision `p`, we want to set all of the costs such that
  # - being correct is always better than being incorrect,
  # - if we have a larger-than-`p` chance of being correct, it's better to use
  #   a high-confidence region,
  # - if we have a smaller-than-`p` chance of being correct, it's better to use
  #   a low-confidence region.
  #
  # We can accomplish this by computing expected utilities conditional on having
  # a probability `p` of being correct, and ensuring these are equal for high
  # and low confidence regions. In other words, we want
  #
  #  p * (L * high_conf_match_util)
  #       + (1-p) * (L * high_conf_delete_cost - high_conf_start_edit_cost)
  #    ==
  #    p * (L * low_conf_match_util)
  #         + (1-p) * (L * low_conf_delete_cost - low_conf_start_edit_cost)
  #        - low_conf_region_cost
  #
  # We want this to be true for all L, which implies that both
  #
  # p * high_conf_match_util + (1-p) * -high_conf_delete_cost
  #   == p * low_conf_match_util + (1-p) * -low_conf_delete_cost
  #
  # (1-p) * -high_conf_start_edit_cost
  #   == (1-p) * -low_conf_start_edit_cost - low_conf_region_cost
  #
  # These are two equations over 7 parameters, so we have five degrees of
  # freedom. We choose to set high_conf_match_util = 1, and
  # high_conf_delete_cost = 1, allow high_conf_start_edit_cost to be configured,
  # and also fix the ratios
  #   (
  #     (low_conf_match_util - (-low_conf_delete_cost))
  #     / (high_conf_match_util - (-high_conf_delete_cost))
  #   ) == low_confidence_edit_sensitivity
  #   (
  #     low_conf_start_edit_cost / high_conf_start_edit_cost
  #   ) == low_confidence_edit_sensitivity
  # These are chosen to make the remaining parameters easy to solve
  # for and well-conditioned for different values of `p`. Here
  # `low_confidence_edit_sensitivity` roughly measures how much easier it is
  # to make edits in low-confidence regions.

  p = effective_precision

  # From
  #   p * high_conf_match_util + (1-p) * -high_conf_delete_cost
  #     == p * low_conf_match_util + (1-p) * -low_conf_delete_cost
  # we have
  #   p - (1-p) == p * (low_conf_match_util + low_conf_delete_cost)
  #                - low_conf_delete_cost.
  # Since
  #   low_conf_match_util + low_conf_delete_cost
  #   == 2 * low_confidence_edit_sensitivity,
  # we derive
  low_conf_delete_cost = 1 - 2 * p * (1 - low_confidence_edit_sensitivity)
  low_conf_match_util = (
      2 * low_confidence_edit_sensitivity - low_conf_delete_cost
  )
  # Note that, when `low_confidence_edit_sensitivity = 1` we recover 1 and 1
  # here, and when `low_confidence_edit_sensitivity = 0` the utility is the
  # same for matches and deletions.

  # From
  #   (1-p) * -high_conf_start_edit_cost
  #     == (1-p) * -low_conf_start_edit_cost - low_conf_region_cost
  # we find that
  #   low_conf_region_cost
  #     == (1 - p) * (high_conf_start_edit_cost - low_conf_start_edit_cost)
  #     == (1 - p) * (1 - low_confidence_edit_sensitivity)
  #                * high_conf_start_edit_cost
  low_conf_start_edit_cost = (
      low_confidence_edit_sensitivity * high_confidence_start_editing_cost
  )
  low_conf_region_cost = (
      (1 - p)
      * (1 - low_confidence_edit_sensitivity)
      * high_confidence_start_editing_cost
  )
  # Observations:
  # - When `low_confidence_edit_sensitivity` is close to 0,
  #   `low_conf_start_edit_cost` becomes close to 0 also,
  #   and the low-confidence region cost adjusts so that it costs as much as the
  #   average cost of maybe-editing in a high-confidence region.
  # - When `low_confidence_edit_sensitivity` is close to 1,
  #   `low_conf_start_edit_cost` becomes close to
  #   `high_confidence_start_editing_cost`,
  #   and `low_conf_region_cost` drops to near 0.
  # - As `p` increases, the cost of a low-cost region decreases, making it more
  #   likely that we will add a low-cost region.
  # - If `high_confidence_start_editing_cost` is zero, so are
  #   `low_conf_start_edit_cost` and `low_conf_region_cost`. This means that
  #   the only relevant terms are the per-character utility of matches and
  #   cost of deletions. (Essentially, this means we ignore discrete editing
  #   actions, and just try to identify mispredicted chars.)
  return edit_dags.TrustRegionUtilityParameters(
      high_confidence_match_utility_per_char=1.0,
      high_confidence_delete_cost_per_char=1.0,
      low_confidence_match_utility_per_char=low_conf_match_util,
      low_confidence_delete_cost_per_char=low_conf_delete_cost,
      insert_cost_per_char=0.0,
      low_confidence_region_cost=low_conf_region_cost,
      high_confidence_start_editing_cost=high_confidence_start_editing_cost,
      low_confidence_start_editing_cost=low_conf_start_edit_cost,
  )


class UncertaintyRegionsWrapper(
    edit_distance_wrapper_base.EditDistanceWrapperBase
):
  """Wrapper object that tags uncertain tokens to make them easier to identify."""
  show_start_editing_marker = True

  def __init__(
      self,
      effective_precision: float = 0.7,
      high_confidence_start_inserting_cost: float = 5.0,
      low_confidence_edit_sensitivity: float = 0.5,
      use_numba: bool = True,
      baseline_token_prob_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9),
      baseline_example_fractions: tuple[float, ...] = (0.0, 0.5, 1.0),
      baseline_depth_cutoffs: tuple[tuple[int, int], ...] = (),
  ):
    """Constructs a UncertaintyRegionsWrapper.

    See `derive_uncertainty_region_costs` for details on the
    meaning of arguments.

    Args:
      effective_precision: The probability level that is high enough to not show
        a low-confidence region.
      high_confidence_start_inserting_cost: The cost for having to start
        inserting some characters into high confidence regions.
      low_confidence_edit_sensitivity: Relative cost (between 0 and 1) of edits
        in low-confidence regions, compared to edits in high-confidence regions.
      use_numba: Whether to use Numba to accelerate computation.
      baseline_token_prob_thresholds: List of thresholds to use for token-prob
        baselines. Any token with a lower conditional probability than this will
        be put into a low-confidence region.
      baseline_example_fractions: List of fractions to use for example fraction
        baselines. Any token that is farther than this fraction into the
        suggestion will be put in a low-confidence region.
      baseline_depth_cutoffs: List of pairs (ancestor_cutoff, child_cutoff)
        giving depth cutoffs. See method `baseline_assignment_from_depth_cutoff`
        of `UncertaintyRegionsBaselineBuilder`.
    """
    self._baseline_token_prob_thresholds = baseline_token_prob_thresholds
    self._baseline_example_fractions = baseline_example_fractions
    self._baseline_depth_cutoffs = baseline_depth_cutoffs
    self._character_count_cost_config = derive_uncertainty_region_costs(
        effective_precision,
        high_confidence_start_inserting_cost,
        low_confidence_edit_sensitivity,
    )
    super().__init__(
        utility_config=self._character_count_cost_config,
        use_numba=use_numba,
    )

  def process_prototype_and_context(
      self,
      context_and_prototype: list[sequence_nodes.SequenceNode],
      prediction_location: int,
  ) -> tuple[PackedSequenceNodeStorage, ContextInfo]:
    """See HighLevelUtilityWrapper.process_prototype_and_context."""
    return self._process_prototype_and_context_maybe_exit(
        context_and_prototype,
        prediction_location,
        can_insert_low_confidence=True,
        can_early_exit=False,
    )

  def build_baseline_assignments(
      self,
      prototype: PackedSequenceNodeStorage,
      prototype_suggestion_as_target: PackedSequenceNodeStorage,
      context_info: ContextInfo,
      model_tokens_and_log_probs: list[tuple[str, float]],
  ) -> dict[str, dict[DecisionKey, DecisionValue]]:
    """See HighLevelUtilityWrapper.build_baseline_assignments."""
    baseline_builder = UncertaintyRegionsBaselineBuilder(
        wrapper=self,
        prototype=prototype,
        prototype_suggestion_as_target=prototype_suggestion_as_target,
        context_info=context_info,
        model_tokens_and_log_probs=model_tokens_and_log_probs,
    )

    results = {}
    for prob_threshold in self._baseline_token_prob_thresholds:
      # results[f"prob_threshold_{prob_threshold}_with_empty"] = (
      #     baseline_builder.baseline_assignment_from_token_probabilities(
      #         prob_threshold,
      #         only_constrain_nonempty=False,
      #         cumulative=False,
      #     )
      # )
      results[f"prob_threshold_{prob_threshold}_nonempty"] = (
          baseline_builder.baseline_assignment_from_token_probabilities(
              prob_threshold,
              only_constrain_nonempty=True,
              cumulative=False,
          )
      )
      results[f"prob_threshold_{prob_threshold}_cumulative"] = (
          baseline_builder.baseline_assignment_from_token_probabilities(
              prob_threshold,
              only_constrain_nonempty=True,
              cumulative=True,
          )
      )

    for example_fraction in self._baseline_example_fractions:
      results[f"example_fraction_{example_fraction}"] = (
          baseline_builder.baseline_assignment_from_example_fraction(
              example_fraction
          )
      )

    for ancestor_cutoff, child_cutoff in self._baseline_depth_cutoffs:
      results[f"cutoff_anc_{ancestor_cutoff}_child_{child_cutoff}"] = (
          baseline_builder.baseline_assignment_from_depth_cutoff(
              child_depth_cutoff=child_cutoff,
              ancestor_depth_cutoff=ancestor_cutoff,
          )
      )

    return results


class UncertaintyRegionsBaselineBuilder:
  """Helper to compute baselines for uncertainty regions."""

  def __init__(
      self,
      wrapper: UncertaintyRegionsWrapper,
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

    # Make a system with just the prototype. We use this to facilitate
    # generating valid assignments that respect the constraints.
    # (Technically this is a bit more expensive than it needs to be, but it's
    # easier to implement this way. Runtime should still be dominated by the
    # MBR optimization over the larger set of samples.)
    self.baseline_system = wrapper.build_system(
        prototype=prototype,
        context_info=context_info,
        targets=[prototype_suggestion_as_target],
    )

  def _assignment_from_low_confidence_nodes(
      self,
      low_confidence_preorder_indices: list[int],
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Builds an assignment based on constraints on low-confidence tokens.

    Args:
      low_confidence_preorder_indices: List of preorder indices of tokens that
        must be placed in low-confidence regions.

    Returns:
      A full assignment that assigns those indices to low-confidence regions
      while maximizing the utility for the prototype itself otherwise.
    """
    constraints = {}
    for constrained_index in low_confidence_preorder_indices:
      constraints[
          region_decisions.DecisionKey(
              prototype_preorder_index=constrained_index,
              category=region_decisions.DecisionCategory.NODE_IN_REGION,
          )
      ] = region_decisions.DecisionValue.TRUE
    system_copy = consistent_path_dual_solver.clone_system(self.baseline_system)
    # Penalize with a very large number, but not infinite, just in case there
    # are some nodes we are not structurally allowed to put in a low-confidence
    # region (for instance, if there aren't any region end nodes after it).
    consistent_path_dual_solver.constrain_system(
        system_copy, constraints, penalty=1e6
    )
    # Solve our mini-system. This should be very fast, since the optimal thing
    # to do is just put as little into regions as possible. We set a
    # timeout just to be safe.
    consistent_path_dual_solver.solve_system_with_sweeps(
        system_data=system_copy.data, soft_timeout=10.0
    )
    greedy_assignments, _ = consistent_path_dual_solver.greedy_extract(
        system_copy.data,
        direction=consistent_path_dual_solver.SweepDirection.FORWARD,
    )
    return consistent_path_dual_solver.assignments_from_assignment_vector(
        system_copy, greedy_assignments
    )

  def baseline_assignment_from_token_probabilities(
      self,
      probability_threshold: float,
      only_constrain_nonempty: bool,
      cumulative: bool,
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Puts all tokens with a lower probability than this into low-conf regions."""
    low_confidence_nodes = []
    current_log_prob = 0.0
    for preorder_index, log_prob in self.aligned_log_probs.items():
      if cumulative:
        current_log_prob += log_prob
      else:
        current_log_prob = log_prob
      if math.exp(current_log_prob) < probability_threshold:
        if only_constrain_nonempty:
          node_id = self.prototype.preorder_traversal[preorder_index]
          assert node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE
          token_node = self.prototype.text_token_nodes[
              node_id.index_in_category
          ]
          if token_node.text_contents:
            low_confidence_nodes.append(preorder_index)
        else:
          low_confidence_nodes.append(preorder_index)
    return self._assignment_from_low_confidence_nodes(low_confidence_nodes)

  def baseline_assignment_from_character_offset(
      self, offset_in_chars: float
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Puts all tokens further than this into the example into low-conf regions."""
    low_confidence_nodes = []
    seen_so_far = 0
    for node_id in self.prototype.preorder_traversal:
      if node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
        token_node = self.prototype.text_token_nodes[node_id.index_in_category]
        seen_so_far += len(token_node.text_contents)
        # If this token ends after the offset, it should be put in a
        # low-confidence region. Note: we don't apply this to tokens that don't
        # contain any text.
        if token_node.text_contents and seen_so_far > offset_in_chars:
          low_confidence_nodes.append(node_id.preorder_index)
      elif node_id.category == PackedSequenceNodeCategory.TEXT_DECORATION_NODE:
        dec_node = self.prototype.text_decoration_nodes[
            node_id.index_in_category
        ]
        seen_so_far += len(dec_node.text_contents)
    return self._assignment_from_low_confidence_nodes(low_confidence_nodes)

  def baseline_assignment_from_example_fraction(
      self, fraction: float
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Puts all tokens further than this into the example into low-conf regions."""
    # Measure the length of the suggestion.
    total_size = 0
    for node_id in self.prototype.preorder_traversal:
      if node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
        token_node = self.prototype.text_token_nodes[node_id.index_in_category]
        total_size += len(token_node.text_contents)
      elif node_id.category == PackedSequenceNodeCategory.TEXT_DECORATION_NODE:
        dec_node = self.prototype.text_decoration_nodes[
            node_id.index_in_category
        ]
        total_size += len(dec_node.text_contents)
    # Put that much of it in a low-confidence region.
    offset = fraction * total_size
    return self.baseline_assignment_from_character_offset(offset)

  def baseline_assignment_from_depth_cutoff(
      self,
      ancestor_depth_cutoff: int,
      child_depth_cutoff: int,
  ) -> dict[region_decisions.DecisionKey, region_decisions.DecisionValue]:
    """Puts tokens into low-confidence regions based on their depth.

    All depths are computed relative to the depth of the first token node in
    the suggestion, and based on the number of (nested) group nodes containing
    the token.

    Args:
      ancestor_depth_cutoff: Any node that is this many levels more shallow than
        the first token will be put in a low-confidence region.
      child_depth_cutoff: Any node that is this many levels deeper than the
        first token will be put in a low-confidence region.

    Returns:
      Baseline assignment.
    """
    if child_depth_cutoff < 1:
      raise ValueError("`child_depth_cutoff` must be >= 1")
    if ancestor_depth_cutoff < 1:
      raise ValueError("`ancestor_depth_cutoff` must be >= 1")
    low_confidence_nodes = []
    current_depth = 0
    first_token_depth = None
    for traversal_item in packed_sequence_nodes.in_order_traversal(
        self.prototype, strip_decorations=True
    ):
      if traversal_item.category == InOrderTraversalCategory.BEFORE_GROUP:
        current_depth += 1
      elif traversal_item.category == InOrderTraversalCategory.AFTER_GROUP:
        current_depth -= 1
      elif (
          traversal_item.node_id.category
          == PackedSequenceNodeCategory.TEXT_TOKEN_NODE
      ):
        if first_token_depth is None:
          first_token_depth = current_depth
        elif (
            current_depth <= first_token_depth - ancestor_depth_cutoff
            or current_depth >= first_token_depth + child_depth_cutoff
        ):
          low_confidence_nodes.append(traversal_item.node_id.preorder_index)

    return self._assignment_from_low_confidence_nodes(low_confidence_nodes)
