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

"""Base API for high level task wrappers."""

import bisect
from typing import Any, Optional, Sequence

import numpy as np

from r_u_sure.decision_diagrams import consistent_path_dual_solver
from r_u_sure.decision_diagrams import packed_dags
from r_u_sure.tree_structure import packed_sequence_nodes
from r_u_sure.tree_structure import sequence_nodes

DecisionKey = Any
DecisionValue = Any
ContextInfo = Any

PackedSequenceNodeStorage = packed_sequence_nodes.PackedSequenceNodeStorage
DualDecompositionDiagramSystem = (
    consistent_path_dual_solver.DualDecompositionDiagramSystem
)


class HighLevelUtilityWrapper:
  """Interface for a high-level cost function wrapper."""

  def process_prototype_and_context(
      self,
      context_and_prototype: list[sequence_nodes.SequenceNode],
      prediction_location: int,
  ) -> tuple[PackedSequenceNodeStorage, ContextInfo]:
    """Transforms an input string into a prototype object for later use.

    Prototype objects include special markers to determine the possible set of
    outputs the optimization process could produce (in particular, early exit
    nodes and region start/end nodes).

    Args:
      context_and_prototype: Parsed representation of context and prototype
        suggestion, concatenated together.
      prediction_location: Offset into `context_and_prototype` representing the
        end of the context and the start of the prototype.

    Returns:
      Tuple (processed_prototype, context_info). Here context_info is any
      extra information about the context that we need for building the system.
    """
    raise NotImplementedError

  def process_target(
      self,
      context_and_target: list[sequence_nodes.SequenceNode],
      prediction_location: int,
  ) -> PackedSequenceNodeStorage:
    """Transforms an input string into a target object for later use.

    Target objects represent plausible ground truth completions. They can either
    be samples from the model (for use in optimization), or actual ground truth
    completions (for use in evaluation).

    Args:
      context_and_target: Parsed representation of context and target
        suggestion, concatenated together.
      prediction_location: Offset into `context_and_target` representing the end
        of the context and the start of the target.

    Returns:
      Processed version of target.
    """
    raise NotImplementedError

  def build_system(
      self,
      prototype: PackedSequenceNodeStorage,
      context_info: ContextInfo,
      targets: list[PackedSequenceNodeStorage],
      target_utility_scale_factors: Optional[list[float]] = None,
  ) -> DualDecompositionDiagramSystem:
    """Combines a prototype, targets, and context into a system.

    A system allows both:
    - computing the cost of showing a template (derived from the prototype)
      averaged across a number of possible targets,
    - solving for a template to minimize this cost.

    Two typical uses:
    - building a system whose targets are samples from a ML model, then
      optimizing to find the best template for that system,
    - building a system with one target representing the ground truth output,
      then evaluating it with a concrete template to compute a cost.

    Args:
      prototype: The prototype object, used to determine the template, as
        returned by `process_prototype_and_context`.
      context_info: Information about the context, as returned by
        `process_prototype_and_context`.
      targets: A list of target objects, as returned by `process_target`. All
        targets should share the same context as the prototype.
      target_utility_scale_factors: Optional scale factors for utility
        functions; these should usually sum to one. If not provided, assumes a
        uniform distribution.
    """
    raise NotImplementedError

  def solution_info(
      self,
      prototype: PackedSequenceNodeStorage,
      evaluation_target: Optional[PackedSequenceNodeStorage],
      context_info: ContextInfo,
      system: Optional[DualDecompositionDiagramSystem],
      assignments: dict[DecisionKey, DecisionValue],
      sample_system: Optional[DualDecompositionDiagramSystem] = None,
  ) -> dict[str, Any]:
    """Computes a dictionary of summary information about a solution.

    Args:
      prototype: The prototype object, used to determine the template, as
        returned by `process_prototype_and_context`.
      evaluation_target: The target object used to evaluate the solution.
      context_info: Information about the context, as returned by
        `process_prototype_and_context`.
      system: A system constructed by `build_system` for `evaluation_target`.
      assignments: An assignment dictionary, as computed by
        `consistent_path_dual_solver.assignments_from_assignment_vector`
      sample_system: Optionally, an additional system build from model samples,
        which can be used to evaluate how in-distribution the evaluation results
        are.

    Returns:
      Arbitrary info about a solution, which might be useful for e.g. plotting
      across a dataset of examples.
    """
    raise NotImplementedError

  def render_solution_html(
      self,
      prototype: PackedSequenceNodeStorage,
      context_info: ContextInfo,
      assignments: dict[DecisionKey, DecisionValue],
  ) -> str:
    """Renders the solution suggestion to HTML.

    This is used for interactive inspection and debugging.

    Args:
      prototype: The prototype object, used to determine the template, as
        returned by `process_prototype_and_context`.
      context_info: Information about the context, as returned by
        `process_prototype_and_context`.
      assignments: An assignment dictionary, as computed by
        `consistent_path_dual_solver.assignments_from_assignment_vector`

    Returns:
      An HTML source string that can be rendered in a Colab/IPython notebook.
    """
    raise NotImplementedError

  def render_pairwise_matching_html(
      self,
      prototype: PackedSequenceNodeStorage,
      targets: list[PackedSequenceNodeStorage],
      context_info: ContextInfo,
      system: DualDecompositionDiagramSystem,
      assignments: dict[DecisionKey, DecisionValue],
  ) -> str:
    """Renders information about how a suggestion matches with samples to HTML.

    This is used for interactive inspection and debugging.

    Args:
      prototype: The prototype object, used to determine the template, as
        returned by `process_prototype_and_context`.
      targets: List of targets we are comparing it with.
      context_info: Information about the context, as returned by
        `process_prototype_and_context`.
      system: A system constructed by `build_system` for `targets`.
      assignments: An assignment dictionary, as computed by
        `consistent_path_dual_solver.assignments_from_assignment_vector`

    Returns:
      An HTML source string that can be rendered in a Colab/IPython notebook.
    """
    raise NotImplementedError

  def build_baseline_assignments(
      self,
      prototype: PackedSequenceNodeStorage,
      prototype_suggestion_as_target: PackedSequenceNodeStorage,
      context_info: ContextInfo,
      model_tokens_and_log_probs: list[tuple[str, float]],
  ) -> dict[str, dict[DecisionKey, DecisionValue]]:
    """Constructs assignments for baselines, potentially using token log probs.

    Args:
      prototype: The prototype object, used to determine the template, as
        returned by `process_prototype_and_context`.
      prototype_suggestion_as_target: The same suggestion, but processed as a
        target, e.g. returned by `process_target`.
      context_info: Information about the context, as returned by
        `process_prototype_and_context`.
      model_tokens_and_log_probs: Pairs of model token and log probability to
        use when constructing baselines.

    Returns:
      Dictionary mapping baseline names to an assignment to use for each.
    """
    raise NotImplementedError


def compute_sample_system_info(
    all_packed_dags: Sequence[packed_dags.PackedStateDAG],
    conversion_datas: Sequence[packed_dags.PackedStateDAGConversionData],
    assignments: dict[DecisionKey, DecisionValue],
    total_cost: Optional[float] = None,
) -> dict[str, Any]:
  """Helper function to extract sample system costs."""
  costs = []
  for packed_dag, conversion_data in zip(all_packed_dags, conversion_datas):
    # NOTE: We want the cost for the *unweighted* version of packed_dag,
    # without accounting for `target_utility_scale_factors`. This works
    # because `constrained_best_path` extracts paths from the original
    # DAG and not the postprocessed one. However, if you directly inspect
    # the cost of the path in the internal representation, it will be
    # scaled down!
    path, _ = packed_dags.constrained_best_path(
        packed_dag, conversion_data, assignments
    )
    costs.append(sum(edge.cost for edge in path))

  sorted_costs = sorted(costs)
  result = {}
  result["sample_system_costs_unsorted"] = costs
  result["sample_system_costs"] = sorted_costs
  result["sample_system_cost_average"] = np.mean(sorted_costs)
  result["sample_system_cost_variance"] = np.var(sorted_costs, ddof=1)
  if total_cost is not None:
    result["true_cost_rank_in_sample_system"] = bisect.bisect_left(
        sorted_costs, total_cost
    )
  return result
