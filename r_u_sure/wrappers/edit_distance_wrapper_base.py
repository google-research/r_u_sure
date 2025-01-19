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

"""Base class for wrappers that use confidence-adjusted edit distance."""

import functools
import html
from typing import Any, Optional
import numpy as np
from r_u_sure.decision_diagrams import consistent_path_dual_solver
from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.decision_diagrams import packed_dags
from r_u_sure.edit_distance_utility import constraint_dags
from r_u_sure.edit_distance_utility import edit_dags
from r_u_sure.edit_distance_utility import region_decisions
from r_u_sure.tree_structure import packed_sequence_nodes
from r_u_sure.tree_structure import sequence_node_helpers
from r_u_sure.tree_structure import sequence_nodes
from r_u_sure.tree_structure import transforms
from r_u_sure.wrappers import parser_tools
from r_u_sure.wrappers import wrapper_base

PackedSequenceNodeStorage = packed_sequence_nodes.PackedSequenceNodeStorage
DualDecompositionDiagramSystem = (
    consistent_path_dual_solver.DualDecompositionDiagramSystem
)
ContextInfo = Any


class EditDistanceWrapperBase(wrapper_base.HighLevelUtilityWrapper):
  """Base class for UncertaintyRegionsWrapper and PrefixByEditDistanceWrapper."""

  show_start_editing_marker = True

  def __init__(
      self,
      utility_config: edit_dags.TrustRegionUtilityParameters,
      use_numba: bool = True,
      rewrite_states: bool = True,
  ):
    """Constructs a EditDistanceWrapperBase.

    Should be overridden.

    Args:
      utility_config: Config for the utility function.
      use_numba: Whether to use Numba to accelerate computation.
      rewrite_states: If using Numba, whether to rewrite states to be integers,
        which is slightly faster.
    """
    self._utility_config = utility_config
    self._use_numba = use_numba
    if use_numba:
      self._pack_edit_dag, self._edit_dag_converter = (
          edit_dags.make_specialized_dag_packer(
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
      self._pack_edit_dag = edit_dags.pack_dag
      self._edit_dag_converter = lambda dag: dag
      self._pack_constraint_dag = constraint_dags.pack_dag
      self._constraint_dag_converter = lambda dag: dag

    self._construct_edit_dag = edit_dags.make_edit_dag_builder(
        with_numba=use_numba
    )
    self._construct_constraint_dag = (
        constraint_dags.make_constraint_dag_builder(with_numba=use_numba)
    )

  def _process_prototype_and_context_maybe_exit(
      self,
      context_and_prototype: list[sequence_nodes.SequenceNode],
      prediction_location: int,
      can_insert_low_confidence: bool,
      can_early_exit: bool,
  ) -> tuple[PackedSequenceNodeStorage, ContextInfo]:
    """Processes a prototype sequence, maybe inserting early exit.."""
    sequence = context_and_prototype
    if can_insert_low_confidence:
      sequence = transforms.insert_region_options_around_subsequences(
          sequence,
          allow_empty_regions=True,  # to support insertion points
          node_filter=parser_tools.allow_regions_around_pseudoparse_node,
      )
    if can_early_exit:
      sequence = transforms.insert_early_exit(sequence)
    sequence = transforms.truncate_prefix_at_offset(
        sequence, prediction_location
    )
    packed = parser_tools.pack_sequence_from_pseudoparser(
        sequence, with_numba=self._use_numba
    )
    context_info = {
        "context_prefix_contents": sequence_node_helpers.render_text_contents(
            context_and_prototype
        )[:prediction_location]
    }
    return packed, context_info

  def process_target(
      self,
      context_and_target: list[sequence_nodes.SequenceNode],
      prediction_location: int,
  ) -> PackedSequenceNodeStorage:
    """See HighLevelUtilityWrapper.process_target."""
    sequence = transforms.truncate_prefix_at_offset(
        context_and_target, prediction_location
    )
    packed = parser_tools.pack_sequence_from_pseudoparser(
        sequence, with_numba=self._use_numba
    )
    return packed

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

    # Edit dags
    for target, scale_factor in zip(targets, target_utility_scale_factors):
      dag, _ = self._construct_edit_dag(
          prototype=prototype, target=target, parameters=self._utility_config
      )
      reachable_dag = self._prune_to_reachable(dag)
      packed_dag, conversion_data = self._pack_edit_dag(reachable_dag)
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
      assert len(system.data.dag_tables) == 2  # edit dag, constraint dag
      packed_dag = system.data.dag_tables[0].dag
      conversion_data = system.conversion_data[0]
      path, _ = packed_dags.constrained_best_path(
          packed_dag, conversion_data, assignments
      )
      system_cost = sum(edge.cost for edge in path)
      result["total_cost"] = system_cost

      evaluation_summary_metrics = edit_dags.extract_edit_summary_metrics(
          path=path, prototype=prototype, target=evaluation_target
      )
      result.update(evaluation_summary_metrics)

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
    common_prefix = context_info["context_prefix_contents"]
    return (
        '<span style="white-space: pre; font-family: monospace; color:'
        ' #72A0C1;">'
        + html.escape(common_prefix)
        + "</span>"
        + region_decisions.render_regions_to_html(prototype, assignments)
        + "<br>"
    )

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
    common_prefix = context_info["context_prefix_contents"]

    # Assumption (based on `build_system` above): the system contains all
    # of the edit dags in order, then one constraint dag. We just need the edit
    # dags.
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
          + html.escape(common_prefix)
          + "</span>"
          + edit_dags.extract_edit_sequence_html(
              path,
              prototype=prototype,
              target=target,
              start_editing_marker=(
                  "፠" if self.show_start_editing_marker else ""
              ),
          )
          + "<br>"
      )

    return "".join(source)
