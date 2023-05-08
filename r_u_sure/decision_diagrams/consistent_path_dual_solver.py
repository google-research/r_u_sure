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

"""Dual decomposition solver for (packed) state DAGs.

This module implements the dual decomposition algorithm described in Section 3
and Appendix B of our paper. Equivalently, it can be seen as a solver for the
"consistent path problem" discussed in

  Lozano, Leonardo, David Bergman, and J. Cole Smith.
  "On the consistent path problem." Operations Research 68.6 (2020): 1913-1931.
  http://www.optimization-online.org/DB_FILE/2018/03/6540.pdf

This solver specifically is designed to quickly solve the dual Lagrangian
relaxation of the consistent path problem, a special case of the general
concept of dual decomposition. Specifically, it is a generalization of the
min-marginal averaging approach described in

  Lange, Jan-Hendrik, and Paul Swoboda. "Efficient message passing for 0–1 ILPs
  with binary decision diagrams." International Conference on Machine Learning.
  PMLR, 2021.
  https://arxiv.org/pdf/2009.00481.pdf

The generalization is that we support categorical state DAGs instead of just
binary decision diagrams. In particular, we allow decisions to be choices over
a small finite set (beyond just 0, 1), and we allow auxiliary edges that exist
"within" a single layer of the BDD that are more efficient to compute (instead
of requiring that there is exactly one arc labeled 0 and one arc labeled 1 from
each node). The finite set restriction can be seen as an implicit set of
indicator variables for each choice, and the auxiliary variables lift our
algorithm to operate on *possibly-ambiguous nondeterministic* decision diagrams.
See Appendix B of our paper for details.
"""
import dataclasses
import enum
import time
from typing import NamedTuple, Optional, Union

import numba

import numpy as np
import numpy.typing  # pylint: disable=unused-import

from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.decision_diagrams import packed_dags


class SweepDirection(enum.Enum):
  """Direction for a sweep through subproblem variables."""

  FORWARD = enum.auto()
  REVERSE = enum.auto()


class DualDecompositionDiagramSystemComputationData(NamedTuple):
  """Necessary state for running dual decomposition on a set of DAGs.

  Note that Numpy arrays are passed around by reference. By convention, users
  of a DualDecompositionDiagramSystem are allowed to modify the penalties for
  each of the dag_tables, but can only do so using `packed_dags.set_penalties`
  to ensure that the prefix/suffix states are consistent. Users also should NOT
  modify the dags themselves (e.g. dag_tables[i].dag) or the index map.

  Users are generally expected to maintain the invariant that
  `sum(dag_table.penalties for dag_table in dag_tables) == extra_unary_costs`
  To modify `extra_unary_costs`, use `constrain_system` or other functions in
  this file.

  Attributes:
    dag_tables: List of tables for computing min marginals of all of the DAGs we
      are solving.
    dag_variable_index_map: NDArray of shape [num_variables, num_dags] such that
      if k = dag_variable_index_map[i, d] and k >= 0, then variable_keys[i] ==
      conversion_data[d].variable_values[k] and if k = -1, then variable_keys[i]
      is not in conversion_data[d].variable_values.  In other words,
      `dag_variable_index_map` tells us where to find each of our variables in
      each of our DAGs. If every factor uses every variable, this will just be a
      matrix of identical increasing sequences like [[0, 1, 2, ...], [0, 1, 2,
      ...], ...]. If some factors skip some variables, these increasing
      sequences will desynchronize with each other.
    dag_update_proportions: NDArray of shape [num_dags], determining how to
      distribute min-marginals across subproblems. Weights should be positive
      but do not have to sum to one; they will be normalized for each variable
      over the set of subproblems that assign that variable.
    extra_unary_costs: NDArray of shape [num_variable_keys, num_variable_values]
      with extra costs for individual variable assignments. Can be used to
      enforce particular choices.
    num_variable_keys: Number of variable keys.
    num_variable_values: Number of variable values.
  """

  dag_tables: list[packed_dags.PenalizedShortestPrefixSuffixTables]
  dag_variable_index_map: np.typing.NDArray[np.int64]
  dag_update_proportions: np.typing.NDArray[np.float64]
  extra_unary_costs: np.typing.NDArray[np.float64]
  num_variable_keys: int
  num_variable_values: int


@dataclasses.dataclass
class DualDecompositionDiagramSystem:
  """Data and metadata about a system of decision diagrams.

  Attributes:
    data: Information necessary to run min-marginal computations on the system.
    variable_keys: List of all variables we are solving over. Should be the
      sorted union of the variables used in all of the DAGs in the system.
    variable_values: List of variable values for those variables. Should be the
      same as `variable_values` for every DAG in the system.
    conversion_data: List of conversion data objects used when creating the
      system.
  """

  data: DualDecompositionDiagramSystemComputationData
  variable_keys: list[gated_state_dag.SharedVariableKey]
  variable_values: list[gated_state_dag.SharedVariableValue]
  conversion_data: list[packed_dags.PackedStateDAGConversionData]


@numba.njit
def _numba_extract_keys(
    current_variable_keys: numba.typed.List[gated_state_dag.SharedVariableKey],
    accum_variable_key_map: Union[
        numba.typed.Dict, dict[gated_state_dag.SharedVariableKey, bool]
    ],
    accum_variable_keys: numba.typed.List[gated_state_dag.SharedVariableKey],
) -> None:
  """Helper function to extract unseen keys."""
  for key in current_variable_keys:
    if key not in accum_variable_key_map:
      accum_variable_keys.append(key)
      accum_variable_key_map[key] = True


@numba.njit
def _numba_populate_variable_index_map(
    variable_keys: numba.typed.List[gated_state_dag.SharedVariableKey],
    dag_variable_index_map: np.typing.NDArray[np.int64],
    dag_index: int,
    variable_key_index_from_key: Union[
        numba.typed.Dict, dict[gated_state_dag.SharedVariableKey, int]
    ],
) -> None:
  """Helper function to extract unseen keys."""
  for i, variable in enumerate(variable_keys):
    variable_key_index = variable_key_index_from_key.get(variable, -1)
    dag_variable_index_map[i, dag_index] = variable_key_index


def make_system(
    converted_dags: list[
        tuple[
            packed_dags.PackedStateDAG, packed_dags.PackedStateDAGConversionData
        ]
    ],
    dag_update_proportions: Optional[np.typing.NDArray[np.float64]] = None,
) -> DualDecompositionDiagramSystem:
  """Constructs a system from a set of DAGs.

  Args:
    converted_dags: List of DAGs to combine. All DAGs must have exactly the same
      set of possible variable values.
    dag_update_proportions: NDArray of shape [num_dags], determining how to
      distribute min-marginals across subproblems. Weights should be positive
      but do not have to sum to one; they will be normalized for each variable
      over the set of subproblems that assign that variable.

  Returns:
    A system with empty penalty tables.
  """
  if not converted_dags:
    raise ValueError("Must provide at leats one DAG to make a system.")

  if dag_update_proportions is None:
    dag_update_proportions = np.ones((len(converted_dags),), np.float64)

  variable_values = converted_dags[0][1].variable_values
  for _, conversion_data in converted_dags[1:]:
    if conversion_data.variable_values != variable_values:
      raise ValueError(
          "DAGs must have the same set of variable values to build a system!"
      )

  if isinstance(converted_dags[0][1].variable_keys, numba.typed.List):
    # Numba version of below logic.
    key_type = converted_dags[0][1].variable_keys._numba_type_.item_type  # pylint: disable=protected-access
    variable_key_map = numba.typed.Dict.empty(key_type, numba.typeof(True))
    variable_keys = numba.typed.List.empty_list(key_type)
    for _, conversion_data in converted_dags:
      _numba_extract_keys(
          conversion_data.variable_keys, variable_key_map, variable_keys
      )

    variable_keys.sort()
    dag_variable_index_map = np.empty(
        (len(variable_keys), len(converted_dags)), np.int64
    )
    for d, (_, conversion_data) in enumerate(converted_dags):
      _numba_populate_variable_index_map(
          variable_keys,
          dag_variable_index_map,
          d,
          conversion_data.variable_key_index_from_key,
      )
  else:
    # Each DAG already has its variables in sorted order. Instead of throwing
    # away this order right away, we take advantage of Timsort's detection of
    # partially sorted subruns, and keep things in sorted order as much as we
    # can.
    variable_key_set = set()
    variable_keys = []
    for _, conversion_data in converted_dags:
      for key in conversion_data.variable_keys:
        if key not in variable_key_set:
          variable_key_set.add(key)
          variable_keys.append(key)
    variable_keys.sort()
    # An alternative implementation would just be to set
    # variable_keys = sorted(variable_key_set)
    # but this would throw away the order that we already know exists.

    dag_variable_index_map = np.empty(
        (len(variable_keys), len(converted_dags)), np.int64
    )
    for i, variable in enumerate(variable_keys):
      for d, (_, conversion_data) in enumerate(converted_dags):
        if variable in conversion_data.variable_key_index_from_key:
          k = conversion_data.variable_key_index_from_key[variable]
          dag_variable_index_map[i, d] = k
        else:
          dag_variable_index_map[i, d] = -1

  # We always start with zero unary costs.
  extra_unary_costs = np.zeros(
      (len(variable_keys), len(variable_values)), np.float64
  )
  dag_tables = [
      packed_dags.empty_unpenalized_table(dag) for dag, _ in converted_dags
  ]
  return DualDecompositionDiagramSystem(
      variable_keys=variable_keys,
      variable_values=variable_values,
      conversion_data=[
          conversion_data for _, conversion_data in converted_dags
      ],
      data=DualDecompositionDiagramSystemComputationData(
          dag_tables=numba.typed.List(dag_tables),
          dag_variable_index_map=dag_variable_index_map,
          dag_update_proportions=dag_update_proportions,
          extra_unary_costs=extra_unary_costs,
          num_variable_keys=len(variable_keys),
          num_variable_values=len(variable_values),
      ),
  )


def clone_system(
    system: DualDecompositionDiagramSystem,
) -> DualDecompositionDiagramSystem:
  """Makes a copy of a system with different penalties and unary costs but a shared DAG.

  This can be used when we want to try manipulating penalties in a different
  way without clobbering the current penalties, for instance, in order to
  extract a solution.

  Args:
    system: A system to copy.

  Returns:
    An equivalent system with non-shared penalties and unary costs.
  """
  return DualDecompositionDiagramSystem(
      variable_keys=system.variable_keys,
      variable_values=system.variable_values,
      conversion_data=system.conversion_data,
      data=clone_system_data(system.data),
  )


@numba.extending.register_jitable
def clone_system_data(
    system_data: DualDecompositionDiagramSystemComputationData,
) -> DualDecompositionDiagramSystemComputationData:
  """Makes a copy of a system's data with different penalties and unary costs but a shared DAG.

  Args:
    system_data: Data to copy.

  Returns:
    An equivalent system with non-shared penalties.
  """
  return DualDecompositionDiagramSystemComputationData(
      dag_tables=_clone_memos(system_data.dag_tables),
      dag_variable_index_map=system_data.dag_variable_index_map,
      dag_update_proportions=system_data.dag_update_proportions,
      extra_unary_costs=np.copy(system_data.extra_unary_costs),
      num_variable_keys=system_data.num_variable_keys,
      num_variable_values=system_data.num_variable_values,
  )


@numba.njit
def _clone_memos(
    memos: numba.typed.List[packed_dags.PackedStateDAG],
) -> numba.typed.List[packed_dags.PackedStateDAG]:
  """Helper function to copy a set of memos."""
  result_list = numba.typed.List()
  for memo in memos:
    result_list.append(packed_dags.copy_memo(memo))
  return result_list  # pytype: disable=bad-return-type


@numba.njit
def system_min_marginals(
    system_data: DualDecompositionDiagramSystemComputationData,
    variable_index: int,
) -> np.typing.NDArray[np.float64]:
  """Aggregates min marginals for a particular variable in the system."""

  total_min_marginals = np.zeros((system_data.num_variable_values,), np.float64)
  subprob_var_indices = system_data.dag_variable_index_map[variable_index, :]
  for subproblem_index, sub_variable_index in enumerate(subprob_var_indices):
    if sub_variable_index == -1:
      # This variable doesn't participate in this subproblem.
      # However, we still want to account for its cost in the min marginals.
      subproblem_cost = packed_dags.compute_minimal_cost(
          system_data.dag_tables[subproblem_index]
      )
      total_min_marginals += subproblem_cost
    else:
      min_marginals = packed_dags.compute_min_marginals_for_variable(
          memo=system_data.dag_tables[subproblem_index],
          variable_index=sub_variable_index,
      )
      total_min_marginals += min_marginals

  return total_min_marginals


@numba.njit
def min_marginal_average(
    system_data: DualDecompositionDiagramSystemComputationData,
    variable_index: int,
) -> float:
  """Performs min-marginal averaging for a single variable in the system.

  The min-marginal averaging algorithm is described in

    Lange, Jan-Hendrik, and Paul Swoboda. "Efficient message passing for 0–1
    ILPs with binary decision diagrams." International Conference on Machine
    Learning. PMLR, 2021.
    https://arxiv.org/pdf/2009.00481.pdf

  It is also conceptually the same as the "max-marginal averaging" approach
  in Section 6.4 of

    Werner, Tomas, Daniel Prusa, and Tomas Dlask. "Relative interior rule in
    block-coordinate descent." Proceedings of the IEEE/CVF Conference on
    Computer Vision and Pattern Recognition. 2020.
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Werner_Relative_Interior_Rule_in_Block-Coordinate_Descent_CVPR_2020_paper.pdf

  This same algorithm has likely been described in a variety of other settings.

  Note that the proof that this is a valid coordinate ascent step, in
  Section 3.2 and Appendix A.2, actually holds for any set of weights that sums
  to 1, not just 1/|num subproblems|. So we implement a generalization that
  allows the min marginals to be redistributed to subproblems in unequal ways.
  Note that this generalization is also proved to be valid in

    Abbas, Ahmed, and Paul Swoboda. "DOGE-Train: Discrete Optimization on GPU
    with End-to-end Training." arXiv preprint arXiv:2205.11638 (2022).
    https://arxiv.org/pdf/2205.11638.pdf

  We also implement a generalization in that we allow more than just binary
  variables. This is handled by treating a single categorical variable (over K
  choices) as a set of K binary indicator variables. However, since we know
  exactly one of these indicator variables will be set, and they will all be
  set simultaneously, we are free to reorder these virtual indicator variables
  on the fly. We thus choose to reorder them to make computation convenient.
  In particular, by first updating penalties for all indicator variables that
  are NOT the best (local) choice for the current variable, we can update them
  in parallel, and by the time we get to the best choice, all subproblems
  already agree.

  Finally, our implementation also supports *nondeterministic* decision
  diagrams, for which there are some edges that do not have decisions, and we
  allow multiple outgoing edges from a node with the same decision. However,
  this is all handled by the PenalizedShortestPrefixSuffixTables implementation,
  and we don't have to explicitly worry about that here; all we need is an
  efficient way to compute min marginals for the sequence of variables.

  Args:
    system_data: Computation-relevant info for the system we are solving.
    variable_index: The variable index to update. Note that, due to the
      prefix/suffix table computations, the efficiency of this update is roughly
      proportional to the difference between `variable_index` in consecutive
      calls.

  Returns:
    Updated dual bound after this iteration, extracted from the min marginals.
  """
  # Roughly, we let
  #   J: The set of subproblems.
  #   I = {(key, value) : key \in keys, value \in values}: A set of indicator
  #       variables for each possible variable assignment. We treat an
  #       assignment of `key := value` as an assignment of `(key, value) := 1`,
  #       which reduces our problem to one over binary decision diagrams, and
  #       simplifies notation.
  #   X = {0, 1}^I: The set of all assignments to all indicator variables. Note
  #       that, in practice, exactly one indicator will be set for each key, but
  #       we don't make that explicit in the notation.
  #   f_j(x): The cost of assignment vector `x \in X` for subproblem `j \in J`.
  #       This can be infinite, and in particular will always be infinite for
  #       `x`s that assign no value or more than one value for any key.
  #
  # We want to minimize
  #
  #   C(x) =   \sum     f_j(x)
  #           j \in J
  #
  # We do this approximately by constructing a dual problem of the form
  #
  #                     [                         [                 ] ]
  #   L(x, u) =  \sum   [ f_j(x[j, :]) +   \sum   [ u[j, i] x[j, i] ] ]
  #             j \in J [                 i \in I [                 ] ]
  #
  # where the `f_j` are the DAGs, and the `u[j, i]` are the penalty vectors
  # associated with the prefix/suffix tables, under the restriction that
  # `\sum_j u[j,i] = 0` for all `j`.
  # We then perform block coordinate ascent on the slice u[:, i] for each of
  # the indicators {(current_key, possible_value) : possible_value \in values}.
  # Any update that causes the x[:, i] to agree (e.g. that causes all
  # subproblems to choose to assign x[j, i] to either 0 or 1) is a valid update
  # step.
  #
  # How should we divide the penalties up to make them agree on this assignment?
  # We use a variant of min-marginal averaging; intuitively, we assign penalties
  # so that the relative preferences of all subproblems become the same.
  #
  # More specifically, let
  #
  #   L_j(x, u) = f_j(x) + \sum_i u[i]x[i]
  #
  # be the Lagrangian-penalized version of subproblem `j`, and let
  #
  #   m0[j,i] = \min_x L_j(x, u) s.t. x[j,i] = 0
  #   m1[j,i] = \min_x L_j(x, u) s.t. x[j,i] = 1
  #
  # be the min-marginals for this problem, e.g. the cost of the minimum
  # penalized assignment that assigns indicator `i` to 0 or 1. Note that the
  # difference between these is roughtly how much subproblem `j` "cares" about
  # setting `i` to 0 or 1.
  #
  # We wish to set penalties such that
  #
  #   (m1[j,i] - m0[j,i]) / weight[j] == (m1[j',i] - m0[j',i]) / weight[j']
  #
  # for all j, j', e.g. we want to distribute min marginal preferences such that
  # every subproblem makes the same choice, and the min marginals are
  # distributed proportional to their weight (or, if weights are all 1, that
  # they all have equal preferences).
  #
  # Here we can make use of our prior knowledge that every graph assigns exactly
  # one value to each key. Our packed state DAG solver computes categorical
  # min-marginals instead of indicator min marginals, e.g. it computes
  #
  #   M[j, k, v] = \min_x L_j(x, u) s.t. x[j,(k,v)] = 1, and
  #                                      x[j,(k,v')] = 0 for v' != v
  #
  # where `k` is a key, `v` is a value, and `(k,v) \in I` is an indicator.
  #
  # If we let
  #
  #   vj*  = \argmin_v M[j, k, v]           # (the best value for `k`)
  #   vj*' = \argmin_{v != vj*} M[j, k, v]  # (the second best value)
  #
  # we then have
  #
  #   m1[j,(k,v)] = M[j, k, v]
  #   m0[j,(k,v)] = | M[j, k, vj*']    if v = vj*,
  #                 | M[j, k, vj*]     if v != vj*
  #
  # Our strategy for making all of the (m1[j,(k,v)] - m0[j,(k,v)]) match is
  # to set
  #
  #   u[j,(k,v)] := u[j,(k,v)] + (
  #       M+[k, v] * weight[j] / (\sum_j weight[j])
  #       - M[j, k, v]
  #   )
  #
  # where M+[k, v] = \sum_j M[j, k, v].
  # This both maintains the sum-to-zero constraint on `u` and ensures that
  # after the update, we have
  #
  #   M[j, k, v] := M+[k, v] * weight[j] / (\sum_j weight[j]).
  #
  # We can then see that the desired property also holds for the indicator
  # representation, because for all `j` we have
  #
  #   vj*  = v*  = \argmin_v M+[k, v]
  #   vj*' = v*' = \argmin_{v != v*} M+[k, v]
  #
  # Note that, for a fixed `k` and `v`, `M+[k,v]` has the same value regardless
  # of the values of `u[j,(k,v)]` across `j`, since the constraint that
  # `\sum_j u[j,(k,v)] = 0` allows us to cancel out all of the terms that
  # mention `(k,v)`. We thus "zero out" all of the `u[j,(k,v)]` when aggregating
  # min marginals, which allows us to simply set
  #
  #   u[j,(k,v)] := (
  #       M+[k, v] * weight[j] / (\sum_j weight[j])
  #       - M[j, k, v]
  #   )
  #
  # in the second step.
  #
  # In practice we actually implement a slightly more complex version of this,
  # where we want to solve
  #
  #   C(x) =  extra_unary_costs(x)  +  \sum     f_j(x)
  #                                   j \in J
  #
  # The main change is that, instead of summing to zero, the min marginals
  # have to sum to the value given in `extra_unary_costs`.
  # (`extra_unary_costs` is used to solve problems under individual variable
  # constraints.)

  cost_for_non_participating_subproblems = 0.0

  # Implementation: We start by computing \sum_j M[j, k, v]
  total_min_marginals = np.zeros((system_data.num_variable_values,), np.float64)
  total_weight = 0.0
  subprob_var_indices = system_data.dag_variable_index_map[variable_index, :]
  subprob_info = []
  for subproblem_index, sub_variable_index in enumerate(subprob_var_indices):
    if sub_variable_index == -1:
      # This variable doesn't participate in this subproblem.
      # But we should still compute costs for this subproblem, which we will
      # use when computing the new dual bound.
      subproblem_cost = packed_dags.compute_minimal_cost(
          system_data.dag_tables[subproblem_index]
      )
      cost_for_non_participating_subproblems += subproblem_cost
      continue

    # Zero out penalties for this variable.
    packed_dags.set_penalties(
        memo=system_data.dag_tables[subproblem_index],
        variable_index=sub_variable_index,
        penalties=np.zeros_like(total_min_marginals),
    )
    min_marginals = packed_dags.compute_min_marginals_for_variable(
        memo=system_data.dag_tables[subproblem_index],
        variable_index=sub_variable_index,
    )
    total_min_marginals += min_marginals
    weight = system_data.dag_update_proportions[subproblem_index]
    total_weight += weight
    subprob_info.append(
        (subproblem_index, sub_variable_index, min_marginals, weight)
    )

  # Since we zeroed-out the penalties for all of our subproblems, they no longer
  # sum to `extra_unary_costs`, so we need to add that back in.
  # We will redistribute this cost to the penalty tables below.
  total_min_marginals += system_data.extra_unary_costs[variable_index, :]

  # We next update the penalties to make them agree.
  for (
      subproblem_index,
      sub_variable_index,
      min_marginals,
      weight,
  ) in subprob_info:
    desired_min_marginals = total_min_marginals * (weight / total_weight)
    # special case: if the aggregate min marginals are infinite, set penalties
    # to infinite for those assignments, "evenly dividing" infinity among the
    # factors.
    penalty_update = desired_min_marginals - np.where(
        np.isinf(desired_min_marginals),
        0,
        min_marginals,
    )
    # The below only works because we set the penalties to zero before
    # computing min marginals above! Conceptually this is
    #   new_penalties = old_penalties + penalty_update
    new_penalties = penalty_update
    packed_dags.set_penalties(
        memo=system_data.dag_tables[subproblem_index],
        variable_index=sub_variable_index,
        penalties=new_penalties,
    )

  # All subproblems now agree (in a weighted sense) on the min marginals,
  # and in particular agree on what the best choice for this variable is.
  # Our new dual bound is the min marginal of making that choice, plus the
  # best cost for all subproblems that don't make any choice here.
  best_cost = (
      np.min(total_min_marginals) + cost_for_non_participating_subproblems
  )
  return best_cost


class MinMarginalSweepSolverResults(NamedTuple):
  """Results from the min-marginal-averaging solver.

  Attributes:
    objective_at_step: Breakdown of dual ascent objective at each step.
    variable_at_step: Which variable we updated at each step.
    objective_at_sweep: Summaries of dual ascent objective after running each
      sweep, which is a set of consecutive updates of variables in either a
      forward or reverse order.
    steps_at_sweep: Cumulative number of steps after running each sweep.
    time_at_sweep: Total runtime after running each sweep.
    reached_plateau: Whether we reached a point where the dual objective stopped
      improving for an entire sweep.
  """

  objective_at_step: np.typing.NDArray[np.float64]
  variable_at_step: np.typing.NDArray[np.int64]
  objective_at_sweep: np.typing.NDArray[np.float64]
  steps_at_sweep: np.typing.NDArray[np.int64]
  time_at_sweep: np.typing.NDArray[np.float64]
  reached_plateau: bool


@numba.njit
def solve_system_with_sweeps(
    system_data: DualDecompositionDiagramSystemComputationData,
    soft_timeout: float = np.inf,
    tolerance: float = 1e-5,
) -> MinMarginalSweepSolverResults:
  """Heuristic block-coordinate-ascent solver using min marginal averaging.

  This function attempts to maximize the dual Lagrangian objective (the lower
  bound on the true primal objective) with respect to the penalties (a.k.a. the
  Lagrange multipliers or "messages"). It uses `min_marginal_average` to perform
  individual block-coordinate-ascent steps, and uses a set of heuristics to
  determine which variables to average min marginals for in which order.

  The sweep heuristic is designed to take advantage of the structure of the
  min-marginal computations of PenalizedShortestPrefixSuffixTables, which are
  fastest for computing min-marginals for variables that are adjacent to the
  previous variable that we computed min-marginals (and updated penalties) for.
  Thus, the high-level approach is to first process variables in a forward
  direction, then process them in the reverse direction. In other words, if
  there are 5 variables, we might update them in order

      (forward)   (reverse)   (forward)
    1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, ...

  The solver stops when the dual objective stops improving for an entire pair
  of sweeps, or once a timeout is reached. (I'm not sure whether
  or not this always indicates we have reached a maximum of the dual objective;
  it's possible that the algorithm can get "stuck" in a fixed point, or that
  it isn't even in a fixed point and the messages will eventually lead to a
  decrease.)

  Args:
    system_data: Data for the system we are solving.
    soft_timeout: Maximum time to run a search for. If we have not converged by
      this point, we will exit after the first sweep that exceeds the timeout.
      (Note that this will likely happen slightly after the timeout itself.)
    tolerance: The amount of improvement that we consider significant for
      terminating the search.

  Returns:
    A variety of information about the progress of the search.
  """
  objective_at_step = numba.typed.List()
  variable_at_step = numba.typed.List()
  objective_at_sweep = numba.typed.List()
  steps_at_sweep = numba.typed.List()
  times_at_sweep = numba.typed.List()

  with numba.objmode(base_time=numba.float64):
    # perf_counter must be called in pure-Python mode.
    base_time = time.perf_counter()

  # How many steps have we run?
  step_count = 0
  # What was the last bound we saw?
  last_dual_bound = -np.inf

  # Keep track of how many sweeps we have run without seeing any change.
  reached_plateau = False

  while True:
    saw_any_change = False

    for sweep_direction in (SweepDirection.FORWARD, SweepDirection.REVERSE):
      if sweep_direction == SweepDirection.FORWARD:
        # 0, 1, ..., len-3, len-2
        variables_in_sweep = range(0, system_data.num_variable_keys - 1)
      else:
        # len-1, len-2, ..., 2, 1
        variables_in_sweep = range(system_data.num_variable_keys - 1, 0, -1)

      # Run the sweep.
      for current_variable in variables_in_sweep:
        new_dual_bound = min_marginal_average(system_data, current_variable)
        did_improve = (new_dual_bound - last_dual_bound) > tolerance
        last_dual_bound = new_dual_bound

        if did_improve:
          saw_any_change = True

        objective_at_step.append(new_dual_bound)
        variable_at_step.append(current_variable)
        step_count += 1

      # Sweep-level metadata.
      with numba.objmode(stamp=numba.float64):
        # perf_counter must be called in pure-Python mode.
        stamp = time.perf_counter()
      objective_at_sweep.append(last_dual_bound)
      steps_at_sweep.append(step_count)
      times_at_sweep.append(stamp - base_time)
      if stamp - base_time > soft_timeout:
        break

    # Check for exit criteria.
    if not saw_any_change:
      reached_plateau = True
      break

  return MinMarginalSweepSolverResults(
      objective_at_step=np.asarray(objective_at_step),
      variable_at_step=np.asarray(variable_at_step),
      objective_at_sweep=np.asarray(objective_at_sweep),
      steps_at_sweep=np.asarray(steps_at_sweep),
      time_at_sweep=np.asarray(times_at_sweep),
      reached_plateau=reached_plateau,
  )


@numba.njit
def greedy_extract(
    system_data: DualDecompositionDiagramSystemComputationData,
    direction: SweepDirection,
) -> tuple[np.typing.NDArray[np.int64], float]:
  """Greedily extracts a primal solution.

  Iterates through variables one at a time, and fixes them to the value that
  has the best min-marginal, when allowing all not-yet-fixed variables to take
  different values in different subproblems.

  Args:
    system_data: Data for the system to extract. Will be copied to avoid
      destroying any penalties that are already present.
    direction: The direction to sweep when greedy decoding.

  Returns:
    A tuple (assignment_vector, cost), where assignment_vector maps each
    variable index to a value index, and cost is the cost of the returned
    solution.
  """
  # Avoid clobbering all the penalties by making a copy.
  system_data = clone_system_data(system_data)

  assignments = np.full((system_data.num_variable_keys,), -1, dtype=np.int64)
  if direction == SweepDirection.FORWARD:
    # 0, 1, ..., len-3, len-2, len-1
    iteration_order = range(0, system_data.num_variable_keys)
  else:
    # len-1, len-2, ..., 2, 1, 0
    iteration_order = range(system_data.num_variable_keys - 1, -1, -1)

  last_cost = np.nan
  for variable in iteration_order:
    min_marginals = system_min_marginals(system_data, variable)
    choice = np.argmin(min_marginals)
    assignments[variable] = choice
    last_cost = min_marginals[choice]

    # Enforce this choice by updating `extra_unary_costs`. Note that we still
    # have to keep the other penalties consistent as well.
    extra_unary_costs_here = np.full((system_data.num_variable_values,), np.inf)
    extra_unary_costs_here[choice] = system_data.extra_unary_costs[
        variable, choice
    ]
    system_data.extra_unary_costs[variable] = extra_unary_costs_here

    subprob_var_indices = system_data.dag_variable_index_map[variable, :]
    for subproblem_index, sub_variable_index in enumerate(subprob_var_indices):
      if sub_variable_index == -1:
        # This variable doesn't participate in this subproblem.
        continue

      # Update all penalties to force this choice to be made.
      new_penalties = np.full((system_data.num_variable_values,), np.inf)
      new_penalties[choice] = system_data.dag_tables[
          subproblem_index
      ].penalties[sub_variable_index, choice]
      packed_dags.set_penalties(
          memo=system_data.dag_tables[subproblem_index],
          variable_index=sub_variable_index,
          penalties=new_penalties,
      )

  # At this point we have made a decision for every variable.
  return assignments, last_cost


class PartialCommitmentStatus(enum.Enum):
  """Result of partial commitment."""

  MADE_PARTIAL_ASSIGNMENT = enum.auto()
  NEEDS_GREEDY_EXTRACT = enum.auto()
  STUCK = enum.auto()


def partially_commit(
    system_data: DualDecompositionDiagramSystemComputationData,
    direction: SweepDirection,
    target_commitments_per_variable: float,
    gap_strictness: float,
    epsilon: float = 1e-6,
    verbose: bool = False,
) -> PartialCommitmentStatus:
  """Partially commits to a primal solution by forbidding bad assignments.

  This function identifies the variable assignments that are known to lead to
  bad solutions (because they have low min marginals), and forbids them. At
  a high level, the idea is to sort variable assignments by their aggregated
  min marginals, then prevent any assignment whose min marginal is too small.

  One complication is that the gaps can change as we iterate through variables.
  If we are not careful, we can end up committing to an infeasible assignment
  (one with infinite cost). To prevent this, we don't threshold based on an
  absolute value, but instead based on how suboptimal they are relative to our
  current best guess about the optimal solution, which we update as we make
  assignments.

  One interesting property to be aware of: For a single subproblem, the
  minimum min-marginal for each variable is equal to the minimum-cost solution
  of that subproblem (since there's at least one choice for every variable that
  attains the minimum cost). However, when we aggregate min marginals over
  subproblems, it is NO LONGER the case that the minimum aggregated min
  marginals are the same for all variables, since each aggregated min marginal
  vector essentially corresponds to constraining a different variable to match
  over the subproblems. All of these are valid bounds for our ultimate primal
  objective, so we take the LARGEST such bound, which will be the most tight.
  Note, however, that if we are already at a stationary point of the
  min-marginal averaging procedure, then all samples agree on all min marginals,
  which DOES imply that the minimum aggregated min marginal is the same for
  all variables. This property may hold at first, but it breaks as soon as we
  start changing the penalties.

  Using a smaller value of `target_commitments_per_variable` and a larger value
  of `gap_strictness` means we fix a smaller number of variables. This
  sometimes leads to better solutions but sometimes leads to worse solutions;
  the largest-gap heuristic may not be a good one for choosing variables in
  general.

  Args:
    system_data: Data for the system to apply penalties to.
    direction: The direction to process the system in.
    target_commitments_per_variable: Target number of assignments to forbid for
      each variable, on average. Can be less than one to only choose a subset of
      variables. Even if greater than one, this function may choose to forbid
      two bad options for a single variable and keep all options possible for
      another variable.
    gap_strictness: How strict to be about the size of the gap. If this is 1, we
      use exactly the gap that would lead to the target number of commitments.
      If this is 0, we will greedily discard ANY assignment that is worse than
      our guess at the optimal solution (e.g. this reduces to `greedy_extract`).
      In general, we throw away values that are farther than `gap_strictness *
      target_gap` away from the optimal solution.
    epsilon: Small value used to determine if floating point values should be
      treated as equal.
    verbose: Whether to print debug messages.

  Returns:
    A status indicator. NEEDS_GREEDY_EXTRACT is returned if the heuristic
    determines that the partial extraction process will be equivalent to greedy
    decoding, allowing actually retrieving the greedy solution.
    MADE_PARTIAL_ASSIGNMENT is returned if the heuristic made progress
    penalizing at least one variable. STUCK is returned if the heuristic didn't
    make any progress; greedy decoding is a good choice in this case as well.
  """
  if direction == SweepDirection.FORWARD:
    # 0, 1, ..., len-3, len-2, len-1
    iteration_order = range(0, system_data.num_variable_keys)
  else:
    # len-1, len-2, ..., 2, 1, 0
    iteration_order = range(system_data.num_variable_keys - 1, -1, -1)

  # Iterate through once to extract our aggregated min marginals.
  aggregated_min_marginals = np.full(
      (system_data.num_variable_keys, system_data.num_variable_values), np.nan
  )

  for variable in iteration_order:
    aggregated_min_marginals[variable, :] = system_min_marginals(
        system_data, variable
    )

  # Compute our best guess at the true solution's cost.
  # We take the tightest lower bound by maximizing over variables. (This step
  # may be unnecessary if we just ran min-marginal averaging.)
  best_cost_bound = np.max(np.min(aggregated_min_marginals, axis=1))
  if verbose:
    print("Cost bound from min marginals:", best_cost_bound)

  # Figure out what to forbid to get the desired amount of target commitments.
  # We do this using quantiles, but ignoring things that we have already
  # forbidden, to make sure we are making progress.
  # Start by replacing inf with -inf:
  prepped_for_quantile = np.where(
      np.isinf(aggregated_min_marginals), -np.inf, aggregated_min_marginals
  )
  # Now figure out which quantile we want. We want a quantile so that
  # `target_commitments_per_variable * num_variable_keys` assignments end up
  # ABOVE the threshold. Since the array is of size
  # `system_data.num_variable_keys * system_data.num_variable_values`
  # we can do this by dividing by the number of variable values.
  q = 1 - target_commitments_per_variable / system_data.num_variable_values
  absolute_threshold = np.quantile(prepped_for_quantile, q)
  if verbose:
    print(
        "Target assignments forbidden:",
        target_commitments_per_variable * system_data.num_variable_keys,
    )
    print("Target quantile:", q)
    print("Threshold:", absolute_threshold)
    print(
        "Estimated number of assignments above threshold:",
        np.sum(prepped_for_quantile > absolute_threshold),
    )
    print(
        "Relative to number of variables:",
        np.sum(prepped_for_quantile > absolute_threshold)
        / system_data.num_variable_keys,
    )
  # Figure out what gap we need.
  if absolute_threshold <= best_cost_bound + epsilon:
    # Meeting this threshold requires us to do greedy decoding.
    return PartialCommitmentStatus.NEEDS_GREEDY_EXTRACT
  else:
    target_gap = max(0, absolute_threshold - best_cost_bound)

  adjusted_gap = target_gap * gap_strictness
  if verbose:
    print("Gap:", target_gap, "adjusted gap:", adjusted_gap)

  count_new_forbidden = 0

  # Iteratively forbid choices.
  for variable in iteration_order:
    min_marginals = system_min_marginals(system_data, variable)
    best_updated_cost_at_variable = np.min(min_marginals)
    if best_updated_cost_at_variable > best_cost_bound:
      # Every assignment to this variable is worse than our old cost bound,
      # so we can update our bound.
      best_cost_bound = best_updated_cost_at_variable

    # Forbid any assignment that exceeds the best cost by a large enough gap.
    forbidden_choices = min_marginals > best_cost_bound + adjusted_gap
    count_new_forbidden += np.sum(
        forbidden_choices & np.isfinite(min_marginals)
    )

    # Update `extra_unary_costs` to account for forbidden choice.
    extra_unary_costs_here = np.full((system_data.num_variable_values,), np.inf)
    extra_unary_costs_here = np.where(
        forbidden_choices, np.inf, system_data.extra_unary_costs[variable, :]
    )
    system_data.extra_unary_costs[variable] = extra_unary_costs_here

    subprob_var_indices = system_data.dag_variable_index_map[variable, :]
    for subproblem_index, sub_variable_index in enumerate(subprob_var_indices):
      if sub_variable_index == -1:
        # This variable doesn't participate in this subproblem.
        continue

      old_penalties = system_data.dag_tables[subproblem_index].penalties[
          sub_variable_index, :
      ]
      new_penalties = np.where(forbidden_choices, np.inf, old_penalties)
      packed_dags.set_penalties(
          memo=system_data.dag_tables[subproblem_index],
          variable_index=sub_variable_index,
          penalties=new_penalties,
      )

  if verbose:
    print("Total choices newly forbidden:", count_new_forbidden)
  if count_new_forbidden:
    return PartialCommitmentStatus.MADE_PARTIAL_ASSIGNMENT
  else:
    return PartialCommitmentStatus.STUCK


def assignments_from_assignment_vector(
    system: DualDecompositionDiagramSystem,
    assignment_vector: np.typing.NDArray[np.int64],
) -> dict[
    gated_state_dag.SharedVariableKey, gated_state_dag.SharedVariableValue
]:
  """Extracts an assignment dictionary from an assignment vector.

  Args:
    system: The system that this assignment vector corresponds to.
    assignment_vector: The vector of assignments, as extracted from
      `greedy_extract`.

  Returns:
    Dictionary mapping variable keys to their values.
  """
  return {
      system.variable_keys[k]: system.variable_values[v]
      for k, v in enumerate(assignment_vector)
  }


def constrain_system(
    system: DualDecompositionDiagramSystem,
    constraints: dict[
        gated_state_dag.SharedVariableKey, gated_state_dag.SharedVariableValue
    ],
    penalty: float = np.inf,
) -> None:
  """Constrains a system so that solutions must make particular assignments.

  Args:
    system: The system to constrain.
    constraints: Dict of constraints to enforce.
    penalty: How much of a penalty to assign to other choices. If `np.inf`, the
      decisions are effectively removed, but this may lead to an unsolvable
      system.
  """
  index_of_variable_key = {k: i for i, k in enumerate(system.variable_keys)}
  index_of_variable_value = {v: i for i, v in enumerate(system.variable_values)}
  for key, value in constraints.items():
    variable_index = index_of_variable_key[key]
    value_index = index_of_variable_value[value]

    penalty_vector = np.full(
        (system.data.num_variable_values,), np.float64(penalty)
    )
    penalty_vector[value_index] = 0.0
    # Update unary penalties.
    system.data.extra_unary_costs[variable_index, :] += penalty_vector

    # Evenly divide the new penalty across relevant subproblems to maintain the
    # invariant.
    subprob_var_indices = system.data.dag_variable_index_map[variable_index, :]
    num_relevant = 0
    for subproblem_index, sub_variable_index in enumerate(subprob_var_indices):
      if sub_variable_index != -1:
        num_relevant += 1

    for subproblem_index, sub_variable_index in enumerate(subprob_var_indices):
      if sub_variable_index == -1:
        # This variable doesn't participate in this subproblem.
        continue

      new_penalties = (
          system.data.dag_tables[subproblem_index].penalties[
              sub_variable_index, :
          ]
          + penalty_vector / num_relevant
      )
      packed_dags.set_penalties(
          memo=system.data.dag_tables[subproblem_index],
          variable_index=sub_variable_index,
          penalties=new_penalties,
      )


def break_symmetry_randomly(
    system: DualDecompositionDiagramSystem,
    noise_scale: float = 1.0,
    random_seed: Optional[int] = None,
) -> None:
  """Breaks symmetry in a system.

  Greedy decoding can get stuck if there are multiple optimal solutions, because
  it does not propagate the constraints after each decision. This means it can
  take a series of choices that each are part of some optimal solution but are
  mutually incompatible with each other.

  While this could in principle be fixed by backtracking, that's somewhat
  complex to implement. Instead, we take advantage of the existing dual
  decomposition marginal propagation logic. We assign random penalties to all
  of the different possible variable assignments, then use dual decomposition
  to propagate these throughout the system. This will hopefully mean that there
  is a unique optimal solution that greedy decoding can find.

  Args:
    system: The system to constrain.
    noise_scale: How much to perturb each decision's cost. If this is being
      combined with a real cost function, the noise scale should be small enough
      to not affect the overall original solution by much. (In practice, we can
      probably expect the standard deviation of the actual perturbations to be
      roughly proportional to the square root of the number of variables, times
      this noise scale.)
    random_seed: Optional random seed to ensure deterministic perturbations.
  """
  if random_seed is None:
    rng = np.random.RandomState()
  else:
    rng = np.random.RandomState(seed=random_seed)

  for variable_index in range(system.data.num_variable_keys):
    penalty_vector = rng.uniform(
        low=0.0, high=noise_scale, size=(system.data.num_variable_values,)
    )
    system.data.extra_unary_costs[variable_index, :] += penalty_vector

    # Evenly divide the new penalty across relevant subproblems to maintain the
    # invariant.
    subprob_var_indices = system.data.dag_variable_index_map[variable_index, :]
    num_relevant = 0
    for subproblem_index, sub_variable_index in enumerate(subprob_var_indices):
      if sub_variable_index != -1:
        num_relevant += 1

    for subproblem_index, sub_variable_index in enumerate(subprob_var_indices):
      if sub_variable_index == -1:
        # This variable doesn't participate in this subproblem.
        continue

      new_penalties = (
          system.data.dag_tables[subproblem_index].penalties[
              sub_variable_index, :
          ]
          + penalty_vector / num_relevant
      )
      packed_dags.set_penalties(
          memo=system.data.dag_tables[subproblem_index],
          variable_index=sub_variable_index,
          penalties=new_penalties,
      )
