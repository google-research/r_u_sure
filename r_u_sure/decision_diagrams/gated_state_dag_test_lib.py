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

"""Library for testing state DAGs."""

import enum
from typing import Any

from r_u_sure.decision_diagrams import gated_state_dag
# The below import makes enum.__hash__ available in Numba code:
from r_u_sure.numba_helpers import register_enum_hash  # pylint: disable=unused-import


class States(enum.Enum):
  """Collection of states for the example DAG."""

  START_1 = enum.auto()
  START_2 = enum.auto()
  START_3 = enum.auto()
  VARIANT_1_A = enum.auto()
  VARIANT_1_B = enum.auto()
  VARIANT_1_C = enum.auto()
  VARIANT_1_D = enum.auto()
  VARIANT_1_E = enum.auto()
  VARIANT_1_F = enum.auto()
  VARIANT_2_A = enum.auto()
  VARIANT_2_B = enum.auto()
  VARIANT_2_C = enum.auto()
  VARIANT_2_D = enum.auto()
  VARIANT_2_E = enum.auto()
  VARIANT_2_F = enum.auto()
  END_1 = enum.auto()
  END_2 = enum.auto()
  END_3 = enum.auto()
  SPECIAL_DEADEND = enum.auto()
  SPECIAL_UNREACHABLE_1 = enum.auto()
  SPECIAL_UNREACHABLE_2 = enum.auto()

  # Numba-compatible __hash__ implementation.
  __hash__ = register_enum_hash.jitable_enum_hash


def build_example_dag():
  """Constructs an example DAG."""
  partial_graph = gated_state_dag.partial_state_dag_starting_from(
      States.START_1
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph, gated_state_dag.Edge(States.START_1, States.START_2, 100)
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph, gated_state_dag.Edge(States.START_1, States.START_2, 200)
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(States.START_1, States.SPECIAL_DEADEND, 0),
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.START_2,
          States.START_3,
          1,
          gated_state_dag.SharedVariableAssignment("w", "1"),
      ),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.START_2,
          States.START_3,
          2,
          gated_state_dag.SharedVariableAssignment("w", "2"),
      ),
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.START_3,
          States.VARIANT_1_A,
          2,
          gated_state_dag.SharedVariableAssignment("x", "1"),
      ),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.START_3,
          States.VARIANT_2_A,
          0,
          gated_state_dag.SharedVariableAssignment("x", "2"),
      ),
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(States.VARIANT_1_A, States.VARIANT_1_B, 0),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(States.VARIANT_1_B, States.VARIANT_1_C, 0),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.VARIANT_1_C,
          States.VARIANT_1_D,
          0,
          gated_state_dag.SharedVariableAssignment("y", "A"),
      ),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.VARIANT_1_C,
          States.VARIANT_1_D,
          1000,
          gated_state_dag.SharedVariableAssignment("y", "B"),
      ),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(States.VARIANT_1_D, States.VARIANT_1_E, 0),
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(States.VARIANT_2_A, States.VARIANT_2_B, 0),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(States.VARIANT_2_B, States.VARIANT_2_C, 0),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.VARIANT_2_C,
          States.VARIANT_2_D,
          1000,
          gated_state_dag.SharedVariableAssignment("y", "A"),
      ),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.VARIANT_2_C,
          States.VARIANT_2_D,
          0,
          gated_state_dag.SharedVariableAssignment("y", "B"),
      ),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(States.VARIANT_2_D, States.VARIANT_2_E, 0),
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.VARIANT_2_D,
          States.VARIANT_2_F,
          15,
          gated_state_dag.SharedVariableAssignment("z-v2", "15"),
      ),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.VARIANT_2_E,
          States.VARIANT_2_F,
          25,
          gated_state_dag.SharedVariableAssignment("z-v2", "25"),
      ),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.VARIANT_1_D,
          States.VARIANT_1_F,
          10,
          gated_state_dag.SharedVariableAssignment("z-v1", "10"),
      ),
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.VARIANT_1_E,
          States.VARIANT_1_F,
          20,
          gated_state_dag.SharedVariableAssignment("z-v1", "20"),
      ),
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph, gated_state_dag.Edge(States.VARIANT_1_F, States.END_1, 0)
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph, gated_state_dag.Edge(States.VARIANT_2_F, States.END_1, 0)
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(
          States.SPECIAL_UNREACHABLE_1, States.SPECIAL_UNREACHABLE_2, 0
      ),
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph, gated_state_dag.Edge(States.END_1, States.END_2, 0)
  )
  gated_state_dag.partial_state_dag_add_edge(
      partial_graph, gated_state_dag.Edge(States.END_2, States.END_3, 0)
  )

  gated_state_dag.partial_state_dag_add_edge(
      partial_graph,
      gated_state_dag.Edge(States.SPECIAL_UNREACHABLE_2, States.END_3, 0),
  )

  graph = gated_state_dag.partial_state_dag_finish(partial_graph, States.END_3)
  return graph


def dag_from_paths(
    paths: list[tuple[dict[Any, Any], float]]
) -> gated_state_dag.CompleteStateDAG:
  """Builds a simple state DAG with a specific set of costs.

  Args:
    paths: A sequence of `(assignments, cost)` tuples, where `assignments` is a
      dict of variable assignments, and `cost` is the cost. For instance,
      [({0:"a", 1:"b", 2:2}, 10), ({0:"x", 1:"y", 2:1}, 2)] represents a graph
      with two paths: the first assigns {0:"a", 1:"b", 2:2} and has cost 10, and
      the second assigns {0:"x", 1:"y", 2:1} and has cost 2.

  Returns:
    A CompleteStateDAG with one path for each entry in `paths`. If there are
      no repeats of assignments, then the cost of each assignment will be
      exactly the cost associated with the entry in `paths`, or `inf` if there
      are no matching assignments.
  """
  initial_state = 0
  final_state = 1
  next_state = 2
  partial_dag = gated_state_dag.partial_state_dag_starting_from(initial_state)
  for assignments, cost in paths:
    current_state = initial_state
    for key, value in sorted(assignments.items()):
      gated_state_dag.partial_state_dag_add_edge(
          partial_dag,
          gated_state_dag.Edge(
              source=current_state,
              dest=next_state,
              cost=0.0,
              required_assignment=gated_state_dag.SharedVariableAssignment(
                  key=key,
                  value=value,
              ),
          ),
      )
      current_state = next_state
      next_state += 1
    gated_state_dag.partial_state_dag_add_edge(
        partial_dag,
        gated_state_dag.Edge(
            source=current_state,
            dest=final_state,
            cost=cost,
            required_assignment=None,
        ),
    )
  return gated_state_dag.partial_state_dag_finish(partial_dag, final_state)
