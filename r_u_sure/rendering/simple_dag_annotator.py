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

"""Module to help build simple annotators."""

from __future__ import annotations

import abc
import textwrap

import numpy as np

from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.rendering import dag_annotator
from r_u_sure.rendering import rendering
from r_u_sure.rendering import svg_renderer


State = gated_state_dag.State
Edge = gated_state_dag.Edge
CompleteStateDAG = gated_state_dag.CompleteStateDAG


def _sigmoid(x: float) -> float:
  return 1.0 / (1.0 + np.exp(-x))


def _summarize_edge(edge: Edge) -> str:
  summary = f"{edge.cost}"
  if edge.required_assignment:
    summary += (
        f" if {edge.required_assignment.key} = {edge.required_assignment.value}"
    )
  return summary


class SimpleDAGAnnotator(dag_annotator.StateDAGAnnotator, abc.ABC):
  """Base class for simple annotators, useful for prototyping and demos."""

  font_size = 10.0
  horizontal_scale = 100.0
  vertical_scale = 100.0
  known_colors = (
      "red",
      "darkred",
      "limegreen",
      "darkgreen",
      "lightblue",
      "blue",
      "orange",
      "magenta",
      "cyan",
  )

  @abc.abstractmethod
  def bounds_for_state(self, state: State) -> rendering.BoundingBox:
    """Constructs a bounding box for a given state."""
    raise NotImplementedError(
        "Subclasses of SimpleDAGAnnotator must implement bounds_for_state."
    )

  def annotate_state(self, state: State) -> dag_annotator.StateAnnotation:
    """Annotates a state in a generic way."""
    bounds = self.bounds_for_state(state)
    return dag_annotator.StateAnnotation(
        bounds=bounds,
        display_text=str(state),
        hover_text=repr(state),
        text_size=self.font_size,
    )

  def annotate_edge(self, edge: Edge) -> dag_annotator.EdgeAnnotation:
    """Annotates an edge in a generic way."""
    source_bounds = self.bounds_for_state(edge.source)
    dest_bounds = self.bounds_for_state(edge.dest)

    horizontal_gap = dest_bounds.left - source_bounds.right

    if horizontal_gap >= 0:
      orientation = dag_annotator.ElbowEdgeOrientation.HORIZONTAL
      secondary_offset = (
          dest_bounds.vertical_center - source_bounds.vertical_center
      )
      secondary_scale = self.vertical_scale
    else:
      orientation = dag_annotator.ElbowEdgeOrientation.VERTICAL
      secondary_offset = (
          dest_bounds.horizontal_center - source_bounds.horizontal_center
      )
      secondary_scale = self.horizontal_scale

    relative_offset_for_endpoints = (
        2 * _sigmoid(secondary_offset / (0.5 * secondary_scale)) - 1
    )

    # Jitter offset based on assignment
    if edge.required_assignment:
      jitter_base = rendering.to_bounded_irrational(
          hash(edge.required_assignment.value) % 100
      )
      jitter = 2 * jitter_base - 1
    else:
      jitter = 0

    # Alignment of start and end along secondary axis are determined by the
    # secondary offset relative to the length scale.
    start_align = 0.5 + 0.3 * relative_offset_for_endpoints + 0.1 * jitter
    end_align = 0.5 - 0.3 * relative_offset_for_endpoints + 0.1 * jitter

    relative_offset_for_middle = (
        2 * _sigmoid(secondary_offset / (1.5 * secondary_scale)) - 1
    )
    # Elbow is chosen to try to reduce intersections.
    if np.abs(relative_offset_for_middle) < 0.01:
      # Effectively a straight line.
      elbow_align = 0.5
      text_align = 0.5
    else:
      # Elbow may intersect other lines.
      if relative_offset_for_middle > 0:
        elbow_align = 0.9 - 0.8 * relative_offset_for_middle - 0.1 * jitter
        text_align = 1.0 - elbow_align
      else:
        elbow_align = 0.9 + 0.8 * relative_offset_for_middle + 0.1 * jitter
        text_align = elbow_align

    if isinstance(edge.info, dict) and "color" in edge.info:
      color = edge.info["color"]
      if color in self.known_colors:
        style_tags = (f"color_{color}",)
      else:
        raise ValueError(f"Unrecognized color {color}.")
    else:
      style_tags = ()

    return dag_annotator.ElbowEdgeAnnotation(
        primary_axis=orientation,
        elbow_distance=elbow_align,
        start_alignment=start_align,
        text_alignment=text_align,
        end_alignment=end_align,
        display_text=_summarize_edge(edge),
        text_size=self.font_size,
        hover_text=repr(edge),
        style_tags=style_tags,
    )

  def renderer_specific_setup(self, renderer: rendering.Renderer) -> None:
    if isinstance(renderer, svg_renderer.SVGRenderer):
      style_css_parts = []
      for color in self.known_colors:
        style_css_parts.append(
            textwrap.dedent(
                f"""\
                .edge-annotation.color_{color} path {{
                    stroke: {color};
                }}
                .edge-annotation.color_{color} text {{
                    fill: {color};
                }}
                """
            )
        )
      renderer.configure_style_css("\n".join(style_css_parts))
