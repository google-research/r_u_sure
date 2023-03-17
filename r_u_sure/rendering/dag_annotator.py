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

"""Classes that define how to render annotations for GatedStateDAGs."""

from __future__ import annotations

import abc
import dataclasses
import enum
from typing import Iterable, Optional, Protocol, Union

from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.rendering import rendering
from r_u_sure.rendering import svg_renderer

State = gated_state_dag.State
Edge = gated_state_dag.Edge
CompleteStateDAG = gated_state_dag.CompleteStateDAG


@dataclasses.dataclass
class StateAnnotation:
  """Annotation for a state in a figure.

  Attributes:
    bounds: Bounding box to render the state into.
    display_text: Text to display in this state.
    hover_text: Text to display on mouse hover for this state (if applicable).
    text_size: Font size for the display text.
    style_tags: Tags used by the renderer to style this annotation.
  """

  bounds: rendering.BoundingBox
  display_text: str
  text_size: float
  hover_text: Optional[str] = None
  style_tags: tuple[str, ...] = ()


class ElbowEdgeOrientation(enum.Enum):
  """Orientation for an elbow edge."""

  HORIZONTAL = 0
  VERTICAL = 1


@dataclasses.dataclass
class ElbowEdgeAnnotation:
  """Elbow-style annotation for an edge (transition) in a figure.

  Attributes:
    primary_axis: Whether the elbow connector should start and end as a
      horizontal line, with a vertical segment in the middle, or the opposite.
    elbow_distance: How far along the primary axis to include the perpendicular
      segment, between 0 to 1.
    start_alignment: How far along the (secondary axis) side of the source state
      to draw the edge, between 0 to 1.
    text_alignment: How far along the perpendicular segment to draw the text
      annotation, between 0 to 1.
    end_alignment: How far along the (secondary axis) side of the destination
      state to draw the edge, between 0 to 1.
    display_text: Text to display for this connector.
    hover_text: Text to display on mouse hover for this edge (if applicable)
    text_size: Font size for the display text.
    style_tags: Tags used by the renderer to style this annotation.
    elbow_distance_adjust: Adjustment to elbow distance in absolute units.
    text_alignment_adjust: Adjustment to text alignment in absolute units.
  """

  primary_axis: ElbowEdgeOrientation
  elbow_distance: float
  start_alignment: float
  text_alignment: float
  end_alignment: float
  display_text: str
  text_size: float
  hover_text: Optional[str] = None
  style_tags: tuple[str, ...] = ()
  elbow_distance_adjust: float = 0.0
  text_alignment_adjust: float = 0.0


EdgeAnnotation = Union[ElbowEdgeAnnotation]


@dataclasses.dataclass
class RegionAnnotation:
  """Annotation for a rectangular region in a figure.

  Attributes:
    bounds: Bounding box to render the state into.
    style_tags: Tags used by the renderer to style this annotation.
  """

  bounds: rendering.BoundingBox
  style_tags: tuple[str, ...] = ()


@dataclasses.dataclass
class TextAnnotation:
  """Annotation for some text.

  Attributes:
    bounds: Bounding box to render the state into.
    display_text: Text to display here.
    hover_text: Text to display on mouse hover for this edge (if applicable)
    text_size: Font size for the display text.
    style_tags: Tags used by the renderer to style this annotation.
  """

  bounds: rendering.BoundingBox
  display_text: str
  text_size: float
  hover_text: Optional[str] = None
  style_tags: tuple[str, ...] = ()


class StateDAGAnnotator(Protocol):
  """Helper object that annotates states (states) for a figure."""

  @abc.abstractmethod
  def annotate_state(
      self,
      state: State,
  ) -> StateAnnotation:
    """Assigns rendering annotations to a state.

    Args:
      state: The state to assign annotations to.

    Returns:
      Information on how to render this state.
    """
    raise NotImplementedError(
        "StateDAGAnnotator must implement `annotate_state`"
    )

  @abc.abstractmethod
  def annotate_edge(
      self,
      edge: Edge,
  ) -> EdgeAnnotation:
    """Assigns rendering annotations to an edge.

    Args:
      edge: The edge to assign annotations to.

    Returns:
      Information on how to render this edge.
    """
    raise NotImplementedError(
        "StateDAGAnnotator must implement `annotate_edge`"
    )

  def extra_annotations(
      self,
  ) -> Iterable[Union[RegionAnnotation, TextAnnotation]]:
    """Produces additional annotations."""
    return []

  def renderer_specific_setup(self, renderer: rendering.Renderer) -> None:
    """Hook to customize rendering for a particular renderer."""


def render_dag(
    dag: CompleteStateDAG,
    annotator: StateDAGAnnotator,
    renderer: rendering.Renderer,
    emphasized_edges: Optional[Iterable[Edge]] = None,
) -> None:
  """Renders a DAG using an annotator.

  Renders all region annotations, state annotations, and edge annotations
  for the graph. Optionally also highlights particular paths through the graph.

  Args:
    dag: The directed acyclic graph to render.
    annotator: The annotator to use.
    renderer: The underlying renderer, used to emit drawing commands.
    emphasized_edges: An optional list of edges to emphasize during drawing, for
      instance, to visualize the shortest path through the graph.
  """
  annotator.renderer_specific_setup(renderer)

  if isinstance(renderer, svg_renderer.SVGRenderer):
    renderer.configure_style_css(default_svg_style_css())
    renderer.add_custom_svg(
        """
        <filter id="hoverhighlight">
          <feMorphology
              operator="dilate" radius="5"
              in="SourceGraphic" result="dilated"/>
          <feColorMatrix
              type="matrix" values="0.5 0   0   0.5 0
                                    0   0.5 0   0.5 0
                                    0   0   0.5 0.5 0
                                    0   0   0   1   0"
              in="dilated" result="glow"/>
          <feGaussianBlur stdDeviation="3"
              in="glow" result="glowblur"/>
          <feBlend in="SourceGraphic" in2="glowblur" mode="normal" />
        </filter>
        """
    )

  # Find states and determine reachability.
  all_states = gated_state_dag.extract_state_list(dag)
  reachable_states = gated_state_dag.compute_reachable_states(dag)

  # Draw annotation regions.
  for extra_annotation in annotator.extra_annotations():
    if isinstance(extra_annotation, RegionAnnotation):
      renderer.container(
          bounds=extra_annotation.bounds,
          style_tags=extra_annotation.style_tags + ("text-annotation",),
      )
    elif isinstance(extra_annotation, TextAnnotation):
      renderer.label_in_box(
          bounds=extra_annotation.bounds,
          text=extra_annotation.display_text,
          hover_text=extra_annotation.hover_text,
          text_size=extra_annotation.text_size,
          style_tags=extra_annotation.style_tags + ("text-annotation",),
      )

  # Draw states.
  for state in all_states:
    state_annotation = annotator.annotate_state(state)
    style_tags = state_annotation.style_tags + ("state-annotation",)
    if state in reachable_states:
      style_tags += ("reachable-state",)
    else:
      style_tags += ("unreachable-state",)

    renderer.label_in_box(
        bounds=state_annotation.bounds,
        text=state_annotation.display_text,
        hover_text=state_annotation.hover_text,
        text_size=state_annotation.text_size,
        style_tags=style_tags,
    )

  # Draw edges.
  if emphasized_edges:
    emphasized_edges_set = set()
    for edge in emphasized_edges:
      emphasized_edges_set.add(
          Edge(
              source=edge.source,
              dest=edge.dest,
              cost=edge.cost,
              required_assignment=edge.required_assignment,
              info=None,
          )
      )

  for edge in dag.edges:
    edge_annotation = annotator.annotate_edge(edge)
    source_annotation = annotator.annotate_state(edge.source)
    dest_annotation = annotator.annotate_state(edge.dest)

    if emphasized_edges:
      # Lowlight all edges not in the emphasis set.
      infoless_edge = Edge(
          source=edge.source,
          dest=edge.dest,
          cost=edge.cost,
          required_assignment=edge.required_assignment,
          info=None,
      )
      highlight = infoless_edge in emphasized_edges_set
      lowlight = not highlight
    else:
      highlight = False
      lowlight = False

    if isinstance(edge_annotation, ElbowEdgeAnnotation):
      if edge_annotation.primary_axis == ElbowEdgeOrientation.HORIZONTAL:
        start_point = source_annotation.bounds.point_within(
            toward_right=1.0, toward_bottom=edge_annotation.start_alignment
        )
        end_point = dest_annotation.bounds.point_within(
            toward_right=0.0, toward_bottom=edge_annotation.end_alignment
        )

        interpolator = rendering.point_interpolator(start_point, end_point)
        points = [
            interpolator(0.0, 0.0),
            interpolator(edge_annotation.elbow_distance, 0.0).shift(
                dx=edge_annotation.elbow_distance_adjust
            ),
            interpolator(edge_annotation.elbow_distance, 1.0).shift(
                dx=edge_annotation.elbow_distance_adjust
            ),
            interpolator(1.0, 1.0),
        ]

        if abs(points[2].y - points[1].y) > edge_annotation.text_size:
          text_anchor = interpolator(
              edge_annotation.elbow_distance, edge_annotation.text_alignment
          ).shift(dx=edge_annotation.elbow_distance_adjust)
          label_direction = rendering.PointDirection.RIGHT
        else:
          text_anchor = interpolator(edge_annotation.elbow_distance, 0.0).shift(
              dx=edge_annotation.elbow_distance_adjust
          )
          label_direction = rendering.PointDirection.AROUND
        text_point = text_anchor.shift(dy=edge_annotation.text_alignment_adjust)

      else:
        start_point = source_annotation.bounds.point_within(
            toward_right=edge_annotation.start_alignment, toward_bottom=1.0
        )
        end_point = dest_annotation.bounds.point_within(
            toward_right=edge_annotation.end_alignment, toward_bottom=0.0
        )

        interpolator = rendering.point_interpolator(start_point, end_point)
        points = [
            interpolator(0.0, 0.0),
            interpolator(0.0, edge_annotation.elbow_distance).shift(
                dy=edge_annotation.elbow_distance_adjust
            ),
            interpolator(1.0, edge_annotation.elbow_distance).shift(
                dy=edge_annotation.elbow_distance_adjust
            ),
            interpolator(1.0, 1.0),
        ]

        text_anchor = interpolator(
            edge_annotation.text_alignment, edge_annotation.elbow_distance
        ).shift(dy=edge_annotation.elbow_distance_adjust)
        text_point = text_anchor.shift(dx=edge_annotation.text_alignment_adjust)
        label_direction = rendering.PointDirection.DOWN_RIGHT

      style_tags = edge_annotation.style_tags + ("edge-annotation",)
      if highlight:
        style_tags += ("highlight",)
      if lowlight:
        style_tags += ("lowlight",)

      if edge.source in reachable_states and edge.dest in reachable_states:
        style_tags += ("reachable-edge",)
      else:
        style_tags += ("unreachable-edge",)

      with renderer.group(
          style_tags=style_tags + ("hover_to_top",),
          hover_text=edge_annotation.hover_text,
      ):
        renderer.line(points=points)
        renderer.dot(point=text_anchor)
        renderer.label_at_point(
            point=text_point,
            text=edge_annotation.display_text,
            text_size=edge_annotation.text_size,
            direction=label_direction,
        )

    else:
      raise ValueError(f"Invalid edge annotation {edge_annotation}")


def default_svg_style_css() -> str:
  """Returns a set of default styles for rendering to SVG."""
  return """
      rect.state-annotation.unreachable-state {
          fill: white;
          stroke-dasharray: 2 2;
      }
      .unreachable-edge path.line {
          stroke-opacity: 0.7;
          stroke-dasharray: 2 4;
      }
      .reachable-edge path.line {
          stroke-width: 2;
      }
      circle.dot {
          r: 2;
      }

      .edge-annotation.reachable-edge.highlight path {
          stroke-width: 3;
      }

      .edge-annotation.lowlight text {
          opacity: 0.5;
      }
      .edge-annotation.lowlight path {
          stroke-opacity: 0.3;
          stroke-width: 1;
      }
      .edge-annotation.unreachable-edge.lowlight path {
          stroke-dasharray: 2 8;
      }

      .edge-annotation:hover {
          filter: url(#hoverhighlight);
      }

      .edge-annotation.lowlight:hover text {
          opacity: 1.0;
      }
      .edge-annotation.lowlight:hover path {
          stroke-opacity: 1.0;
      }
  """
