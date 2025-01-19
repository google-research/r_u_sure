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

"""Implementation of a renderer using SVG."""

import contextlib
import html
from typing import Optional

from r_u_sure.rendering import rendering

BoundingBox = rendering.BoundingBox
Point = rendering.Point


class SVGRenderer(rendering.Renderer):
  """Renderer that renders to an SVG string."""
  items: list[str]
  styles: list[str]
  bounds: Optional[BoundingBox]
  with_hover_text: bool

  BASIC_STYLES = """
    text.label_in_box {
        text-anchor: middle;
        dominant-baseline: middle;
    }
  """

  DEFAULT_STYLES = """
    rect.label_in_box {
        fill: lightgray;
        stroke: black;
        stroke-width: 1;
    }
    rect.container {
        fill: none;
        stroke: blue;
        stroke-width: 1;
    }
    path.line {
        fill: none;
        stroke: black;
        stroke-width: 2;
    }
    circle.dot {
        r: 2;
        fill: black;
    }
  """

  def __init__(self,
               with_default_styles: bool = True,
               with_hover_text: bool = True):
    """Constructs an SVG renderer.

    Args:
      with_default_styles: Whether to include default styles in the CSS.
      with_hover_text: Whether to include hover text.
    """
    self.items = []
    self.styles = [SVGRenderer.BASIC_STYLES]
    if with_default_styles:
      self.styles.append(SVGRenderer.DEFAULT_STYLES)
    self.bounds = None
    self.with_hover_text = with_hover_text

  def _update_bounds(self, item_bounds: BoundingBox):
    """Updates the total bounding box to include the updated bounds."""
    if self.bounds is None:
      self.bounds = item_bounds
    else:
      self.bounds = self.bounds.union(item_bounds)

  def _render_hover_title(self, hover_text: Optional[str]) -> str:
    """Optionally constructs a SVG title element for the given text."""
    if hover_text is None or not self.with_hover_text:
      return ""
    else:
      return "<title>" + html.escape(hover_text) + "</title>"

  def label_in_box(
      self,
      bounds: BoundingBox,
      text: str,
      hover_text: str,
      text_size: float,
      style_tags: tuple[str, ...] = (),
  ):
    """Draws a label within a given bounds."""
    self._update_bounds(bounds)
    center = bounds.center_point
    lines = text.split("\n")
    offset = -text_size * (len(lines) - 1) / 2
    texts = []
    for i, line in enumerate(lines):
      texts.append(f"<text font-size={text_size} "
                   f'x="{center.x}" y="{center.y + offset + text_size * i}" '
                   f'class="label_in_box">{html.escape(line)}</text>')
    texts = "\n".join(texts)
    # pylint: disable=g-complex-comprehension
    self.items.append(f"""
      <g>
        {self._render_hover_title(hover_text)}
        <rect x="{bounds.left}" y="{bounds.top}"
              width="{bounds.width}" height="{bounds.height}"
              class="{' '.join(style_tags)} label_in_box"
              />
        {texts}
      </g>
    """)

  def label_at_point(
      self,
      point: Point,
      text: str,
      text_size: float,
      direction: rendering.PointDirection,
      hover_text: Optional[str] = None,
      padding_fraction: float = 0.5,
      style_tags: tuple[str, ...] = (),
  ):
    """Draws a label at a given location."""
    self._update_bounds(BoundingBox(point.x, point.y, 0, 0))
    lines = text.split("\n")
    inline_styles = []
    inline_styles.append("dominant-baseline: middle;")

    # Vertical offset
    if direction in {
        rendering.PointDirection.UP,
        rendering.PointDirection.UP_LEFT,
        rendering.PointDirection.UP_RIGHT,
    }:
      vertical_align_offset = (len(lines) + 0.5) * text_size
      vertical_pad_offset = -padding_fraction * text_size
    elif direction in {
        rendering.PointDirection.DOWN,
        rendering.PointDirection.DOWN_LEFT,
        rendering.PointDirection.DOWN_RIGHT,
    }:
      vertical_align_offset = 0.5 * text_size
      vertical_pad_offset = padding_fraction * text_size
    elif direction in {
        rendering.PointDirection.LEFT,
        rendering.PointDirection.RIGHT,
    }:
      vertical_align_offset = text_size * (len(lines) - 1) / 2
      vertical_pad_offset = 0
    elif direction in {rendering.PointDirection.AROUND}:
      lines_above = (len(lines) + 1) // 2
      vertical_align_offset = (-lines_above + 0.5) * text_size
      vertical_pad_offset = 0
    else:
      raise ValueError(f"Invalid direction {direction}")

    # Horizontal offset
    if direction in {
        rendering.PointDirection.LEFT,
        rendering.PointDirection.UP_LEFT,
        rendering.PointDirection.DOWN_LEFT,
    }:
      inline_styles.append("text-anchor: end;")
      horizontal_pad_offset = -padding_fraction * text_size
    elif direction in {
        rendering.PointDirection.RIGHT,
        rendering.PointDirection.UP_RIGHT,
        rendering.PointDirection.DOWN_RIGHT,
    }:
      inline_styles.append("text-anchor: start;")
      horizontal_pad_offset = padding_fraction * text_size
    elif direction in {
        rendering.PointDirection.UP,
        rendering.PointDirection.DOWN,
        rendering.PointDirection.AROUND,
    }:
      inline_styles.append("text-anchor: middle;")
      horizontal_pad_offset = 0

    texts = []
    base_x = point.x + horizontal_pad_offset
    base_y = point.y + vertical_align_offset + vertical_pad_offset
    for i, line in enumerate(lines):
      texts.append(f"<text font-size={text_size} "
                   f'x="{base_x}" y="{base_y + text_size * i}" '
                   f'style="{" ".join(inline_styles)}" '
                   f'class="{" ".join(style_tags)} label_at_point">'
                   f"{html.escape(line)}</text>")
    texts = "\n".join(texts)
    self.items.append(f"""
      <g>
        {self._render_hover_title(hover_text)}
        {texts}
      </g>
    """)

  def container(
      self,
      bounds: BoundingBox,
      style_tags: tuple[str, ...] = (),
  ):
    """Draws a box at a given location."""
    self._update_bounds(bounds)
    self.items.append(f"""
      <g>
        <rect x="{bounds.left}" y="{bounds.top}"
              width="{bounds.width}" height="{bounds.height}"
              class="{' '.join(style_tags)} container"
              />
      </g>
    """)

  def line(
      self,
      points: list[Point],
      hover_text: Optional[str] = None,
      style_tags: tuple[str, ...] = (),
  ):
    """Draws a line between keypoints."""
    self._update_bounds(
        BoundingBox.from_boundary(
            left=min(point.x for point in points),
            right=max(point.x for point in points),
            top=min(point.y for point in points),
            bottom=max(point.y for point in points)))
    pathspec = []
    pathspec.append(f"M {points[0].x} {points[0].y}")
    for point in points[1:]:
      pathspec.append(f"L {point.x} {point.y}")
    self.items.append(f"""
      <g>
        {self._render_hover_title(hover_text)}
        <path d="{' '.join(pathspec)}" class="{' '.join(style_tags)} line"
              marker-start="url(#arrowstart)"
              marker-end="url(#arrowend)"/>
      </g>
    """)

  def dot(
      self,
      point: Point,
      hover_text: Optional[str] = None,
      style_tags: tuple[str, ...] = (),
  ) -> None:
    """Draws a circle at the given location."""
    self._update_bounds(
        BoundingBox.from_boundary(
            left=point.x, right=point.x, top=point.y, bottom=point.y))
    self.items.append(f"""
      <g>
        {self._render_hover_title(hover_text)}
        <circle cx="{point.x}" cy="{point.y}" class="{' '.join(style_tags)} dot"/>
      </g>
    """)

  @contextlib.contextmanager
  def group(
      self,
      hover_text: Optional[str] = None,
      style_tags: tuple[str, ...] = (),
  ):
    """Context manager to group nodes together."""
    self.items.append(f"""
      <g class="{' '.join(style_tags)}">
        {self._render_hover_title(hover_text)}
    """)
    yield
    self.items.append("""
      </g>
    """)

  def configure_style_css(self, style_string: str):
    """Adds CSS style information."""
    self.styles.append(style_string)

  def add_custom_svg(self, svg_source: str) -> None:
    """Adds custom SVG items."""
    self.items.append(svg_source)

  def to_html(self, padding: float = 0.0) -> str:
    """Assembles all rendered items into an HTML SVG string."""
    bounds = self.bounds or BoundingBox(0, 0, 10, 10)
    viewbox = BoundingBox.from_boundary(
        left=bounds.left - padding,
        right=bounds.right + padding,
        top=bounds.top - padding,
        bottom=bounds.bottom + padding)
    styles = "\n".join(self.styles)
    items = "\n".join(self.items)
    # identifier = uuid.uuid4()

    return f"""
      <style type="text/css">
        {styles}
      </style>
      <svg class="rendered_graph" viewBox="{viewbox.left} {viewbox.top} {viewbox.width} {viewbox.height}">
      <defs>
        <marker id="arrowend"
            markerWidth="5" markerHeight="4"
            refX="5" refY="2"
            orient="auto" markerUnits="strokeWidth"
        >
          <path d="M0,0 L0,4 L5,2 z" fill="black" />
        </marker>
        <marker id="arrowstart"
            markerWidth="5" markerHeight="5"
            refX="0" refY="2"
            orient="auto" markerUnits="strokeWidth"
        >
          <circle cx="1.5" cy="2" r=1.5 fill="black" />
        </marker>
      </defs>
        <g>
          {items}
        </g>
      </svg>
    """

