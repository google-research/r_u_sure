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

"""Protocol for backends for drawing figures."""
from __future__ import annotations
import dataclasses
import enum
from typing import Callable, Protocol, ContextManager, Optional


@dataclasses.dataclass(frozen=True)
class Point:
  """A point in space."""
  x: float
  y: float

  def shift(self, dx: float = 0.0, dy: float = 0.0) -> Point:
    """Returns a shifted copy of this point."""
    return Point(self.x + dx, self.y + dy)


@dataclasses.dataclass(frozen=True)
class BoundingBox:
  """A bounding box in 2D space, oriented from top-left corner."""
  left: float
  top: float
  width: float
  height: float

  @classmethod
  def from_boundary(
      cls,
      left: float,
      top: float,
      right: float,
      bottom: float,
  ) -> BoundingBox:
    return cls(left=left, top=top, width=right - left, height=bottom - top)

  @property
  def right(self) -> float:
    return self.left + self.width

  @property
  def bottom(self) -> float:
    return self.top + self.height

  @property
  def horizontal_center(self) -> float:
    return self.left + 0.5 * self.width

  @property
  def vertical_center(self) -> float:
    return self.top + 0.5 * self.height

  @property
  def center_point(self) -> Point:
    return Point(self.horizontal_center, self.vertical_center)

  def point_within(self, toward_right: float, toward_bottom: float) -> Point:
    return Point(self.left + toward_right * self.width,
                 self.top + toward_bottom * self.height)

  def union(self, other: BoundingBox) -> BoundingBox:
    """Constructs the union of the two bounding boxes."""
    return BoundingBox.from_boundary(
        left=min(self.left, other.left),
        top=min(self.top, other.top),
        right=max(self.right, other.right),
        bottom=max(self.bottom, other.bottom))


class PointDirection(enum.Enum):
  """A direction away from a point."""
  UP = enum.auto()
  UP_RIGHT = enum.auto()
  RIGHT = enum.auto()
  DOWN_RIGHT = enum.auto()
  DOWN = enum.auto()
  DOWN_LEFT = enum.auto()
  LEFT = enum.auto()
  UP_LEFT = enum.auto()
  AROUND = enum.auto()


class Renderer(Protocol):
  """Interface for a renderer, which allows drawing figures."""

  def label_in_box(
      self,
      bounds: BoundingBox,
      text: str,
      hover_text: str,
      text_size: float,
      style_tags: tuple[str, ...] = (),
  ) -> None:
    """Draws a label within a given bounds.

    Args:
      bounds: Bounds in which to draw.
      text: The text to draw.
      hover_text: Text to show on mouse hover, if applicable.
      text_size: Size of the text.
      style_tags: Tuple of tags which can be used to style elements.
    """
    raise NotImplementedError()

  def label_at_point(
      self,
      point: Point,
      text: str,
      text_size: float,
      direction: PointDirection,
      hover_text: Optional[str] = None,
      padding_fraction: float = 0.5,
      style_tags: tuple[str, ...] = (),
  ) -> None:
    """Draws a label at a given location.

    Args:
      point: Point around which to draw.
      text: The text to draw.
      text_size: Size of the text.
      direction: Direction to orient the text relative to the point.
      hover_text: Text to show on mouse hover, if applicable.
      padding_fraction: How far to offset the text from the point, relative to
        text_size.
      style_tags: Tuple of tags which can be used to style elements.
    """
    raise NotImplementedError()

  def container(
      self,
      bounds: BoundingBox,
      style_tags: tuple[str, ...] = (),
  ) -> None:
    """Draws a box at a given location.

    Args:
      bounds: Bounds in which to draw.
      style_tags: Tuple of tags which can be used to style elements.
    """
    raise NotImplementedError()

  def line(
      self,
      points: list[Point],
      hover_text: Optional[str] = None,
      style_tags: tuple[str, ...] = (),
  ) -> None:
    """Draws a (poly)line between keypoints.

    Args:
      points: List of points which will be connected in sequence.
      hover_text: Text to show on mouse hover, if applicable.
      style_tags: Tuple of tags which can be used to style elements.
    """
    raise NotImplementedError()

  def dot(
      self,
      point: Point,
      hover_text: Optional[str] = None,
      style_tags: tuple[str, ...] = (),
  ) -> None:
    """Draws a circle at the given location.

    Size can be adjusted using the style tags.

    Args:
      point: Point to draw the dot at.
      hover_text: Text to show on mouse hover, if applicable.
      style_tags: Tuple of tags which can be used to style elements.
    """
    raise NotImplementedError()

  def group(
      self,
      hover_text: Optional[str] = None,
      style_tags: tuple[str, ...] = (),
  ) -> ContextManager[None]:
    """Context manager to group nodes together.

    Args:
      hover_text: Text to show on mouse hover, if applicable.
      style_tags: Tuple of tags which can be used to style elements.

    Returns:
      Context manager instance, for use in a `with` clause.
    """
    raise NotImplementedError()


def point_interpolator(first: Point,
                       second: Point) -> Callable[[float, float], Point]:
  """Returns a helper function that interpolates between the given points."""

  def _interpolate(x: float, y: float) -> Point:
    return Point(first.x + x * (second.x - first.x),
                 first.y + y * (second.y - first.y))

  return _interpolate


def to_bounded_irrational(index: int) -> float:
  """Returns an irrational number between 0 and 1.

  Results are based on the golden ratio, which should result in the outputs
  being roughly evenly spaced over the unit interval.

  See https://asknature.org/strategy/fibonacci-sequence-optimizes-packing/

  Args:
    index: An integer index.

  Returns:
    A floating point number between 0 and 1, such that outputs tend to be far
    apart from all previous outputs.
  """

  return (index / 1.61803 + 0.5) % 1.0
