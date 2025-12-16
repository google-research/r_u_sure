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

"""Helper functions to manage exceptions in Numba code."""

import numba


def safe_raise(cls, message_parts):
  """Raises the given exception with a literal message and format arguments."""
  raise cls(" ".join(str(part) for part in message_parts))


@numba.extending.overload(safe_raise, prefer_literal=True)
def _safe_raise_overload(cls, message_parts):
  """Numba-safe overload of `safe_raise`."""
  cls_name = str(cls)
  literal_parts = []
  for item in message_parts:
    if isinstance(item, numba.types.Literal):
      literal_parts.append(str(item.literal_value))
    else:
      literal_parts.append("<????>")

  literal_message = " ".join(literal_parts)

  def impl(cls, message_parts):  # pylint: disable=unused-argument
    print("Exception args:", cls_name, message_parts)
    raise RuntimeError(
        cls_name, literal_message, "(see printed output for details)"
    )

  return impl
