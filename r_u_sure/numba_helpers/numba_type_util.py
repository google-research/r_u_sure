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

"""Helpers to manipulate Numba's type logic.

`numba.typeof` is used to figure out the type of a value in Numba's type system.
Unfortunately it cannot be used with all types; it fails to infer optional types
passed in as arguments, and it doesn't handle lists properly (assuming they are
homogenous and that the type of the first element is the type of all others).
This can sometimes actually crash the interpreter.

These classes can be used inside `numba.typeof` to infer a better type signature
for an object, by forcing it to infer a particular type.
"""

import dataclasses
from typing import Any

import numba


@dataclasses.dataclass
class PretendType:
  """Wrapper type that Numba will interpret as having a given numba type."""

  numba_type: Any


@numba.extending.typeof_impl.register(PretendType)
def _typeof_impl_pretend_type(boxed, context):  # pylint: disable=unused-argument
  return boxed.numba_type


@dataclasses.dataclass
class PretendOptional:
  """Wrapper type that Numba will interpret as an optional."""

  inner: Any


@numba.extending.typeof_impl.register(PretendOptional)
def _typeof_impl_pretend_optional(boxed, context):  # pylint: disable=unused-argument
  return numba.optional(numba.typeof(boxed.inner))


def as_numba_type(value: Any, numba_type: Any) -> Any:
  """Helper function which casts a value to the given numba type."""
  del numba_type
  return value


_CASTERS = {}


@numba.extending.overload(as_numba_type, inline="always")
def _as_numba_type_overload(value, numba_type):
  """Numba overload for `as_numba_type`."""
  del value
  if isinstance(numba_type, (numba.types.TypeRef, numba.types.NumberClass)):
    # This part is running during numba typechecking, so it wraps numba types
    # in TypeRef or NumberClass.
    # Extract the type from the reference to it.
    underlying_type = numba_type.instance_type

    if underlying_type in _CASTERS:
      ensure_casted = _CASTERS[underlying_type]

    else:
      # Construct a compiled function that requires a specific type.
      @numba.njit((underlying_type,), cache=True)
      def ensure_casted(value):
        return value

      _CASTERS[underlying_type] = ensure_casted

    # Call the function, ensuring that the cast occurs.
    def impl(value, numba_type):  # pylint: disable=unused-argument
      # ignore the runtime type
      return ensure_casted(value)

    return impl


@numba.extending.intrinsic
def reveal_numba_type(typingctx: Any, value: Any) -> None:
  del typingctx
  print(f"reveal_numba_type called with {value}")
  raise NotImplementedError(f"reveal_numba_type called with {value}")
