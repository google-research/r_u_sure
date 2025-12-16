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

"""Tests for numba enum hash patch."""

import enum

from absl.testing import absltest
import numba
from r_u_sure.numba_helpers import register_enum_hash  # pylint: disable=unused-import.


class MyHashableEnum(enum.Enum):
  FOO = 0
  BAR = 1
  BAZ = 42

  # Numba-compatible __hash__ implementation.
  __hash__ = register_enum_hash.jitable_enum_hash


class MyNotHashableEnum(enum.Enum):
  FOO = 0
  BAR = 1
  BAZ = 42


@numba.njit
def jit_hash(item):
  return hash(item)


class NumbaEnumHashTest(absltest.TestCase):

  def test_hashable_enums(self):
    for member in MyHashableEnum:
      self.assertEqual(jit_hash(member), hash(member))

  def test_hashable_enum_requires_hash_override(self):
    with self.assertRaisesWithPredicateMatch(
        numba.core.errors.TypingError,
        lambda exc: (  # pylint: disable=g-long-lambda
            "Untyped global name 'instance_hash_function': Cannot determine"
            " Numba type of <class 'function'>"
            in exc.args[0]
        ),
    ):
      jit_hash(MyNotHashableEnum.FOO)


if __name__ == "__main__":
  # Set up Numba debugging for better errors.
  numba.config.FULL_TRACEBACKS = True
  numba.config.NUMBA_CAPTURED_ERRORS = "new_style"
  numba.config.DEVELOPER_MODE = 1
  numba.core.utils.DEVELOPER_MODE = 1

  absltest.main()
