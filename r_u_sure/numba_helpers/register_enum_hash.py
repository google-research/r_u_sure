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

"""Enables hashing of enums in Numba.

Importing this module registers the __hash__ method of all enums as a jittable
function. Due to the way Numba handles enums, this takes effect globally on all
enums.

If this module has been imported, calling `hash` on any enum member in jitted
code will try to use the overloaded definition of `__hash__`, which must be
callable in Numba.
If you don't have specific requirements about the behavior of __hash__, the
easiest way to make an enum hashable inside numba code is to set __hash__
to `jitable_enum_hash` from this module.

This hash function will then work both inside and outside Numba jitted code.
(Note that the default __hash__ implementation uses the member names, which are
not available in numba, so this must be overridden for this to work.)

Unfortunately, affects the behavior of `__hash__` on every enum, since numba
only allows a
single overload of a method on a particular numba typeclass, and all enums have
the same numba typeclass (numba.types.EnumMember or numba.types.IntEnumMember).
This shouldn't cause any problems most of the time, since enums aren't hashable
in Numba by default. The main changes would be:

- trying to hash an enum without a __hash__ method defined will give a slightly
  different error message.
- if some other module tries to overload __hash__ on enums, this could lead to
  conflicts between the two modules.

One more caveat to be aware of is that Numba has an assertion that prevents
the use of Enums in `set` instances, because `set` has an explicit allowlist
of types that it can contain (even though this doesn't seem necessary for the
implementation). A workaround is to instead use a dictionary whose values are
None or some other simple sentinel value, and just use its keys as if they were
a set.
"""

import enum
import numba


@numba.core.extending.overload_method(numba.types.EnumMember, "__hash__")
@numba.core.extending.overload_method(numba.types.IntEnumMember, "__hash__")
def _overload_enum_hash(self):
  """Overload for __hash__ on enum classes.

  When this module is imported, the decorators of this function will register
  this function as the overload of __hash__ on Enum classes, making it so that
  hash(member) works inside Numba code. We redirect the implementation to the
  instance method itself, which must have an overload defined.

  Args:
    self: The Numba type of the Enum member we are calling `__hash__` on. It is
      called `self` so that it is consistent with the definition of __hash__,
      due to Numba's signature restrictions.

  Returns:
    An implementation for Numba to use when compiling `__hash__`. We redirect
    to a call to the (unbound) method definition for the specific subclass
    we are trying to hash, assuming it is registered with Numba.
  """
  # Here `self` is the numba type of the `self` argument that will be passed
  # at runtime. We can extract the numba class using `self.instance_class`
  instance_hash_function = self.instance_class.__hash__

  # Next we redirect the hash function to the instance method implementation.
  # This implementation must be registered as jittable, or else an error will
  # be thrown.
  def impl(self):
    return instance_hash_function(self)

  return impl


@numba.extending.register_jitable
def jitable_enum_hash(member: enum.Enum) -> int:
  """Numba-compatible implementation of __hash__, based on member.value.

  The easiset way to make a subclass of Enum hashable is to set this function
  as __hash__ for it, e.g.

  class MyEnum(enum.Enum):
    # define members

    __hash__ = register_enum_hash.jitable_enum_hash

  This is provided as a convenience for cases where it's not necessary to
  customize the hash for the enum.

  Args:
    member: The member we want to compute a hash for. When used as an instance
      method, this will be bound to `self`.

  Returns:
    The hash of the value associated with the member, which is available in
    both pure Python and Numba jit functions.
  """
  return hash(member.value)
