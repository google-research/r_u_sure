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

"""Entry point executable to run all tests."""

import sys
from absl.testing import absltest

import r_u_sure.testing.test_flags  # pylint: disable=unused-import

if __name__ == "__main__":
  absltest.main(
      module=None,
      argv=[sys.argv[0], "discover", "-s", "r_u_sure", "-p", "*_test.py"],
  )
