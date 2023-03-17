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

"""Utilities for tests of high-level wrappers."""

import itertools
import math
import re


def split_with_fake_log_probs(source: str) -> list[tuple[str, float]]:
  """Splits a source string into tokens, and assigns fake log probs.

  Args:
    source: Source string. Any alphanumeric identifier of the form `prob_XX`
      will be assigned a probability of XX/100. All other tokens get 100%
      probability.

  Returns:
    List of tuples of source token and fake log probability.
  """
  # Split into whitespace, alphanumeric, and other.
  fake_tokens = re.split(r'(\s+|\w+)', source)
  # Assign fake log probabilities to tokens that start with `prob`.
  fake_tokens_with_log_probs = []
  for token in fake_tokens:
    if token:
      matched = re.fullmatch(r'prob_(\d+)', token)
      if matched:
        prob = float(matched.group(1)) / 100.0
        log_prob = math.log(prob)
      else:
        log_prob = 0.0
      fake_tokens_with_log_probs.append((token, log_prob))
  return fake_tokens_with_log_probs


def group_parts_by_confidence(
    extracted_parts: list[tuple[str, bool]], invert_confidence: bool = False
) -> list[tuple[str, str]]:
  """Groups parts of a suggestion by their confidence."""
  parts_grouped_by_confidence = []
  for is_low_confidence, parts in itertools.groupby(
      extracted_parts, lambda v: v[1]
  ):
    combined_parts = ''.join(part for (part, _) in parts)
    if invert_confidence:
      is_low_confidence = not is_low_confidence
    parts_grouped_by_confidence.append(
        (combined_parts, 'low' if is_low_confidence else 'high')
    )
  return parts_grouped_by_confidence
