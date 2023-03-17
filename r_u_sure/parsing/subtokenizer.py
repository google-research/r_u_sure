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

"""Tokenizer used for subtokenization of string literals.

Adapted from the cross-language tokenization library

https://github.com/google-research/google-research/blob/master/cubert/unified_tokenizer.py

used in the paper

Kanade, Aditya et al. "Learning and Evaluating Contextual Embedding of Source
Code." International Conference on Machine Learning (2019).
"""

import enum


class StateType(enum.IntEnum):
  INITIAL_STATE = 0
  UPPERCASE_STATE = 1
  LOWERCASE_STATE = 2
  NUMBER_STATE = 3
  SPECIAL_STATE = 4


def code_to_tokens_simple_lossless(code):
  r"""Convert python source code to list of tokens.

  This is a simple version using spacing and different classes of characters to
  tokenize a string. A sentence will be split at "|" in the following patterns:

    upper | upper lower
    upper | number
    upper | special
    lower | upper
    lower | number
    lower | special
    number | upper
    number | lower
    number | special
    special | upper
    special | lower
    special | number
  In addition to splits caused by the type changes above, the code is also split
  at whitespace. However, a sequence of spaces or tabs will not be split unless
  its length is longer than 20.
  For example: "12345  \n\n678" -> ["12345", "  ", "\n", "\n", "678"]
  We do not split sequences of spaces/tabs to avoid long sequences of single
  " " or "\t" tokens caused by deep indentation.
  This tokenizer uses a finite state machine. The definition of the states is in
  the StateType class.
  Args:
    code: String containing Python source code.

  Returns:
    The code represented as a string of tokens separated by spaces.
    For example, "foo  ,1" -> ["foo", "  ", ",", "1"]
  """
  # normal state transitions that will result in splitting
  normal_transitions = [
      (StateType.UPPERCASE_STATE, StateType.NUMBER_STATE),
      (StateType.UPPERCASE_STATE, StateType.SPECIAL_STATE),
      (StateType.LOWERCASE_STATE, StateType.UPPERCASE_STATE),
      (StateType.LOWERCASE_STATE, StateType.NUMBER_STATE),
      (StateType.LOWERCASE_STATE, StateType.SPECIAL_STATE),
      (StateType.NUMBER_STATE, StateType.UPPERCASE_STATE),
      (StateType.NUMBER_STATE, StateType.LOWERCASE_STATE),
      (StateType.NUMBER_STATE, StateType.SPECIAL_STATE),
      (StateType.SPECIAL_STATE, StateType.UPPERCASE_STATE),
      (StateType.SPECIAL_STATE, StateType.LOWERCASE_STATE),
      (StateType.SPECIAL_STATE, StateType.NUMBER_STATE),
  ]
  # output, state
  tokens = []
  state = StateType.INITIAL_STATE
  next_state = None
  memory = []
  for i, inputchar in enumerate(code):
    if inputchar.isupper():
      next_state = StateType.UPPERCASE_STATE
    elif inputchar.islower():
      next_state = StateType.LOWERCASE_STATE
    elif inputchar.isdigit():
      next_state = StateType.NUMBER_STATE
    else:
      next_state = StateType.SPECIAL_STATE

    # splitting cases
    if (state, next_state) in normal_transitions:
      tokens.append(''.join(memory))
      memory = []
    elif (state, next_state) == (
        StateType.UPPERCASE_STATE,
        StateType.LOWERCASE_STATE,
    ) and len(memory) > 1:
      tokens.append(''.join(memory[:-1]))
      memory = [memory[-1]]
    elif (state, next_state) == (
        StateType.SPECIAL_STATE,
        StateType.SPECIAL_STATE,
    ):
      if inputchar in [' ', '\t'] and inputchar == code[i - 1]:
        if len(memory) >= 20:
          tokens.append(''.join(memory))
          memory = []
      elif inputchar.isspace() or code[i - 1].isspace():
        tokens.append(''.join(memory))
        memory = []

    # put inputchar into memory, always
    memory.append(inputchar)
    state = next_state
  if memory:
    tokens.append(''.join(memory))
  return tokens
