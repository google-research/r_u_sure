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

"""Aligns probabilities for SentencePiece tokens with a parsed representation."""
import dataclasses
import html
import math
from typing import Iterable

from r_u_sure.tree_structure import packed_sequence_nodes

PackedSequenceNodeCategory = packed_sequence_nodes.PackedSequenceNodeCategory


@dataclasses.dataclass
class SpanProbHelper:
  """Helper class to extract prefix-conditional probabilities for spans.

  We define the probability for a span (a contiguous subsequence of characters)
  as the product of the (conditional) token probabilities for all tokens that
  intersect that span.

  If the span starts and ends at a token boundary, this coincides with the
  conditional probability of that sequence of characters given the previous
  sequence of characters. However, we allow spans to start and end at locations
  other than token boundaries, to support using an arbitrary model vocabulary
  with an arbitrary downstream parser even when they use different tokenization
  schemes. In this case, we still incorporate the probability for that token
  in the output.

  This means that, if one span ends in the middle of a token, and the next span
  starts at that same location, then BOTH spans will include the probability
  of that token in their aggregate probability. This means that the product
  of the extracted probabilities over spans does not correspond to a joint
  distribution over the characters in those spans. However, it still provides
  a measurement of confidence for each of the spans.

  This helper is designed to iteratively produce known-size substrings starting
  from the left, so that we can iterate through a different tokenization (e.g.
  from a parser) and extract the corresponding logprobs for those parsed tokens.
  We accomplish this by storing the original model tokens and log probs in a
  stack, with the leftmost tokens at the end of the stack (e.g. in reverse).

  Attributes:
    token_and_log_prob_stack: List of pairs of model tokens and their log
      probabilities under the model, in REVERSE order.
  """

  token_and_log_prob_stack: list[tuple[str, float]]

  @classmethod
  def build(
      cls, tokens_and_log_probs: list[tuple[str, float]]
  ) -> "SpanProbHelper":
    """Constructs a SpanProbHelper for the given tokens and log probs."""
    return SpanProbHelper(tokens_and_log_probs[::-1])

  def advance_by_length(self, span_length: int) -> tuple[str, float]:
    """Extracts the next `length` characters and their aggregate log prob.

    This method consumes as many tokens as necessary to produce an output of
    the requested length, and sums up the log probs of all of those tokens.

    If there are characters remaining in the last processed token after
    consuming `length`, we leave those remaining characters in
    `token_and_log_prob_stack` with the same probability.

    We stop as soon as we have consumed enough characters to reach
    `span_length`. If there are length-zero tokens, those will be processed
    at the start of the next call to `advance_by_length`. Conversely, calling
    this with `span_length=0` uses the log-prob of the next token in the stack.

    Args:
      span_length: The length of the next span to extract.

    Returns:
      (extracted_span, aggregate_log_prob)
    """
    out = []
    total_log_prob = 0.0
    remaining_span_length = span_length
    while self.token_and_log_prob_stack:
      (token, log_prob) = self.token_and_log_prob_stack.pop()
      if len(token) <= remaining_span_length:
        out.append(token)
        total_log_prob += log_prob
        remaining_span_length -= len(token)
        if remaining_span_length == 0:
          break
      else:
        out.append(token[:remaining_span_length])
        total_log_prob += log_prob
        self.token_and_log_prob_stack.append(
            (token[remaining_span_length:], log_prob)
        )
        remaining_span_length = 0
        break

    return "".join(out), total_log_prob

  def advance_and_get_log_prob_for(self, external_token: str) -> float:
    """Extracts the log probability of `token`, if that is the next token.

    This function is designed to be used when comparing two tokenizations of
    the same sequence. By feeding the tokens from the other tokenization, we
    can extract approximate log-probs for each, while checking to make sure they
    are tokenizations of the same sequence.

    Args:
      external_token: The token we are trying to get the log prob for. This must
        be a prefix of the remaining sequence, but does not have to line up with
        our tokenization's boundaries.

    Returns:
      The effective log probability of this token conditioned on the context.
      As discussed in the class docstring, this may not represent a valid
      probability distribution due to attempting to correct for token
      misalignment.

    Raises:
      ValueError: If `external_token` is not a prefix of the remaining sequence
      (and thus, we cannot estimate a log probability for it).
    """
    token, log_prob = self.advance_by_length(len(external_token))
    if token != external_token:
      raise ValueError(
          f"external_token={repr(external_token)} does not match the"
          f" expected sequence prefix {repr(token)}"
      )
    return log_prob

  def assert_consumed(self) -> None:
    """Checks to make sure there are no remaining non-empty tokens."""
    if not all(not token for token, _ in self.token_and_log_prob_stack):
      remaining_tokens = reversed(self.token_and_log_prob_stack)
      raise ValueError(f"Some tokens were not consumed: {remaining_tokens}")


def align_token_log_probs(
    surface_tokens_and_log_probs: list[tuple[str, float]],
    parsed_sequence: packed_sequence_nodes.PackedSequenceNodeStorage,
    must_consume_all_model_tokens: bool = False,
) -> dict[int, float]:
  """Aligns model token probabilities with parsed sequence nodes.

  Args:
    surface_tokens_and_log_probs: Sequence of model (surface) tokens and
      associated conditional log probabilities.
    parsed_sequence: Packed representation of a parsed sequence or tree of
      programming language tokens that we want to compute probabilities for.
    must_consume_all_model_tokens: Whether to expect that `parsed_sequence`
      contains all of the model tokens. Otherwise, allows `parsed_sequence` to
      contain only a prefix of model tokens (e.g. if the suggestion was
      truncated before parsing).

  Returns:
    Mapping such that, for any token node in `parsed_sequence`, if `i` is that
    node's preorder index, `result[i]` is the aggregated probability of the
    model tokens that contributed to that node (conditioned on all previous
    model tokens). This can be used to build baseline methods for uncertainty
    detection, by inserting uncertainty warnings around tokens with low
    probability.
  """
  helper = SpanProbHelper.build(surface_tokens_and_log_probs)
  result = {}
  for node_id in parsed_sequence.preorder_traversal:
    if node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
      token_node = parsed_sequence.text_token_nodes[node_id.index_in_category]
      result[node_id.preorder_index] = helper.advance_and_get_log_prob_for(
          token_node.text_contents
      )
    elif node_id.category == PackedSequenceNodeCategory.TEXT_DECORATION_NODE:
      decoration_node = parsed_sequence.text_decoration_nodes[
          node_id.index_in_category
      ]
      # We don't care about log probs for decorations, we just want to advance
      # past them.
      _ = helper.advance_and_get_log_prob_for(decoration_node.text_contents)

  if must_consume_all_model_tokens:
    helper.assert_consumed()

  return result


def flatten_token_log_probs_from_parsed(
    parsed_sequence: packed_sequence_nodes.PackedSequenceNodeStorage,
    token_log_probs_by_preorder_index: dict[int, float],
) -> Iterable[tuple[str, float]]:
  """Converts per-parsed-token log probs into a flat sequence, for debugging.

  Args:
    parsed_sequence: Packed representation of a parsed sequence or tree of
      programming language tokens.
    token_log_probs_by_preorder_index: Dictionary of per-parsed-token
      probabilities, as constructed by `align_token_log_probs`

  Yields:
    Tokens and their effective log probs, in the same format as the
    input to `align_token_log_probs`, but using the token boundaries from the
    parsed sequence.
  """
  for node_id in parsed_sequence.preorder_traversal:
    if node_id.category == PackedSequenceNodeCategory.TEXT_TOKEN_NODE:
      token_node = parsed_sequence.text_token_nodes[node_id.index_in_category]
      log_prob = token_log_probs_by_preorder_index[node_id.preorder_index]
      yield token_node.text_contents, log_prob

    elif node_id.category == PackedSequenceNodeCategory.TEXT_DECORATION_NODE:
      decoration_node = parsed_sequence.text_decoration_nodes[
          node_id.index_in_category
      ]
      # We don't store log probs for decorations, so we pretend they have
      # probability 1.
      yield decoration_node.text_contents, 0.0


def render_tokens_and_log_probs_to_html(
    surface_tokens_and_log_probs: list[tuple[str, float]],
) -> str:
  """Renders a list of tokens and their log-probs to a HTML source string."""
  parts = []
  parts.append(
      '<span style="white-space: pre; font-family: monospace; font-weight:'
      ' bold">'
  )
  for token, log_prob in surface_tokens_and_log_probs:
    prob = math.exp(log_prob)
    # prob == 0 -> red, prob==1 -> white
    r = 1.0
    g = prob
    b = prob
    escaped_token = html.escape(token.replace("\n", "↩\n"))
    parts.append(
        f'<span style="background-color: rgb({r * 100}%, {g * 100}%,'
        f' {b * 100}%)">{escaped_token}</span>'
    )
  parts.append("</span>")
  return "".join(parts)
