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

"""Tests for python_trim_for_parsing."""

from typing import Any, List, Optional, Tuple, Union

from absl.testing import absltest
from absl.testing import parameterized
from r_u_sure.parsing.pseudo_parser import stack_parser


def drop_trailing_whitespace(s: str) -> str:
  return '\n'.join([line.rstrip() for line in s.split('\n')])


class PseudoParserTest(parameterized.TestCase):

  def assert_equal_and_print(self, actual: Any, expected: Any,
                             ignore_trailing_whitespace: bool = True) -> None:

    if expected is None:
      print('expected not defined, set to the following if it is correct:')
      print(actual)
      print('with repr:')
      print(repr(actual))
      return
    try:
      if ignore_trailing_whitespace:
        assert isinstance(actual, str)
        assert isinstance(expected, str)
        actual_stripped_maybe = drop_trailing_whitespace(actual)
        expected_stripped_maybe = drop_trailing_whitespace(expected)
      else:
        actual_stripped_maybe = actual
        expected_stripped_maybe = expected
      self.assertEqual(actual_stripped_maybe, expected_stripped_maybe)
    except:
      print(f'actual:\n{drop_trailing_whitespace(actual)}')
      raise

  @parameterized.named_parameters(
      dict(
          testcase_name='first',
          tokens=list('ab{cd}{[e][f]}'),
          match_pairs=list(map(tuple, ['()', '[]', '{}'])),
          expected=r'''
| ab{cd}{[e][f]} | Node(0-14, root)
| a_____________ |    Node(0-1, sibling)
| _b____________ |    Node(1-2, sibling)
| __{cd}________ |    Node(2-6, paired)
| __{___________ |       Node(2-3, left)
| ___cd_________ |       Node(3-5, middle)
| ___c__________ |          Node(3-4, sibling)
| ____d_________ |          Node(4-5, sibling)
| _____}________ |       Node(5-6, right)
| ______{[e][f]} |    Node(6-14, paired)
| ______{_______ |       Node(6-7, left)
| _______[e][f]_ |       Node(7-13, middle)
| _______[e]____ |          Node(7-10, paired)
| _______[______ |             Node(7-8, left)
| ________e_____ |             Node(8-9, middle)
| ________e_____ |                Node(8-9, sibling)
| _________]____ |             Node(9-10, right)
| __________[f]_ |          Node(10-13, paired)
| __________[___ |             Node(10-11, left)
| ___________f__ |             Node(11-12, middle)
| ___________f__ |                Node(11-12, sibling)
| ____________]_ |             Node(12-13, right)
| _____________} |       Node(13-14, right)'''[1:],
      ),)
  def test_stack_parser(self, tokens: List[str], match_pairs: List[str],
                        expected: str):

    parsed = stack_parser.stack_based_bracket_match(tokens,
                                                    match_pairs)
    actual = parsed.human_readable_string(tokens)
    self.assert_equal_and_print(
        actual, expected, ignore_trailing_whitespace=True)
    self.assertTrue(
        stack_parser.valid_parse_ranges(parsed, len(tokens)))

  @parameterized.named_parameters(
      dict(
          testcase_name='first',
          tokens=list('abc)}[('),
          match_pairs=list(map(tuple, ['()', '[]', '{}'])),
          expected=r'''
| {(abc)}[()] | Node(0-11, root)
| {(abc)}____ |    Node(0-7, paired)
| {__________ |       Node(0-1, left)
| _(abc)_____ |       Node(1-6, middle)
| _(abc)_____ |          Node(1-6, paired)
| _(_________ |             Node(1-2, left)
| __abc______ |             Node(2-5, middle)
| __a________ |                Node(2-3, sibling)
| ___b_______ |                Node(3-4, sibling)
| ____c______ |                Node(4-5, sibling)
| _____)_____ |             Node(5-6, right)
| ______}____ |       Node(6-7, right)
| _______[()] |    Node(7-11, paired)
| _______[___ |       Node(7-8, left)
| ________()_ |       Node(8-10, middle)
| ________()_ |          Node(8-10, paired)
| ________(__ |             Node(8-9, left)
| ___________ |             Node(9-9, middle)
| _________)_ |             Node(9-10, right)
| __________] |       Node(10-11, right)'''[1:],
          expected_error_correcting_prefix=['{', '('],
          expected_error_correcting_suffix=[')', ']'],
      ),)
  def test_stack_parser_with_error_correction(
      self, tokens: List[str], match_pairs: List[str], expected: str,
      expected_error_correcting_prefix: List[str],
      expected_error_correcting_suffix: List[str]):

    self.assertRaises(stack_parser.ParseError,
                      stack_parser.stack_based_bracket_match,
                      tokens,
                      match_pairs)
    error_correcting_prefix, error_correcting_suffix = (
        stack_parser.error_correcting_prefix_and_suffix(tokens, match_pairs))
    self.assert_equal_and_print(
        error_correcting_prefix,
        expected_error_correcting_prefix,
        ignore_trailing_whitespace=False)
    self.assert_equal_and_print(
        error_correcting_suffix,
        expected_error_correcting_suffix,
        ignore_trailing_whitespace=False)
    tokens_maybe_corrected = (
        error_correcting_prefix + tokens + error_correcting_suffix)
    parsed = stack_parser.stack_based_bracket_match(tokens_maybe_corrected,
                                                    match_pairs)
    actual = parsed.human_readable_string(tokens_maybe_corrected)
    self.assert_equal_and_print(
        actual, expected, ignore_trailing_whitespace=True)
    self.assertTrue(
        stack_parser.valid_parse_ranges(parsed, len(tokens_maybe_corrected)))

  @parameterized.named_parameters(
      dict(
          testcase_name='strings',
          tokens=list('a;{(b;c;);{d;e;};};'),
          match_pairs=list(map(tuple, ['()', '[]', '{}'])),
          split_tokens=[';'],
          un_splittable_match_pairs=list(map(tuple, ['()'])),
          expected_unsplit='''
| a;{(b;c;);{d;e;};}; | Node(0-19, root)
| a__________________ |    Node(0-1, sibling)
| _;_________________ |    Node(1-2, sibling)
| __{(b;c;);{d;e;};}_ |    Node(2-18, paired)
| __{________________ |       Node(2-3, left)
| ___(b;c;);{d;e;};__ |       Node(3-17, middle)
| ___(b;c;)__________ |          Node(3-9, paired)
| ___(_______________ |             Node(3-4, left)
| ____b;c;___________ |             Node(4-8, middle)
| ____b______________ |                Node(4-5, sibling)
| _____;_____________ |                Node(5-6, sibling)
| ______c____________ |                Node(6-7, sibling)
| _______;___________ |                Node(7-8, sibling)
| ________)__________ |             Node(8-9, right)
| _________;_________ |          Node(9-10, sibling)
| __________{d;e;}___ |          Node(10-16, paired)
| __________{________ |             Node(10-11, left)
| ___________d;e;____ |             Node(11-15, middle)
| ___________d_______ |                Node(11-12, sibling)
| ____________;______ |                Node(12-13, sibling)
| _____________e_____ |                Node(13-14, sibling)
| ______________;____ |                Node(14-15, sibling)
| _______________}___ |             Node(15-16, right)
| ________________;__ |          Node(16-17, sibling)
| _________________}_ |       Node(17-18, right)
| __________________; |    Node(18-19, sibling)'''[1:],
          expected_split='''
| a;{(b;c;);{d;e;};}; | Node(0-19, root)
| a;_________________ |    Node(0-2, split)
| a__________________ |       Node(0-1, sibling)
| _;_________________ |       Node(1-2, sibling)
| __{(b;c;);{d;e;};}; |    Node(2-19, split)
| __{(b;c;);{d;e;};}_ |       Node(2-18, paired)
| __{________________ |          Node(2-3, left)
| ___(b;c;);{d;e;};__ |          Node(3-17, middle)
| ___(b;c;);_________ |             Node(3-10, split)
| ___(b;c;)__________ |                Node(3-9, paired)
| ___(_______________ |                   Node(3-4, left)
| ____b;c;___________ |                   Node(4-8, middle)
| ____b______________ |                      Node(4-5, sibling)
| _____;_____________ |                      Node(5-6, sibling)
| ______c____________ |                      Node(6-7, sibling)
| _______;___________ |                      Node(7-8, sibling)
| ________)__________ |                   Node(8-9, right)
| _________;_________ |                Node(9-10, sibling)
| __________{d;e;};__ |             Node(10-17, split)
| __________{d;e;}___ |                Node(10-16, paired)
| __________{________ |                   Node(10-11, left)
| ___________d;e;____ |                   Node(11-15, middle)
| ___________d;______ |                      Node(11-13, split)
| ___________d_______ |                         Node(11-12, sibling)
| ____________;______ |                         Node(12-13, sibling)
| _____________e;____ |                      Node(13-15, split)
| _____________e_____ |                         Node(13-14, sibling)
| ______________;____ |                         Node(14-15, sibling)
| _______________}___ |                   Node(15-16, right)
| ________________;__ |                Node(16-17, sibling)
| _________________}_ |          Node(17-18, right)
| __________________; |       Node(18-19, sibling)'''[1: ],
      ),
      dict(
          testcase_name='with_siblings',
          tokens=list('a (b c) ; d . e ; f g h i ;'),
          match_pairs=list(map(tuple, ['()'])),
          split_tokens=[';'],
          un_splittable_match_pairs=list(map(tuple, ['()'])),
          expected_unsplit='''
| a (b c) ; d . e ; f g h i ; | Node(0-27, root)
| a__________________________ |    Node(0-1, sibling)
| _ _________________________ |    Node(1-2, sibling)
| __(b c)____________________ |    Node(2-7, paired)
| __(________________________ |       Node(2-3, left)
| ___b c_____________________ |       Node(3-6, middle)
| ___b_______________________ |          Node(3-4, sibling)
| ____ ______________________ |          Node(4-5, sibling)
| _____c_____________________ |          Node(5-6, sibling)
| ______)____________________ |       Node(6-7, right)
| _______ ___________________ |    Node(7-8, sibling)
| ________;__________________ |    Node(8-9, sibling)
| _________ _________________ |    Node(9-10, sibling)
| __________d________________ |    Node(10-11, sibling)
| ___________ _______________ |    Node(11-12, sibling)
| ____________.______________ |    Node(12-13, sibling)
| _____________ _____________ |    Node(13-14, sibling)
| ______________e____________ |    Node(14-15, sibling)
| _______________ ___________ |    Node(15-16, sibling)
| ________________;__________ |    Node(16-17, sibling)
| _________________ _________ |    Node(17-18, sibling)
| __________________f________ |    Node(18-19, sibling)
| ___________________ _______ |    Node(19-20, sibling)
| ____________________g______ |    Node(20-21, sibling)
| _____________________ _____ |    Node(21-22, sibling)
| ______________________h____ |    Node(22-23, sibling)
| _______________________ ___ |    Node(23-24, sibling)
| ________________________i__ |    Node(24-25, sibling)
| _________________________ _ |    Node(25-26, sibling)
| __________________________; |    Node(26-27, sibling)'''[1:],
          expected_split='''
| a (b c) ; d . e ; f g h i ; | Node(0-27, root)
| a (b c) ;__________________ |    Node(0-9, split)
| a__________________________ |       Node(0-1, sibling)
| _ _________________________ |       Node(1-2, sibling)
| __(b c)____________________ |       Node(2-7, paired)
| __(________________________ |          Node(2-3, left)
| ___b c_____________________ |          Node(3-6, middle)
| ___b_______________________ |             Node(3-4, sibling)
| ____ ______________________ |             Node(4-5, sibling)
| _____c_____________________ |             Node(5-6, sibling)
| ______)____________________ |          Node(6-7, right)
| _______ ___________________ |       Node(7-8, sibling)
| ________;__________________ |       Node(8-9, sibling)
| _________ d . e ;__________ |    Node(9-17, split)
| _________ _________________ |       Node(9-10, sibling)
| __________d________________ |       Node(10-11, sibling)
| ___________ _______________ |       Node(11-12, sibling)
| ____________.______________ |       Node(12-13, sibling)
| _____________ _____________ |       Node(13-14, sibling)
| ______________e____________ |       Node(14-15, sibling)
| _______________ ___________ |       Node(15-16, sibling)
| ________________;__________ |       Node(16-17, sibling)
| _________________ f g h i ; |    Node(17-27, split)
| _________________ _________ |       Node(17-18, sibling)
| __________________f________ |       Node(18-19, sibling)
| ___________________ _______ |       Node(19-20, sibling)
| ____________________g______ |       Node(20-21, sibling)
| _____________________ _____ |       Node(21-22, sibling)
| ______________________h____ |       Node(22-23, sibling)
| _______________________ ___ |       Node(23-24, sibling)
| ________________________i__ |       Node(24-25, sibling)
| _________________________ _ |       Node(25-26, sibling)
| __________________________; |       Node(26-27, sibling)'''[1:],
      ),
      dict(
          testcase_name='vacuous',
          tokens=list('a;{(b;c;);{d;e;};};'),
          match_pairs=[],
          split_tokens=[],
          un_splittable_match_pairs=[],
          expected_unsplit='''
| a;{(b;c;);{d;e;};}; | Node(0-19, root)
| a__________________ |    Node(0-1, sibling)
| _;_________________ |    Node(1-2, sibling)
| __{________________ |    Node(2-3, sibling)
| ___(_______________ |    Node(3-4, sibling)
| ____b______________ |    Node(4-5, sibling)
| _____;_____________ |    Node(5-6, sibling)
| ______c____________ |    Node(6-7, sibling)
| _______;___________ |    Node(7-8, sibling)
| ________)__________ |    Node(8-9, sibling)
| _________;_________ |    Node(9-10, sibling)
| __________{________ |    Node(10-11, sibling)
| ___________d_______ |    Node(11-12, sibling)
| ____________;______ |    Node(12-13, sibling)
| _____________e_____ |    Node(13-14, sibling)
| ______________;____ |    Node(14-15, sibling)
| _______________}___ |    Node(15-16, sibling)
| ________________;__ |    Node(16-17, sibling)
| _________________}_ |    Node(17-18, sibling)
| __________________; |    Node(18-19, sibling)'''[1:],
          expected_split='''
| a;{(b;c;);{d;e;};}; | Node(0-19, root)
| a__________________ |    Node(0-1, sibling)
| _;_________________ |    Node(1-2, sibling)
| __{________________ |    Node(2-3, sibling)
| ___(_______________ |    Node(3-4, sibling)
| ____b______________ |    Node(4-5, sibling)
| _____;_____________ |    Node(5-6, sibling)
| ______c____________ |    Node(6-7, sibling)
| _______;___________ |    Node(7-8, sibling)
| ________)__________ |    Node(8-9, sibling)
| _________;_________ |    Node(9-10, sibling)
| __________{________ |    Node(10-11, sibling)
| ___________d_______ |    Node(11-12, sibling)
| ____________;______ |    Node(12-13, sibling)
| _____________e_____ |    Node(13-14, sibling)
| ______________;____ |    Node(14-15, sibling)
| _______________}___ |    Node(15-16, sibling)
| ________________;__ |    Node(16-17, sibling)
| _________________}_ |    Node(17-18, sibling)
| __________________; |    Node(18-19, sibling)'''[1:],
      ),
      dict(
          testcase_name='integers',
          tokens=[0, 4, 8, 5, 1, 0, 6, 1, 2, 7, 8, 7, 3],
          match_pairs=[(0, 1), (2, 3)],
          split_tokens=[8],
          un_splittable_match_pairs=[(2, 3)],
          expected_unsplit='''
| 0485106127873 | Node(0-13, root)
| 04851________ |    Node(0-5, paired)
| 0____________ |       Node(0-1, left)
| _485_________ |       Node(1-4, middle)
| _4___________ |          Node(1-2, sibling)
| __8__________ |          Node(2-3, sibling)
| ___5_________ |          Node(3-4, sibling)
| ____1________ |       Node(4-5, right)
| _____061_____ |    Node(5-8, paired)
| _____0_______ |       Node(5-6, left)
| ______6______ |       Node(6-7, middle)
| ______6______ |          Node(6-7, sibling)
| _______1_____ |       Node(7-8, right)
| ________27873 |    Node(8-13, paired)
| ________2____ |       Node(8-9, left)
| _________787_ |       Node(9-12, middle)
| _________7___ |          Node(9-10, sibling)
| __________8__ |          Node(10-11, sibling)
| ___________7_ |          Node(11-12, sibling)
| ____________3 |       Node(12-13, right)'''[1:],
          expected_split='''
| 0485106127873 | Node(0-13, root)
| 04851________ |    Node(0-5, paired)
| 0____________ |       Node(0-1, left)
| _485_________ |       Node(1-4, middle)
| _48__________ |          Node(1-3, split)
| _4___________ |             Node(1-2, sibling)
| __8__________ |             Node(2-3, sibling)
| ___5_________ |          Node(3-4, split)
| ___5_________ |             Node(3-4, sibling)
| ____1________ |       Node(4-5, right)
| _____061_____ |    Node(5-8, paired)
| _____0_______ |       Node(5-6, left)
| ______6______ |       Node(6-7, middle)
| ______6______ |          Node(6-7, sibling)
| _______1_____ |       Node(7-8, right)
| ________27873 |    Node(8-13, paired)
| ________2____ |       Node(8-9, left)
| _________787_ |       Node(9-12, middle)
| _________7___ |          Node(9-10, sibling)
| __________8__ |          Node(10-11, sibling)
| ___________7_ |          Node(11-12, sibling)
| ____________3 |       Node(12-13, right)'''[1:],
      ),
  )
  def test_splitting(self,
                     tokens: List[Union[int, str]],
                     match_pairs: List[Tuple[Union[int, str], Union[int, str]]],
                     split_tokens: List[Union[int, str]],
                     un_splittable_match_pairs:
                     List[Tuple[Union[int, str], Union[int, str]]],
                     expected_unsplit: str, expected_split: str):

    parsed_unsplit = stack_parser.stack_based_bracket_match(tokens, match_pairs)
    actual_unsplit = parsed_unsplit.human_readable_string(tokens)
    self.assert_equal_and_print(
        actual_unsplit, expected_unsplit, ignore_trailing_whitespace=True)
    self.assertTrue(
        stack_parser.valid_parse_ranges(parsed_unsplit, len(tokens)))
    parsed_split = stack_parser.split_parse(
        parsed_unsplit, tokens, split_tokens, un_splittable_match_pairs)
    actual_split = parsed_split.human_readable_string(tokens)
    self.assert_equal_and_print(
        actual_split, expected_split, ignore_trailing_whitespace=True)
    self.assertTrue(stack_parser.valid_parse_ranges(parsed_split, len(tokens)))

  @parameterized.named_parameters(
      dict(
          testcase_name='first',
          code=' aa bb1  \t\n cc',
          expected_decoded_tokenised_code=[
              ' ', 'aa', ' ', 'bb1', '  ', '\t', '\n', ' ', 'cc'],
          lexer_preset='simple',
      ),
      dict(
          testcase_name='second',
          code=' aa bb1\tf(xx,yy , zz)',
          expected_decoded_tokenised_code=[
              ' ', 'aa', ' ', 'bb1', '\t', 'f', '(', 'xx', ',', 'yy', ' ', ',',
              ' ', 'zz', ')'],
          lexer_preset='simple',
      ),
      dict(
          testcase_name='third',
          code=' aa1 bb2 cc(a,b)=12; \n hello ^^',
          expected_decoded_tokenised_code=[
              ' ', 'aa1', ' ', 'bb2', ' ', 'cc', '(', 'a', ',', 'b', ')', '=',
              '12', ';', ' ', '\n', ' ', 'hello', ' ', '^', '^'],
          lexer_preset='simple',
      ),
      dict(
          testcase_name='python_comment',
          code='a=1\nbb=2 # comment\nhello',
          expected_decoded_tokenised_code=['a', '=', '1', '\n', 'bb', '=', '2',
                                           ' ', '# comment', '\n', 'hello'],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='python_string_one',
          code='a=1\nx=\'a string\'  # hello\n',
          expected_decoded_tokenised_code=['a', '=', '1', '\n', 'x', '=',
                                           "'a string'", '  ', '# hello', '\n'],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='python_string_two',
          code='a=1\nx=\'a  \\\' string\'  # hello\n',
          expected_decoded_tokenised_code=[
              'a', '=', '1', '\n', 'x', '=', "'a  \\' string'", '  ', '# hello',
              '\n'],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='python_string_three',
          code='a=1\nx=\'\'\'a string\'\'\'',
          expected_decoded_tokenised_code=[
              'a', '=', '1', '\n', 'x', '=', "'''a string'''"],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='python_string_four',
          code='a=1\nx=\'\'\'a \' string\'\'\'',
          expected_decoded_tokenised_code=[
              'a', '=', '1', '\n', 'x', '=', "'''a ' string'''"],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='python_string_five',
          code='x="a \' string"',
          expected_decoded_tokenised_code=['x', '=', '"a \' string"'],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='python_string_six',
          code='x="""a string"""',
          expected_decoded_tokenised_code=['x', '=', '"""a string"""'],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='python_string_seven',
          code='x="""a \'string"""',
          expected_decoded_tokenised_code=['x', '=', '"""a \'string"""'],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='python_string_eight',
          code='x="""a \' \'\' \'\'\' " "" string"""',
          expected_decoded_tokenised_code=[
              'x', '=', '"""a \' \'\' \'\'\' " "" string"""'],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='python_string_nine',
          code='x="""a \n string"""',
          expected_decoded_tokenised_code=['x', '=', '"""a \n string"""'],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='python_not_comment_one',
          code='x="""a \n # string"""',
          expected_decoded_tokenised_code=['x', '=', '"""a \n # string"""'],
          lexer_preset='python_preprocessor',
      ),
      dict(
          testcase_name='cpp_string_one',
          code='x="a string"',
          expected_decoded_tokenised_code=['x', '=', '"a string"'],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_string_two',
          code='x="a \\"string"',
          expected_decoded_tokenised_code=['x', '=', '"a \\"string"'],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_comment_one',
          code='x=2 // a comment',
          expected_decoded_tokenised_code=['x', '=', '2', ' ', '// a comment'],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_comment_two',
          code='x=2 /* a comment */',
          expected_decoded_tokenised_code=[
              'x', '=', '2', ' ', '/* a comment */'],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_comment_three',
          code='x=2 /* a \n comment */',
          expected_decoded_tokenised_code=[
              'x', '=', '2', ' ', '/* a \n comment */'],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_not_comment_one',
          code='x=2 * "not a // a comment"',
          expected_decoded_tokenised_code=[
              'x', '=', '2', ' ', '*', ' ', '"not a // a comment"'],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_hash_include_one',
          code='#include a\nx=1',
          expected_decoded_tokenised_code=[
              '#include a', '\n', 'x', '=', '1'],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_misc_one',
          code='x=1;\n// hello\ny=2;',
          expected_decoded_tokenised_code=[
              'x', '=', '1', ';', '\n', '// hello', '\n', 'y', '=', '2', ';'],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_preprocessor_one',
          code='x=1;#include abc\n#define X Y\ny=2;',
          expected_decoded_tokenised_code=[
              'x', '=', '1', ';', '#include abc', '\n', '#define X Y', '\n',
              'y', '=', '2', ';'
          ],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_preprocessor_comment_one',
          code='x=1;#include abc // comment\ny=2;',
          expected_decoded_tokenised_code=[
              'x', '=', '1', ';', '#include abc ', '// comment', '\n', 'y', '=',
              '2', ';'
          ],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_preprocessor_comment_two',
          code='x=1;#include abc /* comment\ncomment */',
          expected_decoded_tokenised_code=[
              'x', '=', '1', ';', '#include abc ', '/* comment\ncomment */'
          ],
          lexer_preset='cpp',
      ),
      dict(
          testcase_name='cpp_escaped_newline',
          code='#include xyz \\\ndef\n',
          expected_decoded_tokenised_code=[
              '#include xyz \\\ndef', '\n'
          ],
          lexer_preset='cpp',
      ),
  )
  def test_codec(self, code: str, lexer_preset: str,
                 expected_decoded_tokenised_code: str):

    codec = stack_parser.RegexCodec.preset(lexer_preset)
    token_list, token_types = codec.encode(code)
    decoded_tokenised_code = codec.decode_token(token_list)

    if expected_decoded_tokenised_code is None:
      print(code)
      print(token_types)
    self.assert_equal_and_print(
        decoded_tokenised_code,
        expected_decoded_tokenised_code,
        ignore_trailing_whitespace=False)
    self.assert_equal_and_print(
        ''.join(decoded_tokenised_code), code, ignore_trailing_whitespace=True)

  @parameterized.named_parameters(
      dict(
          testcase_name='second',
          code='({aa]aa',
          language='simple',
          expected=r'''
| ({aa]aa}) | Node(0-7, root)
| ({aa]aa}) |    Node(0-7, paired)
| (________ |       Node(0-1, left)
| _{aa]aa}_ |       Node(1-6, middle)
| _{aa]aa}_ |          Node(1-6, paired)
| _{_______ |             Node(1-2, left)
| __aa]aa__ |             Node(2-5, middle)
| __aa_____ |                Node(2-3, sibling)
| ____]____ |                Node(3-4, sibling)
| _____aa__ |                Node(4-5, sibling)
| _______}_ |             Node(5-6, right)
| ________) |       Node(6-7, right)'''[1:],
      ),
      dict(
          testcase_name='third',
          code='(])',
          language='simple',
          expected=r'''
| (]) | Node(0-3, root)
| (]) |    Node(0-3, paired)
| (__ |       Node(0-1, left)
| _]_ |       Node(1-2, middle)
| _]_ |          Node(1-2, sibling)
| __) |       Node(2-3, right)'''[1:],
      ),
      dict(
          testcase_name='fourth',
          code='([)',
          language='simple',
          expected=r'''
| ([)]) | Node(0-5, root)
| ([)]) |    Node(0-5, paired)
| (____ |       Node(0-1, left)
| _[)]_ |       Node(1-4, middle)
| _[)]_ |          Node(1-4, paired)
| _[___ |             Node(1-2, left)
| __)__ |             Node(2-3, middle)
| __)__ |                Node(2-3, sibling)
| ___]_ |             Node(3-4, right)
| ____) |       Node(4-5, right)'''[1:],
      ),
      dict(
          testcase_name='fifth',
          code='''
class A(object):
  def __init__(self, arg1,
         arg2,
          arg3):

    x=1
    y=2
    if arg1:

      x++
      return x
    return arg2
'''[1:],
          language='python',
          expected=r'''
| class A(object):\n  def __init__(self, arg1,\n         arg2,\n          arg3):\n\n    x=1\n    y=2\n    if arg1:\n\n      x++\n      return x\n    return arg2\n | Node(0-67, root)
| class A(object):\n  def __init__(self, arg1,\n         arg2,\n          arg3):\n\n    x=1\n    y=2\n    if arg1:\n\n      x++\n      return x\n    return arg2\n |    Node(0-67, split)
| class A(object):\n______________________________________________________________________________________________________________________________________________ |       Node(0-8, split)
| class___________________________________________________________________________________________________________________________________________________________ |          Node(0-1, sibling)
| _____ __________________________________________________________________________________________________________________________________________________________ |          Node(1-2, sibling)
| ______A_________________________________________________________________________________________________________________________________________________________ |          Node(2-3, sibling)
| _______(object)_________________________________________________________________________________________________________________________________________________ |          Node(3-6, paired)
| _______(________________________________________________________________________________________________________________________________________________________ |             Node(3-4, left)
| ________object__________________________________________________________________________________________________________________________________________________ |             Node(4-5, middle)
| ________object__________________________________________________________________________________________________________________________________________________ |                Node(4-5, sibling)
| ______________)_________________________________________________________________________________________________________________________________________________ |             Node(5-6, right)
| _______________:________________________________________________________________________________________________________________________________________________ |          Node(6-7, sibling)
| ________________\n______________________________________________________________________________________________________________________________________________ |          Node(7-8, sibling)
| __________________  def __init__(self, arg1,\n         arg2,\n          arg3):\n\n    x=1\n    y=2\n    if arg1:\n\n      x++\n      return x\n    return arg2\n |       Node(8-67, split)
| __________________  def __init__(self, arg1,\n         arg2,\n          arg3):\n\n    x=1\n    y=2\n    if arg1:\n\n      x++\n      return x\n    return arg2__ |          Node(8-66, paired)
| ________________________________________________________________________________________________________________________________________________________________ |             Node(8-9, left)
| __________________  def __init__(self, arg1,\n         arg2,\n          arg3):\n\n    x=1\n    y=2\n    if arg1:\n\n      x++\n      return x\n    return arg2__ |             Node(9-65, middle)
| __________________  def __init__(self, arg1,\n         arg2,\n          arg3):\n________________________________________________________________________________ |                Node(9-29, split)
| __________________  ____________________________________________________________________________________________________________________________________________ |                   Node(9-10, sibling)
| ____________________def_________________________________________________________________________________________________________________________________________ |                   Node(10-11, sibling)
| _______________________ ________________________________________________________________________________________________________________________________________ |                   Node(11-12, sibling)
| __________________________init__________________________________________________________________________________________________________________________________ |                   Node(12-13, sibling)
| ________________________________(self, arg1,\n         arg2,\n          arg3)___________________________________________________________________________________ |                   Node(13-27, paired)
| ________________________________(_______________________________________________________________________________________________________________________________ |                      Node(13-14, left)
| _________________________________self, arg1,\n         arg2,\n          arg3____________________________________________________________________________________ |                      Node(14-26, middle)
| _________________________________self___________________________________________________________________________________________________________________________ |                         Node(14-15, sibling)
| _____________________________________,__________________________________________________________________________________________________________________________ |                         Node(15-16, sibling)
| ______________________________________ _________________________________________________________________________________________________________________________ |                         Node(16-17, sibling)
| _______________________________________arg1_____________________________________________________________________________________________________________________ |                         Node(17-18, sibling)
| ___________________________________________,____________________________________________________________________________________________________________________ |                         Node(18-19, sibling)
| ____________________________________________\n__________________________________________________________________________________________________________________ |                         Node(19-20, sibling)
| ______________________________________________         _________________________________________________________________________________________________________ |                         Node(20-21, sibling)
| _______________________________________________________arg2_____________________________________________________________________________________________________ |                         Node(21-22, sibling)
| ___________________________________________________________,____________________________________________________________________________________________________ |                         Node(22-23, sibling)
| ____________________________________________________________\n__________________________________________________________________________________________________ |                         Node(23-24, sibling)
| ______________________________________________________________          ________________________________________________________________________________________ |                         Node(24-25, sibling)
| ________________________________________________________________________arg3____________________________________________________________________________________ |                         Node(25-26, sibling)
| ____________________________________________________________________________)___________________________________________________________________________________ |                      Node(26-27, right)
| _____________________________________________________________________________:__________________________________________________________________________________ |                   Node(27-28, sibling)
| ______________________________________________________________________________\n________________________________________________________________________________ |                   Node(28-29, sibling)
| ________________________________________________________________________________\n    x=1\n    y=2\n    if arg1:\n\n      x++\n      return x\n    return arg2__ |                Node(29-65, split)
| ________________________________________________________________________________\n______________________________________________________________________________ |                   Node(29-30, split)
| ________________________________________________________________________________\n______________________________________________________________________________ |                      Node(29-30, sibling)
| __________________________________________________________________________________    x=1\n    y=2\n    if arg1:\n\n      x++\n      return x\n    return arg2__ |                   Node(30-65, split)
| __________________________________________________________________________________    x=1\n    y=2\n    if arg1:\n\n      x++\n      return x\n    return arg2__ |                      Node(30-65, paired)
| ________________________________________________________________________________________________________________________________________________________________ |                         Node(30-31, left)
| __________________________________________________________________________________    x=1\n    y=2\n    if arg1:\n\n      x++\n      return x\n    return arg2__ |                         Node(31-64, middle)
| __________________________________________________________________________________    x=1\n_____________________________________________________________________ |                            Node(31-36, split)
| __________________________________________________________________________________    __________________________________________________________________________ |                               Node(31-32, sibling)
| ______________________________________________________________________________________x_________________________________________________________________________ |                               Node(32-33, sibling)
| _______________________________________________________________________________________=________________________________________________________________________ |                               Node(33-34, sibling)
| ________________________________________________________________________________________1_______________________________________________________________________ |                               Node(34-35, sibling)
| _________________________________________________________________________________________\n_____________________________________________________________________ |                               Node(35-36, sibling)
| ___________________________________________________________________________________________    y=2\n____________________________________________________________ |                            Node(36-41, split)
| ___________________________________________________________________________________________    _________________________________________________________________ |                               Node(36-37, sibling)
| _______________________________________________________________________________________________y________________________________________________________________ |                               Node(37-38, sibling)
| ________________________________________________________________________________________________=_______________________________________________________________ |                               Node(38-39, sibling)
| _________________________________________________________________________________________________2______________________________________________________________ |                               Node(39-40, sibling)
| __________________________________________________________________________________________________\n____________________________________________________________ |                               Node(40-41, sibling)
| ____________________________________________________________________________________________________    if arg1:\n______________________________________________ |                            Node(41-47, split)
| ____________________________________________________________________________________________________    ________________________________________________________ |                               Node(41-42, sibling)
| ________________________________________________________________________________________________________if______________________________________________________ |                               Node(42-43, sibling)
| __________________________________________________________________________________________________________ _____________________________________________________ |                               Node(43-44, sibling)
| ___________________________________________________________________________________________________________arg1_________________________________________________ |                               Node(44-45, sibling)
| _______________________________________________________________________________________________________________:________________________________________________ |                               Node(45-46, sibling)
| ________________________________________________________________________________________________________________\n______________________________________________ |                               Node(46-47, sibling)
| __________________________________________________________________________________________________________________\n      x++\n      return x\n_________________ |                            Node(47-60, split)
| __________________________________________________________________________________________________________________\n____________________________________________ |                               Node(47-48, split)
| __________________________________________________________________________________________________________________\n____________________________________________ |                                  Node(47-48, sibling)
| ____________________________________________________________________________________________________________________      x++\n      return x\n_________________ |                               Node(48-60, split)
| ____________________________________________________________________________________________________________________      x++\n      return x___________________ |                                  Node(48-59, paired)
| ________________________________________________________________________________________________________________________________________________________________ |                                     Node(48-49, left)
| ____________________________________________________________________________________________________________________      x++\n      return x___________________ |                                     Node(49-58, middle)
| ____________________________________________________________________________________________________________________      x++\n_________________________________ |                                        Node(49-54, split)
| ____________________________________________________________________________________________________________________      ______________________________________ |                                           Node(49-50, sibling)
| __________________________________________________________________________________________________________________________x_____________________________________ |                                           Node(50-51, sibling)
| ___________________________________________________________________________________________________________________________+____________________________________ |                                           Node(51-52, sibling)
| ____________________________________________________________________________________________________________________________+___________________________________ |                                           Node(52-53, sibling)
| _____________________________________________________________________________________________________________________________\n_________________________________ |                                           Node(53-54, sibling)
| _______________________________________________________________________________________________________________________________      return x___________________ |                                        Node(54-58, split)
| _______________________________________________________________________________________________________________________________      ___________________________ |                                           Node(54-55, sibling)
| _____________________________________________________________________________________________________________________________________return_____________________ |                                           Node(55-56, sibling)
| ___________________________________________________________________________________________________________________________________________ ____________________ |                                           Node(56-57, sibling)
| ____________________________________________________________________________________________________________________________________________x___________________ |                                           Node(57-58, sibling)
| ________________________________________________________________________________________________________________________________________________________________ |                                     Node(58-59, right)
| _____________________________________________________________________________________________________________________________________________\n_________________ |                                  Node(59-60, sibling)
| _______________________________________________________________________________________________________________________________________________    return arg2__ |                            Node(60-64, split)
| _______________________________________________________________________________________________________________________________________________    _____________ |                               Node(60-61, sibling)
| ___________________________________________________________________________________________________________________________________________________return_______ |                               Node(61-62, sibling)
| _________________________________________________________________________________________________________________________________________________________ ______ |                               Node(62-63, sibling)
| __________________________________________________________________________________________________________________________________________________________arg2__ |                               Node(63-64, sibling)
| ________________________________________________________________________________________________________________________________________________________________ |                         Node(64-65, right)
| ________________________________________________________________________________________________________________________________________________________________ |             Node(65-66, right)
| ______________________________________________________________________________________________________________________________________________________________\n |          Node(66-67, sibling)'''[1:],  # pylint: disable=line-too-long
      ),
      dict(
          testcase_name='sixth',
          code='    return x\n  def f(self.x):\n    return self.y',
          language='python',
          expected=r'''
|     return x\n  def f(self.x):\n    return self.y\n | Node(0-29, root)
|     return x\n  def f(self.x):\n    return self.y\n |    Node(0-29, split)
|     return x\n  def f(self.x):\n    return self.y__ |       Node(0-28, paired)
| ___________________________________________________ |          Node(0-1, left)
|     return x\n  def f(self.x):\n    return self.y__ |          Node(1-27, middle)
|     return x\n_____________________________________ |             Node(1-8, split)
|     return x_______________________________________ |                Node(1-7, paired)
| ___________________________________________________ |                   Node(1-2, left)
|     return x_______________________________________ |                   Node(2-6, middle)
|     _______________________________________________ |                      Node(2-3, sibling)
| ____return_________________________________________ |                      Node(3-4, sibling)
| __________ ________________________________________ |                      Node(4-5, sibling)
| ___________x_______________________________________ |                      Node(5-6, sibling)
| ___________________________________________________ |                   Node(6-7, right)
| ____________\n_____________________________________ |                Node(7-8, sibling)
| ______________  def f(self.x):\n    return self.y__ |             Node(8-27, split)
| ______________  def f(self.x):\n___________________ |                Node(8-19, split)
| ______________  ___________________________________ |                   Node(8-9, sibling)
| ________________def________________________________ |                   Node(9-10, sibling)
| ___________________ _______________________________ |                   Node(10-11, sibling)
| ____________________f______________________________ |                   Node(11-12, sibling)
| _____________________(self.x)______________________ |                   Node(12-17, paired)
| _____________________(_____________________________ |                      Node(12-13, left)
| ______________________self.x_______________________ |                      Node(13-16, middle)
| ______________________self_________________________ |                         Node(13-14, sibling)
| __________________________.________________________ |                         Node(14-15, sibling)
| ___________________________x_______________________ |                         Node(15-16, sibling)
| ____________________________)______________________ |                      Node(16-17, right)
| _____________________________:_____________________ |                   Node(17-18, sibling)
| ______________________________\n___________________ |                   Node(18-19, sibling)
| ________________________________    return self.y__ |                Node(19-27, split)
| ________________________________    return self.y__ |                   Node(19-27, paired)
| ___________________________________________________ |                      Node(19-20, left)
| ________________________________    return self.y__ |                      Node(20-26, middle)
| ________________________________    _______________ |                         Node(20-21, sibling)
| ____________________________________return_________ |                         Node(21-22, sibling)
| __________________________________________ ________ |                         Node(22-23, sibling)
| ___________________________________________self____ |                         Node(23-24, sibling)
| _______________________________________________.___ |                         Node(24-25, sibling)
| ________________________________________________y__ |                         Node(25-26, sibling)
| ___________________________________________________ |                      Node(26-27, right)
| ___________________________________________________ |          Node(27-28, right)
| _________________________________________________\n |       Node(28-29, sibling)'''[1:],  # pylint: disable=line-too-long
      ),
      dict(
          testcase_name='seventh',
          code='''
  f(

'''[1:],
          language='python',
          expected=r'''
|   f(\n\n)\n | Node(0-9, root)
|   f(\n\n)\n |    Node(0-9, split)
|   f(\n\n)__ |       Node(0-8, paired)
| ___________ |          Node(0-1, left)
|   f(\n\n)__ |          Node(1-7, middle)
|   _________ |             Node(1-2, sibling)
| __f________ |             Node(2-3, sibling)
| ___(\n\n)__ |             Node(3-7, paired)
| ___(_______ |                Node(3-4, left)
| ____\n\n___ |                Node(4-6, middle)
| ____\n_____ |                   Node(4-5, sibling)
| ______\n___ |                   Node(5-6, sibling)
| ________)__ |                Node(6-7, right)
| ___________ |          Node(7-8, right)
| _________\n |       Node(8-9, sibling)'''[1:],
      ),
      dict(
          testcase_name='cpp_one',
          code='x=1;\n#define A B // comment',
          language='cpp',
          expected=r'''
| x=1;\n#define A B // comment | Node(0-7, root)
| x=1;________________________ |    Node(0-4, split)
| x___________________________ |       Node(0-1, sibling)
| _=__________________________ |       Node(1-2, sibling)
| __1_________________________ |       Node(2-3, sibling)
| ___;________________________ |       Node(3-4, sibling)
| ____\n#define A B __________ |    Node(4-6, split)
| ____\n______________________ |       Node(4-5, sibling)
| ______#define A B __________ |       Node(5-6, sibling)
| __________________// comment |    Node(6-7, split)
| __________________// comment |       Node(6-7, sibling)'''[1:],
      ),
      dict(
          testcase_name='cpp_two',
          code='for(int x;x=0;x++){for(y){a;b;}}',
          language='cpp',
          expected='''
| for(int x;x=0;x++){for(y){a;b;}} | Node(0-26, root)
| for_____________________________ |    Node(0-1, sibling)
| ___(int x;x=0;x++)______________ |    Node(1-14, paired)
| ___(____________________________ |       Node(1-2, left)
| ____int x;x=0;x++_______________ |       Node(2-13, middle)
| ____int x;______________________ |          Node(2-6, split)
| ____int_________________________ |             Node(2-3, sibling)
| _______ ________________________ |             Node(3-4, sibling)
| ________x_______________________ |             Node(4-5, sibling)
| _________;______________________ |             Node(5-6, sibling)
| __________x=0;__________________ |          Node(6-10, split)
| __________x_____________________ |             Node(6-7, sibling)
| ___________=____________________ |             Node(7-8, sibling)
| ____________0___________________ |             Node(8-9, sibling)
| _____________;__________________ |             Node(9-10, sibling)
| ______________x++_______________ |          Node(10-13, split)
| ______________x_________________ |             Node(10-11, sibling)
| _______________+________________ |             Node(11-12, sibling)
| ________________+_______________ |             Node(12-13, sibling)
| _________________)______________ |       Node(13-14, right)
| __________________{for(y){a;b;}} |    Node(14-26, paired)
| __________________{_____________ |       Node(14-15, left)
| ___________________for(y){a;b;}_ |       Node(15-25, middle)
| ___________________for__________ |          Node(15-16, sibling)
| ______________________(y)_______ |          Node(16-19, paired)
| ______________________(_________ |             Node(16-17, left)
| _______________________y________ |             Node(17-18, middle)
| _______________________y________ |                Node(17-18, sibling)
| ________________________)_______ |             Node(18-19, right)
| _________________________{a;b;}_ |          Node(19-25, paired)
| _________________________{______ |             Node(19-20, left)
| __________________________a;b;__ |             Node(20-24, middle)
| __________________________a;____ |                Node(20-22, split)
| __________________________a_____ |                   Node(20-21, sibling)
| ___________________________;____ |                   Node(21-22, sibling)
| ____________________________b;__ |                Node(22-24, split)
| ____________________________b___ |                   Node(22-23, sibling)
| _____________________________;__ |                   Node(23-24, sibling)
| ______________________________}_ |             Node(24-25, right)
| _______________________________} |       Node(25-26, right)'''[1:],
      ),
      dict(
          testcase_name='cpp_escaped_newline',
          code='#define X \\\n Y\n#define A B\n',
          # code='#define X \\ \n Y\n#define A B\n',
          language='cpp',
          expected=r'''
| #define X \\n Y\n#define A B\n | Node(0-4, root)
| #define X \\n Y_______________ |    Node(0-1, split)
| #define X \\n Y_______________ |       Node(0-1, sibling)
| _______________\n#define A B__ |    Node(1-3, split)
| _______________\n_____________ |       Node(1-2, sibling)
| _________________#define A B__ |       Node(2-3, sibling)
| ____________________________\n |    Node(3-4, split)
| ____________________________\n |       Node(3-4, sibling)'''[1:],
      ),
      dict(  # functions signatures and their blocks are not split
          testcase_name='java_blindspot_one',
          code='public f(){x=1;}public g(){y=2;}',
          language='java',
          expected='''
| public f(){x=1;}public g(){y=2;} | Node(0-22, root)
| public__________________________ |    Node(0-1, sibling)
| ______ _________________________ |    Node(1-2, sibling)
| _______f________________________ |    Node(2-3, sibling)
| ________()______________________ |    Node(3-5, paired)
| ________(_______________________ |       Node(3-4, left)
| ________________________________ |       Node(4-4, middle)
| _________)______________________ |       Node(4-5, right)
| __________{x=1;}________________ |    Node(5-11, paired)
| __________{_____________________ |       Node(5-6, left)
| ___________x=1;_________________ |       Node(6-10, middle)
| ___________x=1;_________________ |          Node(6-10, split)
| ___________x____________________ |             Node(6-7, sibling)
| ____________=___________________ |             Node(7-8, sibling)
| _____________1__________________ |             Node(8-9, sibling)
| ______________;_________________ |             Node(9-10, sibling)
| _______________}________________ |       Node(10-11, right)
| ________________public__________ |    Node(11-12, sibling)
| ______________________ _________ |    Node(12-13, sibling)
| _______________________g________ |    Node(13-14, sibling)
| ________________________()______ |    Node(14-16, paired)
| ________________________(_______ |       Node(14-15, left)
| ________________________________ |       Node(15-15, middle)
| _________________________)______ |       Node(15-16, right)
| __________________________{y=2;} |    Node(16-22, paired)
| __________________________{_____ |       Node(16-17, left)
| ___________________________y=2;_ |       Node(17-21, middle)
| ___________________________y=2;_ |          Node(17-21, split)
| ___________________________y____ |             Node(17-18, sibling)
| ____________________________=___ |             Node(18-19, sibling)
| _____________________________2__ |             Node(19-20, sibling)
| ______________________________;_ |             Node(20-21, sibling)
| _______________________________} |       Node(21-22, right)'''[1:],
      ),
      dict(
          testcase_name='py_no_terminating_newline_and_error_correcting_suffix',
          code='f(x',
          language='python',
          expected=r'''
| f(x\n)\n | Node(0-6, root)
| f(x\n)\n |    Node(0-6, split)
| f_______ |       Node(0-1, sibling)
| _(x\n)__ |       Node(1-5, paired)
| _(______ |          Node(1-2, left)
| __x\n___ |          Node(2-4, middle)
| __x_____ |             Node(2-3, sibling)
| ___\n___ |             Node(3-4, sibling)
| _____)__ |          Node(4-5, right)
| ______\n |       Node(5-6, sibling)'''[1:],
      ),
      dict(
          testcase_name='py_class_with_two_functions',
          code='class A:\n  def f(self):\n    F()\n  def g(self):\n    G()',
          language='python',
          expected=r'''
| class A:\n  def f(self):\n    F()\n  def g(self):\n    G()\n | Node(0-39, root)
| class A:\n  def f(self):\n    F()\n  def g(self):\n    G()\n |    Node(0-39, split)
| class A:\n__________________________________________________ |       Node(0-5, split)
| class_______________________________________________________ |          Node(0-1, sibling)
| _____ ______________________________________________________ |          Node(1-2, sibling)
| ______A_____________________________________________________ |          Node(2-3, sibling)
| _______:____________________________________________________ |          Node(3-4, sibling)
| ________\n__________________________________________________ |          Node(4-5, sibling)
| __________  def f(self):\n    F()\n  def g(self):\n    G()\n |       Node(5-39, split)
| __________  def f(self):\n    F()\n  def g(self):\n    G()__ |          Node(5-38, paired)
| ____________________________________________________________ |             Node(5-6, left)
| __________  def f(self):\n    F()\n  def g(self):\n    G()__ |             Node(6-37, middle)
| __________  def f(self):\n    F()\n_________________________ |                Node(6-22, split)
| __________  def f(self):\n__________________________________ |                   Node(6-15, split)
| __________  ________________________________________________ |                      Node(6-7, sibling)
| ____________def_____________________________________________ |                      Node(7-8, sibling)
| _______________ ____________________________________________ |                      Node(8-9, sibling)
| ________________f___________________________________________ |                      Node(9-10, sibling)
| _________________(self)_____________________________________ |                      Node(10-13, paired)
| _________________(__________________________________________ |                         Node(10-11, left)
| __________________self______________________________________ |                         Node(11-12, middle)
| __________________self______________________________________ |                            Node(11-12, sibling)
| ______________________)_____________________________________ |                         Node(12-13, right)
| _______________________:____________________________________ |                      Node(13-14, sibling)
| ________________________\n__________________________________ |                      Node(14-15, sibling)
| __________________________    F()\n_________________________ |                   Node(15-22, split)
| __________________________    F()___________________________ |                      Node(15-21, paired)
| ____________________________________________________________ |                         Node(15-16, left)
| __________________________    F()___________________________ |                         Node(16-20, middle)
| __________________________    ______________________________ |                            Node(16-17, sibling)
| ______________________________F_____________________________ |                            Node(17-18, sibling)
| _______________________________()___________________________ |                            Node(18-20, paired)
| _______________________________(____________________________ |                               Node(18-19, left)
| ____________________________________________________________ |                               Node(19-19, middle)
| ________________________________)___________________________ |                               Node(19-20, right)
| ____________________________________________________________ |                         Node(20-21, right)
| _________________________________\n_________________________ |                      Node(21-22, sibling)
| ___________________________________  def g(self):\n    G()__ |                Node(22-37, split)
| ___________________________________  def g(self):\n_________ |                   Node(22-31, split)
| ___________________________________  _______________________ |                      Node(22-23, sibling)
| _____________________________________def____________________ |                      Node(23-24, sibling)
| ________________________________________ ___________________ |                      Node(24-25, sibling)
| _________________________________________g__________________ |                      Node(25-26, sibling)
| __________________________________________(self)____________ |                      Node(26-29, paired)
| __________________________________________(_________________ |                         Node(26-27, left)
| ___________________________________________self_____________ |                         Node(27-28, middle)
| ___________________________________________self_____________ |                            Node(27-28, sibling)
| _______________________________________________)____________ |                         Node(28-29, right)
| ________________________________________________:___________ |                      Node(29-30, sibling)
| _________________________________________________\n_________ |                      Node(30-31, sibling)
| ___________________________________________________    G()__ |                   Node(31-37, split)
| ___________________________________________________    G()__ |                      Node(31-37, paired)
| ____________________________________________________________ |                         Node(31-32, left)
| ___________________________________________________    G()__ |                         Node(32-36, middle)
| ___________________________________________________    _____ |                            Node(32-33, sibling)
| _______________________________________________________G____ |                            Node(33-34, sibling)
| ________________________________________________________()__ |                            Node(34-36, paired)
| ________________________________________________________(___ |                               Node(34-35, left)
| ____________________________________________________________ |                               Node(35-35, middle)
| _________________________________________________________)__ |                               Node(35-36, right)
| ____________________________________________________________ |                         Node(36-37, right)
| ____________________________________________________________ |             Node(37-38, right)
| __________________________________________________________\n |          Node(38-39, sibling)'''[1:],  # pylint: disable=line-too-long
      ),
  )
  def test_pseudo_parser(self, code: List[str], language: str,
                         expected: Optional[str] = None):

    pseudo_parsed, tokens_maybe_corrected, _, tokens_maybe_corrected_raw, _ = (
        stack_parser.PseudoParser.parse_and_maybe_preprocess(
            language, code, spaces_per_indent=2))
    actual = pseudo_parsed.human_readable_string(tokens_maybe_corrected)
    reconstructed_code = ''.join(tokens_maybe_corrected_raw)
    try:
      self.assert_equal_and_print(
          actual, expected, ignore_trailing_whitespace=True)
      self.assert_equal_and_print(
          code, reconstructed_code, ignore_trailing_whitespace=True)
    except:
      print(tokens_maybe_corrected_raw)
      raise

  @parameterized.named_parameters(
      dict(
          testcase_name='first',
          code='''
class C:
  def f(self, x,
y):
    if x>y:
      x = x + y[0:

1]
    return x+y
'''[1:],
          expected='''
class C:
<python_indent>  def f(self, x,
y):
<python_indent>    if x>y:
<python_indent>      x = x + y[0:

1]<python_dedent>
    return x+y<python_dedent><python_dedent>
'''[1:],
      ),
      dict(
          testcase_name='second',
          code='''
if a:
  x=1
  y=2
elif b:
  x=2
  y=1
'''[1:],
          expected='''
if a:
<python_indent>  x=1
  y=2<python_dedent>
elif b:
<python_indent>  x=2
  y=1<python_dedent>
'''[1:],
      )
  )
  def test_python_indent_dedent_desugar(self, code: str, expected: str) -> None:

    spaces_per_indent = 2
    pseudo_parser = stack_parser.PseudoParser.preset('python_preprocessor')
    (pseudo_parsed,
     tokens_maybe_corrected,
     _,
     tokens_maybe_corrected_raw,
     _) = pseudo_parser.pseudo_parse(code)
    newline_token = pseudo_parser.codec.encode('\n')[0][0]
    indent_token = pseudo_parser.codec.add_token(
        '<python_indent>', stack_parser.TokenType.OTHER)
    dedent_token = pseudo_parser.codec.add_token(
        '<python_dedent>', stack_parser.TokenType.OTHER)
    encoded_tokens_maybe_corrected, _ = pseudo_parser.codec.encode(
        ''.join(tokens_maybe_corrected))
    reconstructed_code = ''.join(tokens_maybe_corrected_raw)
    self.assert_equal_and_print(
        code, reconstructed_code, ignore_trailing_whitespace=True)
    self.assert_equal_and_print(
        len(encoded_tokens_maybe_corrected), len(tokens_maybe_corrected),
        ignore_trailing_whitespace=False)
    encoded_desugared_tokens_maybe_corrected = (
        stack_parser.python_indent_dedent_desugar(
            pseudo_parsed,
            tokens_maybe_corrected,
            encoded_tokens_maybe_corrected,
            indent_token,
            dedent_token,
            newline_token,
            spaces_per_indent))
    actual = ''.join(pseudo_parser.codec.decode_token(
        encoded_desugared_tokens_maybe_corrected))

    self.assert_equal_and_print(
        actual, expected, ignore_trailing_whitespace=True)

if __name__ == '__main__':
  absltest.main()
