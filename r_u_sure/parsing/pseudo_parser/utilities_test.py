# coding=utf-8
# Copyright 2026 The R-U-SURE Authors.
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

"""Tests for the utilities that use the stack parser."""

import textwrap
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
from r_u_sure.parsing.pseudo_parser import stack_parser
from r_u_sure.parsing.pseudo_parser import utilities


class UtilitiesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='first',
          python_code=textwrap.dedent('''
                                      def f(x):
                                        if x > 1:
                                          return x
                                      '''),
          spaces_per_indent=2,
      ),
      dict(
          testcase_name='second',
          python_code=textwrap.dedent('''
                                      def f(x):
                                         if x > 1:
                                            return x
                                      '''),
          spaces_per_indent=3,
      ),
      dict(
          testcase_name='third',
          python_code=textwrap.dedent('''
                                      x = """
                                       a
                                       a
                                       a
                                       """
                                      y = [
                                       a,
                                       a,
                                       a]
                                      def f(x):
                                        if x > 1:
                                          return x
                                      '''),
          spaces_per_indent=2,
      ),
      dict(
          testcase_name='fourth',
          python_code=textwrap.dedent('''
                                      x
                                      '''),
          spaces_per_indent=1,
      ),
      dict(
          testcase_name='fifth',
          python_code=textwrap.dedent(''),
          spaces_per_indent=1,
      ),)
  def test_infer_spaces_per_indent(
      self, python_code: str, spaces_per_indent: int):
    inferred_spaces_per_indent = utilities.infer_python_spaces_per_indent(
        python_code)
    self.assertEqual(inferred_spaces_per_indent, spaces_per_indent)

  @parameterized.named_parameters(
      dict(
          testcase_name='first',
          python_code=textwrap.dedent('''
                                      def f(x):
                                        """Docstring here..
                                        """
                                        return x + 1
                                      '''),
          docstring_indices=[16],
      ),
      dict(
          testcase_name='second',
          python_code=textwrap.dedent('''
                                      class A:
                                        def f(self):
                                          """Docstring here."""
                                          return x + 1
                                        def f(self):
                                          \'\'\'Docstring here.\'\'\'
                                          return x + 1
                                        def f(self):
                                          """Docstring here.
                                          Hello.
                                          """
                                          return x + 1
                                      """No docstring here."""
                                      '''),
          docstring_indices=[32, 90, 148],
      ),)
  def test_infer_python_docstring_indices(
      self, python_code: str, docstring_indices: int):
    inferred_docstring_indices = utilities.find_python_function_docstrings(
        python_code)
    self.assertEqual(inferred_docstring_indices, docstring_indices)
    for i in inferred_docstring_indices:
      assert python_code[i:].startswith('Docstring here.')

  @parameterized.named_parameters(
      dict(
          testcase_name='py_first',
          language='python',
          expected_truncation_position_prefix=None,
          code=textwrap.dedent('''
            a=1
            def f():
              cursor_position = 1
              return
            truncation_position = 1
            hello
            '''[1:]),
      ),
      dict(
          testcase_name='py_second',
          language='python',
          expected_truncation_position_prefix=None,
          code=textwrap.dedent('''
            def f():
              cursor_position = 1
              if True:
                xxx
              return
            truncation_position = 1
            hello
            '''[1:]),
      ),
      dict(
          testcase_name='py_third',
          language='python',
          expected_truncation_position_prefix=None,
          code=textwrap.dedent('''
            if True:
              xxx
            cursor_position = 1
            truncation_position = 1
            yyy
            '''[1:]),
      ),
      dict(
          testcase_name='py_fourth',
          language='python',
          expected_truncation_position_prefix='range(10):\n',
          code=textwrap.dedent('''
            pass
            pass
            for cursor_position in range(10):
              print(i)
              foo(i)
            '''[1:]),
      ),
      dict(
          testcase_name='cpp_first',
          language='cpp',
          expected_truncation_position_prefix=None,
          code=textwrap.dedent('''
            if (x) {
                cursor_position = 1;
                y = 2;
            };truncation_position=1;
            '''[1:]),
      ),
      dict(
          testcase_name='cpp_second',
          language='cpp',
          expected_truncation_position_prefix=None,
          code=textwrap.dedent('''
            if (cursor_position) {
                x = 1;truncation_position=1;
            };
            '''[1:]),
      ),
      dict(
          testcase_name='cpp_third',
          language='cpp',
          expected_truncation_position_prefix=None,
          code=textwrap.dedent('''
            if (x) {
                if (cursor_position) {
                  x = 1;
                };
            };truncation_position=1;
            '''[1:]),
      ),
      dict(
          testcase_name='cpp_fourth',
          language='cpp',
          expected_truncation_position_prefix=None,
          code=textwrap.dedent('''
            x=1;cursor_position=2;truncation_position=3;
            '''[1:]),
      ),)
  def test_infer_truncation_with_fallbacks(
      self, language: str,
      expected_truncation_position_prefix: Optional[str],
      code: str):
    cursor_position = code.find('cursor')
    if expected_truncation_position_prefix is not None:
      # expected_truncation_position is after the occurrence of
      # expected_truncation_position_prefix in code:
      assert expected_truncation_position_prefix in code
      expected_truncation_position = (
          code.find(expected_truncation_position_prefix) +
          len(expected_truncation_position_prefix)
          )
    else:
      # more convenient but sometimes too inflexible:
      # expected_truncation_position is before the occurrence of the special
      # sequence 'truncation' in code:
      assert 'truncation' in code
      expected_truncation_position = code.find('truncation')
    truncation_position = utilities.infer_truncation_with_fallbacks(
        code, cursor_position, language)
    self.assertEqual(truncation_position, expected_truncation_position)

  @parameterized.named_parameters(
      dict(
          testcase_name='a',
          code=textwrap.dedent("""
            def f(x)
              '''a docstring
              with two lines'''
              return x + 1
            """[1:]),
          expected_truncation=textwrap.dedent("""
            def f(x)
              '''a docstring
              with two lines'''"""[1:])
      ),
      dict(
          testcase_name='b',
          code=textwrap.dedent('''
            def f(x)
              """a docstring
              with two lines"""
              return x + 1
            '''[1:]),
          expected_truncation=textwrap.dedent('''
            def f(x)
              """a docstring
              with two lines"""'''[1:])
      ),
      dict(
          testcase_name='c',
          code=textwrap.dedent('''
            def f(x)
              """a docstring
              not terminated
            '''[1:]),
          expected_truncation=textwrap.dedent('''
            def f(x)
              """a docstring
              not terminated
            '''[1:]),
      ),)
  def test_infer_truncation_pydocstring(
      self, code: str,
      expected_truncation: str):
    cursor_position = utilities.find_python_function_docstrings(code)[0]
    truncation_index = utilities.infer_truncation_pydocstring(
        code, cursor_position)
    self.assertEqual(code[:truncation_index], expected_truncation)



class UnrestrictedSplitTruncationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('python', 'python', 'first = 1\nsecond = 2\n', 'first = 1\n'),
      (
          'python_brackets',
          'python',
          'first = [1,\n2]\nnext = 3\n',
          'first = [1,\n2]\n',
      ),
      ('cpp', 'cpp', 'first(); second();', 'first();'),
      ('java', 'java', 'first(); second();', 'first();'),
      ('javascript', 'javascript', 'first(); second();', 'first();'),
      (
          'cpp_comment',
          'cpp',
          'first /* note */ = 1; next();',
          'first /* note */',
      ),
  )
  def test_none_accepts_any_parser_split(self, language, code, prefix):
    root, _, _, tokens, _ = (
        stack_parser.PseudoParser.parse_and_maybe_preprocess(language, code)
    )
    index = utilities.infer_split_truncation_index(root, tokens, None, 0)
    self.assertEqual(code[:index], prefix)

  @parameterized.named_parameters(
      ('semicolon', (';',), 'first /* note */ = 1;'),
      ('empty_filter', (), 'first /* note */ = 1; next();'),
  )
  def test_explicit_filter_is_preserved(self, splitters, prefix):
    code = 'first /* note */ = 1; next();'
    root, _, _, tokens, _ = (
        stack_parser.PseudoParser.parse_and_maybe_preprocess('cpp', code)
    )
    index = utilities.infer_split_truncation_index(root, tokens, splitters, 0)
    self.assertEqual(code[:index], prefix)

  @parameterized.parameters('', 'first();')
  def test_end_of_input(self, code):
    root, _, _, tokens, _ = (
        stack_parser.PseudoParser.parse_and_maybe_preprocess('cpp', code)
    )
    index = utilities.infer_split_truncation_index(
        root, tokens, None, len(code)
    )
    self.assertEqual(index, len(code))


if __name__ == '__main__':
  absltest.main()
