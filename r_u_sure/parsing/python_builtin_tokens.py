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

"""List of builtin tokens for Python."""

# Extracted using keyword.kwlist, keyword.softkwlist, builtins.keys(), token
PYTHON_PUNCTUATION_TOKENS = frozenset({
    '(',
    ')',
    '[',
    ']',
    ':',
    ',',
    ';',
    '+',
    '-',
    '*',
    '/',
    '|',
    '&',
    '<',
    '>',
    '=',
    '.',
    '%',
    '{',
    '}',
    '==',
    '!=',
    '<=',
    '>=',
    '~',
    '^',
    '<<',
    '>>',
    '**',
    '+=',
    '-=',
    '*=',
    '/=',
    '%=',
    '&=',
    '|=',
    '^=',
    '<<=',
    '>>=',
    '**=',
    '//',
    '//=',
    '@',
    '@=',
    '->',
    '...',
    ':=',
})
PYTHON_BUILTIN_TOKENS = frozenset(
    {
        'False',
        'None',
        'True',
        '__peg_parser__',
        'and',
        'as',
        'assert',
        'async',
        'await',
        'break',
        'class',
        'continue',
        'def',
        'del',
        'elif',
        'else',
        'except',
        'finally',
        'for',
        'from',
        'global',
        'if',
        'import',
        'in',
        'is',
        'lambda',
        'nonlocal',
        'not',
        'or',
        'pass',
        'raise',
        'return',
        'try',
        'while',
        'with',
        'yield',
        '__name__',
        '__doc__',
        '__package__',
        '__loader__',
        '__spec__',
        '__build_class__',
        '__import__',
        'abs',
        'all',
        'any',
        'ascii',
        'bin',
        'breakpoint',
        'callable',
        'chr',
        'compile',
        'delattr',
        'dir',
        'divmod',
        'eval',
        'exec',
        'format',
        'getattr',
        'globals',
        'hasattr',
        'hash',
        'hex',
        'id',
        'input',
        'isinstance',
        'issubclass',
        'iter',
        'len',
        'locals',
        'max',
        'min',
        'next',
        'oct',
        'ord',
        'pow',
        'print',
        'repr',
        'round',
        'setattr',
        'sorted',
        'sum',
        'vars',
        'Ellipsis',
        'NotImplemented',
        'bool',
        'memoryview',
        'bytearray',
        'bytes',
        'classmethod',
        'complex',
        'dict',
        'enumerate',
        'filter',
        'float',
        'frozenset',
        'property',
        'int',
        'list',
        'map',
        'object',
        'range',
        'reversed',
        'set',
        'slice',
        'staticmethod',
        'str',
        'super',
        'tuple',
        'type',
        'zip',
        '__debug__',
        'BaseException',
        'Exception',
        'TypeError',
        'StopAsyncIteration',
        'StopIteration',
        'GeneratorExit',
        'SystemExit',
        'KeyboardInterrupt',
        'ImportError',
        'ModuleNotFoundError',
        'OSError',
        'EnvironmentError',
        'IOError',
        'EOFError',
        'RuntimeError',
        'RecursionError',
        'NotImplementedError',
        'NameError',
        'UnboundLocalError',
        'AttributeError',
        'SyntaxError',
        'IndentationError',
        'TabError',
        'LookupError',
        'IndexError',
        'KeyError',
        'ValueError',
        'UnicodeError',
        'UnicodeEncodeError',
        'UnicodeDecodeError',
        'UnicodeTranslateError',
        'AssertionError',
        'ArithmeticError',
        'FloatingPointError',
        'OverflowError',
        'ZeroDivisionError',
        'SystemError',
        'ReferenceError',
        'MemoryError',
        'BufferError',
        'Warning',
        'UserWarning',
        'DeprecationWarning',
        'PendingDeprecationWarning',
        'SyntaxWarning',
        'RuntimeWarning',
        'FutureWarning',
        'ImportWarning',
        'UnicodeWarning',
        'BytesWarning',
        'ResourceWarning',
        'ConnectionError',
        'BlockingIOError',
        'BrokenPipeError',
        'ChildProcessError',
        'ConnectionAbortedError',
        'ConnectionRefusedError',
        'ConnectionResetError',
        'FileExistsError',
        'FileNotFoundError',
        'IsADirectoryError',
        'NotADirectoryError',
        'InterruptedError',
        'PermissionError',
        'ProcessLookupError',
        'TimeoutError',
    }
    | PYTHON_PUNCTUATION_TOKENS
)
