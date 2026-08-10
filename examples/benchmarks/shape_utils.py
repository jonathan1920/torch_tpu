# Copyright 2026 Google LLC
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

"""Common utilities for formatting tensor shapes and dtypes in MLIR syntax.

This module provides common utilities for stringifying tensor shapes, dtypes,
and argument signatures into MLIR syntax (e.g. `<2048x2048xbf16>`, `<100xf32>`,
`<f32>`). These utilities are shared across op-level benchmarks and end-to-end
model benchmarks for consistent labeling of input shapes and tensor signatures.
"""

import ast
from typing import Any, Dict, Sequence, Union

import torch

_DTYPE_SHORT_NAMES: Dict[str, str] = {
    # Floating Point
    "float32": "f32",
    "float": "f32",
    "float64": "f64",
    "double": "f64",
    "float16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    # Signed Integers
    "int64": "i64",
    "long": "i64",
    "int32": "i32",
    "int": "i32",
    "int16": "i16",
    "short": "i16",
    "int8": "i8",
    # Unsigned Integers
    "uint8": "ui8",
    "byte": "ui8",
    "uint16": "ui16",
    "uint32": "ui32",
    "uint64": "ui64",
    # Boolean
    "bool": "i1",
    # Complex
    "complex64": "complex<f32>",
    "cfloat": "complex<f32>",
    "complex128": "complex<f64>",
    "cdouble": "complex<f64>",
    "complex32": "complex<f16>",
    "chalf": "complex<f16>",
    # Float8
    "float8_e4m3fn": "f8E4M3FN",
    "float8_e5m2": "f8E5M2",
    "float8_e4m3fnuz": "f8E4M3FNUZ",
    "float8_e5m2fnuz": "f8E5M2FNUZ",
    # Quantized
    "qint8": "qint8",
    "quint8": "quint8",
    "qint32": "qint32",
    "quint4x2": "quint4x2",
    "quint2x4": "quint2x4",
}


def shorten_dtype_name(dtype: Union[torch.dtype, str]) -> str:
  """Converts a PyTorch dtype or string representation into a concise MLIR abbreviation."""
  clean_name = str(dtype).replace("torch.", "")
  return _DTYPE_SHORT_NAMES.get(clean_name, clean_name)


def format_tensor_spec(
    shape: Sequence[Union[int, str]], dtype: Union[torch.dtype, str]
) -> str:
  """Formats a tensor shape and dtype into standard MLIR tensor syntax.

  Symbolic dimension names (e.g. 'b', 's', 'd') are formatted in uppercase
  (e.g. `<BxSxf32>`) to distinguish them from the lowercase 'x' dimension
  separator and dtype strings.

  Examples:
    format_tensor_spec([2048, 2048], torch.bfloat16) -> "<2048x2048xbf16>"
    format_tensor_spec(["b", "s"], torch.float32) -> "<BxSxf32>"
    format_tensor_spec([100], torch.float32) -> "<100xf32>"
    format_tensor_spec([], torch.float32) -> "<f32>"
  """
  short_dtype = shorten_dtype_name(dtype)
  if not shape:
    return f"<{short_dtype}>"

  def _format_dim(d: Union[int, str]) -> str:
    s = str(d)
    try:
      int(s)
      return s
    except ValueError:
      return s.upper()

  dims = "x".join(_format_dim(d) for d in shape)
  return f"<{dims}x{short_dtype}>"


def format_tensor(tensor: torch.Tensor) -> str:
  """Formats a live PyTorch tensor into standard MLIR tensor syntax.

  Example:
    format_tensor(torch.empty((2048, 2048), dtype=torch.bfloat16))
    -> "<2048x2048xbf16>"
  """
  return format_tensor_spec(tensor.shape, tensor.dtype)


def format_shape_signature(inputs_str: str) -> str:
  """Formats an AST inputs string into a concise MLIR-style argument signature.

  For example:
    ([T([4096, 2048], torch.bfloat16), T([2048, 2048], torch.bfloat16)], {})
    -> "<4096x2048xbf16>, <2048x2048xbf16>"

    ([T([2, 3], torch.float32), 1.0], {'dim': 0})
    -> "<2x3xf32>, 1.0, dim=0"
  """
  try:
    tree = ast.parse(inputs_str, mode="eval")
    if not isinstance(tree, ast.Expression):
      return inputs_str
    root = tree.body
    if not isinstance(root, ast.Tuple) or len(root.elts) != 2:
      return inputs_str

    args_node = root.elts[0]
    kwargs_node = root.elts[1]

    def format_node(node: ast.AST) -> str:
      if (
          isinstance(node, ast.Call)
          and isinstance(node.func, ast.Name)
          and node.func.id == "T"
      ):
        shape_elts = []
        if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
          for x in node.args[0].elts:
            shape_elts.append(format_node(x))
        dtype_str = "float32"
        if len(node.args) > 1:
          if isinstance(node.args[1], ast.Attribute):
            dtype_str = node.args[1].attr
          elif isinstance(node.args[1], ast.Name):
            dtype_str = node.args[1].id
        return format_tensor_spec(shape_elts, dtype_str)
      elif isinstance(node, ast.List):
        return "[" + ", ".join(format_node(x) for x in node.elts) + "]"
      elif isinstance(node, ast.Tuple):
        return "(" + ", ".join(format_node(x) for x in node.elts) + ")"
      elif isinstance(node, ast.Constant):
        return str(node.value)
      elif (
          isinstance(node, ast.UnaryOp)
          and isinstance(node.op, ast.USub)
          and isinstance(node.operand, ast.Constant)
      ):
        return f"-{node.operand.value}"
      elif isinstance(node, ast.Attribute):
        return node.attr
      elif isinstance(node, ast.Name):
        return node.id
      return ""

    parts = []
    if isinstance(args_node, (ast.List, ast.Tuple)):
      for x in args_node.elts:
        formatted = format_node(x)
        if formatted:
          parts.append(formatted)
    elif isinstance(args_node, ast.Call):
      formatted = format_node(args_node)
      if formatted:
        parts.append(formatted)

    if isinstance(kwargs_node, ast.Dict):
      for k, v in zip(kwargs_node.keys, kwargs_node.values):
        key_str = format_node(k) if k is not None else ""
        val_str = format_node(v)
        parts.append(f"{key_str}={val_str}")

    return ", ".join(parts) if parts else inputs_str
  except Exception:
    return inputs_str
