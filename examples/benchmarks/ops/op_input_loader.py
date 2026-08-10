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

import ast
import logging
from typing import Any, Dict, List, Tuple
import torch


def deserialize_tensor(
    shape: List[int], dtype: torch.dtype, device: str = "tpu"
) -> torch.Tensor:
  """Creates a synthetic random tensor on the specified device supporting all dtypes.

  Utilizes PyTorch's native `torch.testing.make_tensor` utility for
  comprehensive
  dtype support (floats, ints, uints, bool, complex, float8, and quantized
  dtypes).
  """
  torch_device = torch.device("tpu" if device == "tpu" else device)

  # PyTorch quantized dtypes require affine quantized tensor constructors
  if dtype in (
      getattr(torch, "qint8", None),
      getattr(torch, "quint8", None),
      getattr(torch, "qint32", None),
      getattr(torch, "quint4x2", None),
      getattr(torch, "quint2x4", None),
  ):
    try:
      return torch._empty_affine_quantized(
          shape, scale=1.0, zero_point=0, dtype=dtype, device=torch_device
      )
    except Exception:
      float_tensor = torch.randn(shape, dtype=torch.float32)
      return torch.quantize_per_tensor(
          float_tensor, scale=0.1, zero_point=0, dtype=dtype
      ).to(torch_device)

  try:
    return torch.testing.make_tensor(shape, dtype=dtype, device=torch_device)
  except Exception:
    try:
      # If direct allocation on custom device is unsupported, construct on CPU and transfer.
      return torch.testing.make_tensor(shape, dtype=dtype, device="cpu").to(
          torch_device
      )
    except Exception as e:
      logging.warning(f"Fallback for dtype {dtype}: {e}")
      return torch.empty(shape, dtype=dtype, device=torch_device)


from examples.benchmarks.shape_utils import _DTYPE_SHORT_NAMES
from examples.benchmarks.shape_utils import format_shape_signature
from examples.benchmarks.shape_utils import format_tensor
from examples.benchmarks.shape_utils import format_tensor_spec
from examples.benchmarks.shape_utils import shorten_dtype_name


def deserialize_args(
    inputs_str: str, device: str = "tpu"
) -> Tuple[List[Any], Dict[str, Any]]:
  """Deserializes the inputs string into actual arguments and keyword arguments,

  creating tensors on the specified device.
  Uses AST parsing for safety.
  """
  try:
    tree = ast.parse(inputs_str, mode="eval")
    if not isinstance(tree, ast.Expression):
      raise ValueError("Expected an expression")

    root = tree.body
    if not isinstance(root, ast.Tuple) or len(root.elts) != 2:
      raise ValueError("Expected a tuple of (args, kwargs)")

    args_node = root.elts[0]
    kwargs_node = root.elts[1]

    def eval_node(node):
      if isinstance(node, ast.Tuple):
        return tuple(eval_node(x) for x in node.elts)
      elif isinstance(node, ast.List):
        return [eval_node(x) for x in node.elts]
      elif isinstance(node, ast.Dict):
        return {
            eval_node(k): eval_node(v) for k, v in zip(node.keys, node.values)
        }
      elif isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, str):
          if val.startswith("torch."):
            parts = val.split(".")
            if len(parts) == 2:
              attr_name = parts[1]
              attr = getattr(torch, attr_name, None)
              if isinstance(attr, (torch.dtype, torch.memory_format)):
                return attr
          elif "tpu" in val and device != "tpu":
            return val.replace("tpu", device)
        return val
      elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
          return -eval_node(node.operand)
        else:
          raise ValueError(f"Unsupported unary operator: {type(node.op)}")
      elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "T":
          if len(node.args) != 2:
            raise ValueError("T expects 2 arguments")
          shape_node = node.args[0]
          dtype_node = node.args[1]

          # Evaluate shape
          if not isinstance(shape_node, ast.List):
            raise ValueError("T shape must be a list")
          shape = []
          for x in shape_node.elts:
            val = eval_node(x)
            if not isinstance(val, int):
              raise ValueError("All shape elements must be integers")
            shape.append(val)

          # Evaluate dtype
          if not isinstance(dtype_node, ast.Attribute):
            raise ValueError("T dtype must be a torch.dtype")
          if (
              not isinstance(dtype_node.value, ast.Name)
              or dtype_node.value.id != "torch"
          ):
            raise ValueError("T dtype must be accessed via torch.")
          dtype_str = dtype_node.attr
          dtype = getattr(torch, dtype_str, None)
          if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unknown or invalid dtype: {dtype_str}")

          return deserialize_tensor(shape, dtype, device=device)
        else:
          raise ValueError(
              "Unsupported function call:"
              f" {node.func.id if isinstance(node.func, ast.Name) else type(node.func)}"
          )
      else:
        raise ValueError(f"Unsupported AST node: {type(node)}")

    args = eval_node(args_node)
    kwargs = eval_node(kwargs_node)

    if not isinstance(args, list):
      raise ValueError("Args must be a list")
    if not isinstance(kwargs, dict):
      raise ValueError("Kwargs must be a dict")

    return args, kwargs

  except Exception as e:
    raise ValueError(
        f"Failed to deserialize inputs string: {inputs_str}. Error: {e}"
    ) from e
