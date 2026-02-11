# Copyright 2025 Google LLC
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

"""Generate scaled dot product attention kernel."""

from collections.abc import Sequence
import functools
from absl import app
from absl import flags
from torch_tpu._internal.pallas import pallas_kernel_generate_utils as kernel_utils
from torch_tpu.ops.scaled_dot_product_attention.kernels import scaled_dot_product_attention_kernels as kernels

_FORWARD = flags.DEFINE_boolean(
    "forward",
    False,
    "Generate forward kernel.",
)

_BACKWARD = flags.DEFINE_boolean(
    "backward",
    False,
    "Generate backward kernel.",
)

_HEADER = flags.DEFINE_string(
    "header",
    None,
    "Header file to include in the generated kernel string.",
)

_DTYPE = flags.DEFINE_string(
    "dtype",
    None,
    "dtype of the inputs/outputs.",
)

_IMPLEMENTATION = flags.DEFINE_string(
    "implementation",
    None,
    "Implementation file to include in the generated kernel string.",
)


#######################################################################
def export_forward_kernel(header_path, implementation_path):
  """Export a dynamic forward kernel."""

  exported = kernels.export_sdpa_forward_kernel()
  if header_path is not None and implementation_path is not None:
    kernel_utils.generate_embedded_file(
        header_path,
        implementation_path,
        [(
            "scaled_dot_product_attention_forward_mlir",
            # TODO(elliotenglish): change this to use mlir bytecode
            exported.mlir_module().encode(),
        )],
    )


def export_backward_kernel(header_path, implementation_path):
  # TODO(elliotenglish): Implement this.
  exported = kernels.export_sdpa_backward_kernel()
  if header_path is not None and implementation_path is not None:
    kernel_utils.generate_embedded_file(
        header_path,
        implementation_path,
        [(
            "scaled_dot_product_attention_backward_mlir",
            # TODO(elliotenglish): change this to use mlir bytecode
            exported.mlir_module().encode(),
        )],
    )


#######################################################################
def main(argv: Sequence[str]) -> None:
  del argv
  assert not (_FORWARD.value and _BACKWARD.value)
  if _FORWARD.value:
    export_forward_kernel(_HEADER.value, _IMPLEMENTATION.value)
  if _BACKWARD.value:
    export_backward_kernel(_HEADER.value, _IMPLEMENTATION.value)


if __name__ == "__main__":
  app.run(main)
