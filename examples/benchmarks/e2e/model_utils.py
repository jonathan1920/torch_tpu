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

import dataclasses
import enum
from typing import Any
from fairscale.nn.model_parallel import initialize as fairscale_init
import llama_models.llama3.model as m
import torch
from torch.distributed import fsdp
import torch.distributed.tensor as dt
from torch.nn import parallel
from torch_tpu._internal.model_runner_hf import modeling_hf
from torch_tpu._internal.utils import device_utils


@dataclasses.dataclass
class ModelAndInput:
  """Dataclass to hold the benchmarking model and its example inputs.

  Attributes:
      model: The model to run the benchmark on.
      example_inputs: The example inputs to run the model with. The model should
        be able to accept these inputs.
  """

  model: torch.nn.Module
  example_inputs: Any


class DistStrat(enum.Enum):
  NONE = "none"
  DDP = "ddp"
  FSDP = "fsdp"


def _load_fsdp_model(
    model: torch.nn.Module,
    model_cpu: torch.nn.Module,
    device: torch.device,
) -> torch.nn.Module:
  """Loads an FSDP-sharded on-device model from a CPU model.

  This function assumes a typical HF Transformer structure.

  Args:
      model: The model to be initialized. Initially it should be on the meta
        device.
      model_cpu: A CPU copy of model with full weights.
      device: The target device for the sharded model (e.g., 'tpu').

  Returns:
      The FSDP-sharded model with weights loaded on the specified device.
  """
  for layer in model.model.layers:
    fsdp.fully_shard(layer)
  fsdp.fully_shard(model.model)
  fsdp.fully_shard(model)
  model.to_empty(device=device)

  empty_sharded_sd = model.state_dict()
  sharded_sd = {}
  for param_name, full_cpu_tensor in model_cpu.state_dict().items():
    sharded_empty_param = empty_sharded_sd.get(param_name)
    if sharded_empty_param is None:
      continue
    sharded_tensor = dt.distribute_tensor(
        full_cpu_tensor,
        device_mesh=sharded_empty_param.device_mesh,
        placements=sharded_empty_param.placements,
        src_data_rank=None,
    )
    sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)

  model.load_state_dict(sharded_sd, assign=True)
  return model


def get_huggingface_llm_model(
    model_name: str,
    *,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
    use_torch_compile: bool,
    sequence_length: int,
    batch_size: int,
    dist_strat: DistStrat,
) -> ModelAndInput:
  """Returns the huggingface LLM model.

  See `get_model` in //torch_tpu/_internal/model_runner_hf/modeling_hf.py for
  the list of supported models.

  Args:
      model_name: The name of the Hugging Face model to load.
      device: The device to load the model and inputs on (e.g., 'tpu', 'cuda').
      weights_dtype: The data type for the model weights (e.g., torch.float32,
        torch.bfloat16).
      is_training: Whether the model should be in training mode.
      use_torch_compile: Whether to wrap the model with `torch.compile`.
      sequence_length: The length of the input sequence for the example inputs.
      batch_size: The batch size for the example inputs.
      dist_strat: strategy for distributing model across devices.

  Returns:
      A ModelAndInput dataclass containing the loaded Hugging Face model
      and example inputs suitable for benchmarking.
  """

  model_cpu = modeling_hf.get_model(model_name, weights_dtype).model
  example_inputs = {
      "input_ids": torch.randint(
          0,
          model_cpu.config.vocab_size,
          (batch_size, sequence_length),
          device=device,
          dtype=torch.int64,
          requires_grad=False,
      )
  }
  if is_training:
    example_inputs["labels"] = torch.randint(
        0,
        model_cpu.config.vocab_size,
        (batch_size, sequence_length),
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )

  model: torch.nn.Module
  if dist_strat == DistStrat.NONE:
    model = model_cpu.to(device)
  elif dist_strat == DistStrat.DDP:
    model = model_cpu.to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = parallel.DistributedDataParallel(
        model,
        init_sync=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=False,
    )
  elif dist_strat == DistStrat.FSDP:
    model_meta = modeling_hf.get_model(
        model_name, weights_dtype, device="meta"
    ).model
    model_meta.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = _load_fsdp_model(
        model=model_meta, model_cpu=model_cpu, device=device
    )
  else:
    raise ValueError(f"Distributed strategy {dist_strat} unrecognized")

  if is_training:
    model.train()
  else:
    model.eval()

  if use_torch_compile:
    model = device_utils.torch_compile(model, device.type)
  return ModelAndInput(model=model, example_inputs=example_inputs)


def _init_model_weights(model):
  """Initializes model weights to small random values."""

  if hasattr(model, "weight") and model.weight is not None:
    tensor = model.weight.data
    tensor.normal_(std=0.01)

  if hasattr(model, "bias") and model.bias is not None:
    tensor = model.bias.data
    tensor.fill_(0.0)


def get_meta_llama_model(
    model_name: str,
    device: torch.device,
    weights_dtype: torch.dtype,
    use_torch_compile: bool,
    sequence_length: int,
    batch_size: int,
) -> ModelAndInput:
  """Returns a ModelAndInput for the specified Meta Llama model.

  This function loads or initializes a Meta Llama model with random weights
  and prepares example inputs suitable for benchmarking. Model parallel
  initialization is handled internally.

  Args:
      model_name: The name of the Meta Llama model to load (e.g.,
        "Llama-3.2-8B", "Llama-3.2-70B").
      device: The torch device to load the model and inputs.
      weights_dtype: The data type for the model weights.
      use_torch_compile: Whether to wrap the model with `torch.compile`.
      sequence_length: The length of the input sequence for the example inputs.
      batch_size: The batch size for the example inputs.

  Returns:
      A ModelAndInput dataclass containing the loaded Meta Llama model
      and example inputs. The example inputs are a tuple of (input_ids,
      start_pos).
  """

  world_size = (
      torch.distributed.get_world_size()
      if torch.distributed.is_initialized()
      else 1
  )
  # Ensure model parallel is initialized
  if not fairscale_init.model_parallel_is_initialized():
    fairscale_init.initialize_model_parallel(world_size)

  if model_name == "Llama-3.2-8B":
    args = m.ModelArgs(
        dim=4096,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        n_heads=32,
        n_kv_heads=8,
        n_layers=32,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        vocab_size=128256,
        max_seq_len=2048,
        max_batch_size=batch_size,
    )
  elif model_name == "Llama-3.2-70B":
    args = m.ModelArgs(
        dim=8192,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        n_heads=64,
        n_kv_heads=8,
        n_layers=80,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        vocab_size=128256,
        max_seq_len=2048,
        max_batch_size=batch_size,
    )
  else:
    raise ValueError(f"Unknown model name: {model_name}")
  prev_dtype = torch.get_default_dtype()
  torch.set_default_dtype(weights_dtype)

  with torch.device(device):
    # We currently only support random weights for benchmarking
    # TODO(b/461516258): investigate need to initialize model with
    # inference_mode()
    with torch.inference_mode():
      model = m.Transformer(args)

    model.apply(_init_model_weights)

  if use_torch_compile:
    model = device_utils.torch_compile(model, device.type)

  input_ids = torch.randint(
      0, args.vocab_size, (batch_size, sequence_length), device=device
  )
  # The model expects (tokens, start_pos)
  example_inputs = (input_ids, 0)

  torch.set_default_dtype(prev_dtype)
  return ModelAndInput(model=model, example_inputs=example_inputs)


def get_ml_layer_model(
    model_name: str,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
    use_torch_compile: bool,
    batch_size: int,
    sequence_length: int,
    **kwargs,
) -> ModelAndInput:
  """Returns a ModelAndInput for the specified ML layer.

  Args:
      model_name: The name of the layer (e.g., "nn.Linear").
      device: The torch device.
      weights_dtype: The data type for the weights.
      is_training: Whether the model should be in training mode.
      use_torch_compile: Whether to wrap the model with `torch.compile`.
      batch_size: The batch size for inputs.
      sequence_length: The length of the input sequence for the example inputs.
      **kwargs: Additional arguments for specific layers (e.g., in_features,
        out_features, etc).

  Returns:
      ModelAndInput dataclass.
  """
  if model_name == "nn.Linear":
    if "in_features" not in kwargs:
      raise ValueError("nn.Linear requires in_features to be specified.")
    if "out_features" not in kwargs:
      raise ValueError("nn.Linear requires out_features to be specified.")

    in_features = kwargs["in_features"]
    out_features = kwargs["out_features"]

    class LinearModel(torch.nn.Module):

      def __init__(self, in_features, out_features, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, dtype=dtype)

      def forward(self, x):
        return self.linear(x)

    model = LinearModel(in_features, out_features, dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, sequence_length, in_features),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.BatchNorm1d":
    num_features = kwargs["num_features"]

    class BatchNormModel(torch.nn.Module):

      def __init__(self, num_features, dtype):
        super().__init__()
        self.batchnorm = torch.nn.BatchNorm1d(num_features, dtype=dtype)

      def forward(self, x):
        return self.batchnorm(x)

    model = BatchNormModel(num_features, dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, num_features, sequence_length),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.LayerNorm":
    normalized_shape = kwargs["normalized_shape"]
    # shape argument for input generation
    input_shape = kwargs["shape"]

    class LayerNormModel(torch.nn.Module):

      def __init__(self, normalized_shape, dtype):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(normalized_shape, dtype=dtype)

      def forward(self, x):
        return self.layernorm(x)

    model = LayerNormModel(normalized_shape, dtype=weights_dtype)
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.Conv2d":
    in_channels = kwargs["in_channels"]
    out_channels = kwargs["out_channels"]
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    height = kwargs["height"]
    width = kwargs["width"]

    class Conv2dModel(torch.nn.Module):

      def __init__(
          self,
          in_channels,
          out_channels,
          kernel_size,
          stride,
          padding,
          dtype,
      ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dtype=dtype,
        )

      def forward(self, x):
        return self.conv(x)

    model = Conv2dModel(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dtype=weights_dtype,
    )
    example_inputs = torch.randn(
        (batch_size, in_channels, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.RMSNorm":
    num_features = kwargs["num_features"]

    class RMSNormModel(torch.nn.Module):

      def __init__(self, num_features, dtype):
        super().__init__()
        self.rmsnorm = torch.nn.RMSNorm(num_features, dtype=dtype)

      def forward(self, x):
        return self.rmsnorm(x)

    model = RMSNormModel(num_features, dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, sequence_length, num_features),
        dtype=weights_dtype,
        device=device,
    )
  else:
    raise ValueError(f"Unknown ML layer model: {model_name}")

  model = model.to(device)
  if is_training:
    model.train()
  else:
    model.eval()

  # Only compile the model for inference. For training, we will compile the
  # train step function which includes the forward and backward pass.
  if use_torch_compile and not is_training:
    model = device_utils.torch_compile(model, device.type)

  return ModelAndInput(model=model, example_inputs=example_inputs)
