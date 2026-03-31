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
from torch_tpu._internal.utils import device_utils
from torch_tpu._internal.utils import modeling_hf
from examples.deepseek import model as deepseek_model
from transformers import activations
from transformers.models.bert import modeling_bert
from transformers.models.qwen3 import configuration_qwen3
from transformers.models.qwen3 import modeling_qwen3


def _get_base_bert_config():
  cfg = modeling_bert.BertConfig(
      vocab_size=30522,
      hidden_size=768,
      num_hidden_layers=12,
      num_attention_heads=12,
      intermediate_size=3072,
      hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1,
      max_position_embeddings=512,
  )
  cfg._attn_implementation = "eager"  # pylint: disable=protected-access
  return cfg


def _get_base_qwen3_config():
  # https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
  config = configuration_qwen3.Qwen3Config.from_dict({
      "architectures": ["Qwen3ForCausalLM"],
      "attention_bias": False,
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 4096,
      "initializer_range": 0.02,
      "intermediate_size": 12288,
      "max_position_embeddings": 40960,
      "max_window_layers": 36,
      "model_type": "qwen3",
      "num_attention_heads": 32,
      "num_hidden_layers": 36,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06,
      "rope_scaling": None,
      "rope_theta": 1000000,
      "sliding_window": None,
      "tie_word_embeddings": False,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.51.0",
      "use_cache": True,
      "use_sliding_window": False,
      "vocab_size": 151936,
  })
  config._attn_implementation = "eager"  # pylint: disable=protected-access
  return config


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

  See `get_model` in //torch_tpu/_internal/utils/modeling_hf.py for
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
  elif model_name == "nn.Embedding":
    num_embeddings = kwargs["num_embeddings"]
    embedding_dim = kwargs["embedding_dim"]

    class EmbeddingModel(torch.nn.Module):

      def __init__(self, num_embeddings, embedding_dim, dtype):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim, dtype=dtype
        )

      def forward(self, x):
        return self.embedding(x)

    model = EmbeddingModel(num_embeddings, embedding_dim, dtype=weights_dtype)
    example_inputs = torch.randint(
        0,
        num_embeddings,
        (batch_size, sequence_length),
        dtype=torch.int32,  # Using int32 as per the reference
        device=device,
    )
  elif model_name == "nn.Dropout":
    p = kwargs["p"]
    input_shape = kwargs["shape"]

    class DropoutModel(torch.nn.Module):

      def __init__(self, p):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=p)

      def forward(self, x):
        return self.dropout(x)

    model = DropoutModel(p)
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.Tanh":
    input_shape = kwargs["shape"]

    class TanhModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()

      def forward(self, x):
        return self.tanh(x)

    model = TanhModel()
    example_inputs = torch.randn(
        input_shape,
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

  elif model_name == "BertLayer":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertLayer(cfg)
    example_inputs = (
        torch.randn(
            batch_size,
            sequence_length,
            cfg.hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
        None,
    )

    class BertLayerWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, attention_mask):
        return self.m(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            output_attentions=False,
            cache_position=None,
        )[0]

    model = BertLayerWrapper(model).to(dtype=weights_dtype)

  elif model_name == "BertSelfOutput":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertSelfOutput(cfg)
    example_inputs = (
        torch.randn(
            batch_size,
            sequence_length,
            cfg.hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
        torch.randn(
            batch_size,
            sequence_length,
            cfg.hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
    )

    class BertSelfOutputWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, input_tensor):
        return self.m(hidden_states=hidden_states, input_tensor=input_tensor)

    model = BertSelfOutputWrapper(model).to(dtype=weights_dtype)

  elif model_name == "BertIntermediate":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertIntermediate(cfg)
    example_inputs = torch.randn(
        batch_size,
        sequence_length,
        cfg.hidden_size,
        dtype=weights_dtype,
        device=device,
    )

    class BertIntermediateWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states):
        return self.m(hidden_states)

    model = BertIntermediateWrapper(model).to(dtype=weights_dtype)

  elif model_name == "BertOutput":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertOutput(cfg)
    example_inputs = (
        torch.randn(
            batch_size,
            sequence_length,
            cfg.intermediate_size,
            dtype=weights_dtype,
            device=device,
        ),
        torch.randn(
            batch_size,
            sequence_length,
            cfg.hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
    )

    class BertOutputWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, input_tensor):
        return self.m(hidden_states=hidden_states, input_tensor=input_tensor)

    model = BertOutputWrapper(model).to(dtype=weights_dtype)

  elif model_name == "GELUActivation":
    cfg = _get_base_bert_config()
    model = activations.GELUActivation(use_gelu_python=False)
    example_inputs = torch.randn(
        batch_size,
        sequence_length,
        cfg.intermediate_size,
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "BertPooler":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertPooler(cfg).to(dtype=weights_dtype)
    example_inputs = torch.randn(
        batch_size,
        sequence_length,
        cfg.hidden_size,
        dtype=weights_dtype,
        device=device,
    )

  elif model_name == "BertEmbeddings":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertEmbeddings(cfg).to(dtype=weights_dtype)
    input_ids = torch.randint(
        0,
        cfg.vocab_size,
        (batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    example_inputs = input_ids
  elif model_name == "Qwen3Attention":
    config = _get_base_qwen3_config()
    model = modeling_qwen3.Qwen3Attention(config, layer_idx=0)

    head_dim = config.hidden_size // config.num_attention_heads

    cos = torch.randn(
        1, sequence_length, head_dim, device=device, dtype=weights_dtype
    )
    sin = torch.randn(
        1, sequence_length, head_dim, device=device, dtype=weights_dtype
    )

    example_inputs = (
        torch.randn(
            batch_size,
            sequence_length,
            config.hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
        (cos, sin),
        None,
    )

    class Qwen3AttentionWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, position_embeddings, attention_mask):
        return self.m(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )[0]

    model = Qwen3AttentionWrapper(model)
    model = model.to(dtype=weights_dtype)

  elif model_name == "Qwen3RMSNorm":
    hidden_size = kwargs["hidden_size"]
    model = modeling_qwen3.Qwen3RMSNorm(hidden_size=hidden_size)
    model = model.to(dtype=weights_dtype)
    example_inputs = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        dtype=weights_dtype,
        device=device,
    )

  elif model_name == "Qwen3MLP":
    config = configuration_qwen3.Qwen3Config(
        hidden_size=kwargs["hidden_size"],
        intermediate_size=kwargs["intermediate_size"],
        hidden_act="silu",
    )
    model = modeling_qwen3.Qwen3MLP(config)
    model = model.to(dtype=weights_dtype)
    example_inputs = torch.randn(
        batch_size,
        sequence_length,
        config.hidden_size,
        dtype=weights_dtype,
        device=device,
    )

  elif model_name == "SiLUActivation":
    model = activations.SiLUActivation()
    input_shape = kwargs["shape"]
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
    )

  elif model_name == "Qwen3RotaryEmbedding":
    config = configuration_qwen3.Qwen3Config(
        max_position_embeddings=kwargs["max_position_embeddings"],
        rope_theta=kwargs["rope_theta"],
    )
    model = modeling_qwen3.Qwen3RotaryEmbedding(config)
    head_dim = kwargs["head_dim"]

    position_ids = (
        torch.arange(0, sequence_length, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    example_inputs = (
        torch.randn(
            batch_size,
            sequence_length,
            head_dim,
            dtype=weights_dtype,
            device=device,
        ),
        position_ids,
    )

    class Qwen3RotaryEmbeddingWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, x, position_ids):
        return self.m(x, position_ids)  # returns cos, sin

    model = Qwen3RotaryEmbeddingWrapper(model)

  elif model_name.startswith("DeepSeek"):
    dim = kwargs["dim"]
    args = deepseek_model.ModelArgs(
        max_batch_size=batch_size,
        max_seq_len=sequence_length,
        vocab_size=kwargs["vocab_size"],
        dim=dim,
        inter_dim=kwargs["inter_dim"],
        moe_inter_dim=kwargs["moe_inter_dim"],
        n_layers=kwargs["n_layers"],
        n_dense_layers=kwargs["n_dense_layers"],
        n_heads=kwargs["n_heads"],
        n_routed_experts=kwargs["n_routed_experts"],
        n_shared_experts=kwargs["n_shared_experts"],
        n_activated_experts=kwargs["n_activated_experts"],
    )

    if model_name == "DeepSeekParallelEmbedding":
      model = deepseek_model.ParallelEmbedding(args.vocab_size, args.dim).to(
          dtype=weights_dtype
      )
      example_inputs = torch.randint(
          0, args.vocab_size, (batch_size, sequence_length), device=device
      )

    elif model_name == "DeepSeekRMSNorm":
      model = deepseek_model.RMSNorm(args.dim).to(dtype=weights_dtype)
      example_inputs = torch.randn(
          batch_size,
          sequence_length,
          args.dim,
          dtype=weights_dtype,
          device=device,
      )

    elif model_name == "DeepSeekExpert":
      model = deepseek_model.Expert(args.dim, args.moe_inter_dim).to(
          dtype=weights_dtype
      )
      example_inputs = torch.randn(
          batch_size,
          sequence_length,
          args.dim,
          dtype=weights_dtype,
          device=device,
      )

    else:
      raise ValueError(f"Unknown DeepSeek layer: {model_name}")

    if model_name not in ["DeepSeekParallelEmbedding"]:
      model = model.to(dtype=weights_dtype)

  elif model_name == "nn.f.scaled_dot_product_attention":
    embed_dim = kwargs["embed_dim"]
    q_num_heads = kwargs["q_num_heads"]
    kv_num_heads = kwargs["kv_num_heads"]
    qk_head_dim = kwargs["qk_head_dim"]
    v_head_dim = kwargs["v_head_dim"]
    is_causal = kwargs["is_causal"]
    enable_gqa = kwargs["enable_gqa"]
    use_math_backend = kwargs.get("use_math_backend", False)

    class AttentionLayer(torch.nn.Module):

      def __init__(self, dtype):
        super().__init__()
        self.q_proj = torch.nn.Linear(
            embed_dim,
            q_num_heads * qk_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.k_proj = torch.nn.Linear(
            embed_dim,
            kv_num_heads * qk_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.v_proj = torch.nn.Linear(
            embed_dim,
            kv_num_heads * v_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.out_proj = torch.nn.Linear(
            q_num_heads * v_head_dim,
            embed_dim,
            bias=False,
            dtype=dtype,
        )

      def forward(self, x):
        bsz, q_len, _ = x.size()
        q = (
            self.q_proj(x)
            .view(bsz, q_len, q_num_heads, qk_head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(bsz, q_len, kv_num_heads, qk_head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(bsz, q_len, kv_num_heads, v_head_dim)
            .transpose(1, 2)
        )

        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.MATH]
            if use_math_backend
            else [torch.nn.attention.SDPBackend.OVERRIDEABLE]
        ):
          attn_output = torch.nn.functional.scaled_dot_product_attention(
              q, k, v, is_causal=is_causal, enable_gqa=enable_gqa
          )
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, q_num_heads * v_head_dim)
        )
        return self.out_proj(attn_output)

    model = AttentionLayer(dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, kwargs["q_seq_len"], embed_dim),
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
