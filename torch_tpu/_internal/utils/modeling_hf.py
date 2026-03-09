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

"""Model definitions for HF models."""

import dataclasses

import torch
import transformers
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.gpt_oss import modeling_gpt_oss
from transformers.models.llama import modeling_llama
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_moe import modeling_qwen3_moe

from rules_python.python.runfiles import runfiles

BASE_MODEL_CONFIG_PATH = "__main__/examples/huggingface_transformers/model_configs"


@dataclasses.dataclass
class HuggingfaceModelAndConfig:
  """A Hugging Face model and its configuration.

  Attributes:
    model: A `torch.nn.Module` representing the Hugging Face model.
    config: A `transformers.PretrainedConfig` object containing the model's
      configuration.
  """

  model: torch.nn.Module
  config: transformers.PretrainedConfig


def get_llama3_model(default_config, torch_dtype: torch.dtype):
  """Returns the Llama3.2 1B model."""
  # pylint: disable=protected-access
  model = modeling_llama.LlamaForCausalLM(default_config).to(torch_dtype)
  return model


def get_qwen3_model(default_config, torch_dtype: torch.dtype):
  """Returns the Qwen3 0.6B model."""
  # pylint: disable=protected-access
  model = modeling_qwen3.Qwen3ForCausalLM(default_config).to(torch_dtype)
  return model


def get_qwen3_moe_model(default_config, torch_dtype: torch.dtype):
  """Returns the Qwen3-Coder-30B-A3B-Instruct model."""
  # pylint: disable=protected-access
  # TODO: b/491105062 - Remove this once performance issue on TPU is resolved.
  # Currently, Run goes OOM and crashes after 31 minutes.
  # By default, the model has 48 hidden layers.
  default_config.num_hidden_layers = 2
  model = modeling_qwen3_moe.Qwen3MoeForCausalLM(default_config).to(torch_dtype)
  return model


def get_qwen3_coder_480b_a35b_instruct_fp8_dynamic_model(
    default_config, torch_dtype: torch.dtype
):
  """Returns the Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic model."""
  # pylint: disable=protected-access
  # By default, the model has 62 hidden layers.
  default_config.num_hidden_layers = 2
  model = modeling_qwen3_moe.Qwen3MoeForCausalLM(default_config).to(torch_dtype)
  return model


def get_gpt_oss_20b_model(default_config, torch_dtype: torch.dtype):
  """Returns the GPT OSS 20B model."""
  # pylint: disable=protected-access
  # By default, the model has 24 hidden layers.
  default_config.num_hidden_layers = 2
  # By default, the model has 24 layers, of which 12 are sliding attention and
  # 12 are full attention.
  default_config.layer_types = ["sliding_attention", "full_attention"]
  model = modeling_gpt_oss.GptOssForCausalLM(default_config).to(torch_dtype)
  return model


def get_gpt_oss_120b_model(default_config, torch_dtype: torch.dtype):
  """Returns the GPT OSS 120B model."""
  # pylint: disable=protected-access
  # By default, the model has 36 hidden layers.
  default_config.num_hidden_layers = 2
  # By default, the model has 36 layers, of which 18 are sliding attention and
  # 18 are full attention.
  default_config.layer_types = ["sliding_attention", "full_attention"]
  model = modeling_gpt_oss.GptOssForCausalLM(default_config).to(torch_dtype)
  return model


def get_gemma3_270m_model(default_config, torch_dtype: torch.dtype):
  """Returns the Gemma 3 270M model."""
  return modeling_gemma3.Gemma3ForCausalLM(default_config).to(torch_dtype)


def get_model(
    model_name: str, dtype: torch.dtype, device: str = "cpu"
) -> HuggingfaceModelAndConfig:
  """Returns a Hugging Face model and its configuration wrapped in a HuggingfaceModelAndConfig.

  Args:
    model_name: The name of the model to load (e.g., "meta-llama/Llama-3.2-1B").
    dtype: The torch.dtype to cast the model to.
    device: the torch.device to load the model onto.

  Returns:
    A HuggingfaceModelAndConfig instance containing the Hugging Face model and
    its configuration.

  Raises:
    ValueError: If the model_name is not supported.
  """

  model_config_path = runfiles.Create().Rlocation(
      f"{BASE_MODEL_CONFIG_PATH}/{model_name}/config.json"
  )
  default_config = transformers.AutoConfig.from_pretrained(model_config_path)
  # default_config = model_configs.create_config_loader(model_name)()
  with torch.device(device):
    match model_name:
      # go/keep-sorted start case=no
      case "BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic":
        model = get_qwen3_coder_480b_a35b_instruct_fp8_dynamic_model(
            default_config, dtype
        )
      case "google/gemma-3-270m":
        model = get_gemma3_270m_model(default_config, dtype)
      case "meta-llama/Llama-3.2-1B":
        model = get_llama3_model(default_config, dtype)
      case "openai/gpt-oss-120b":
        model = get_gpt_oss_120b_model(default_config, dtype)
      case "openai/gpt-oss-20b":
        model = get_gpt_oss_20b_model(default_config, dtype)
      case "Qwen/Qwen3-0.6B":
        model = get_qwen3_model(default_config, dtype)
      case "Qwen/Qwen3-1.7B":
        model = get_qwen3_model(default_config, dtype)
      case "Qwen/Qwen3-Coder-30B-A3B-Instruct":
        model = get_qwen3_moe_model(default_config, dtype)
      case _:
        model = transformers.AutoModelForCausalLM.from_config(
            default_config, torch_dtype=dtype
        )
      # go/keep-sorted end
  return HuggingfaceModelAndConfig(model=model, config=default_config)
