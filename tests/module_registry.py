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

"""Provides a unified registry for instantiating PyTorch models from various sources.

This module defines a generic interface for model providers (`BaseProvider`) and
implementations for popular model libraries:
    - Torchvision
    - TIMM (PyTorch Image Models)
    - Hugging Face Transformers

It exposes a `ModuleRegistry` that aggregates these providers, allowing users
to list available models and retrieve `ModuleSpec` objects containing the
instantiated model and factory functions for generating compatible sample
inputs.
"""

import abc
from collections.abc import Callable, Sequence
from importlib import resources
from importlib.resources import abc as resources_abc
import pathlib
from typing import Any, Iterator

from absl import flags
from absl import logging
from etils import epath
import timm
import torch
import torchvision
import transformers

_MAX_SEQ_LEN_HEURISTIC_CAP = 100_000

_HF_TRANSFORMERS_WEIGHTS_DIR = flags.DEFINE_string(
    "hf_transformers_weights_dir",
# "$DATA_ROOT/$WEIGHTS_SUBDIR/huggingface"
    "",
    "Location of weights and config files for HuggingFace Transformers models.",
)

_HF_TIMM_WEIGHTS_DIR = flags.DEFINE_string(
    "hf_timm_weights_dir",
# "$DATA_ROOT/$WEIGHTS_SUBDIR/huggingface/timm"
    "",
    "Location of weights and config files for HuggingFace TIMM models.",
)


class ModuleSpec:
  """A specification container for a PyTorch model and its inputs.

  Attributes:
    module_factory: A callable that returns an instantiated `torch.nn.Module`.
    sample_inputs_factory: A callable that generates compatible input tensors.
      It accepts optional `shape` (Sequence) and `device` (str) arguments and
      returns a tuple containing `(args, kwargs)` for the model's forward pass.
    preprocessor_factory: A callable that generates a preprocessor for models
      that need it. None by default.
    config: Optional configuration object associated with the model (e.g., a
      Transformers `AutoConfig`).
  """

  def __init__(
      self,
      module_factory: Callable[[], torch.nn.Module],
      sample_inputs_factory: Callable[
          [Sequence[int] | None, str | None],
          tuple[tuple[Any, ...], dict[str, Any]],
      ],
      preprocessor_factory: Callable[[], Any] | None = None,
      config: Any | None = None,
  ):
    self.module_factory = module_factory
    self.sample_inputs_factory = sample_inputs_factory
    self.preprocessor_factory = preprocessor_factory
    self.config = config


class BaseProvider(abc.ABC):
  """Abstract base class for model source providers."""

  @abc.abstractmethod
  def list_modules(self) -> list[str]:
    """Lists the names of all models available from this provider.

    Returns:
      A list of model name strings.
    """
    pass

  @abc.abstractmethod
  def get_module_spec(
      self, name: str, *, load_weights: bool = False
  ) -> ModuleSpec:
    """Retrieves the specification for a specific model.

    Args:
      name: The name of the model to retrieve.
      load_weights: If True, loads pre-trained weights. If False, initializes
        with random weights.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    pass


class TorchvisionProvider(BaseProvider):
  """Provider for standard Torchvision models."""

  def list_modules(self) -> list[str]:
    return torchvision.models.list_models()

  def get_module_spec(
      self, name: str, *, load_weights: bool = False
  ) -> ModuleSpec:
    if load_weights:
      raise NotImplementedError(
          "Loading pretrained weights not yet implemented."
      )

    default_shape = (1, 3, 224, 224)
    return ModuleSpec(
        lambda: torchvision.models.get_model(name),
        lambda shape=default_shape, device="cpu": (
            (torch.randn(shape, device=device),),
            {},
        ),
    )


class TimmProvider(BaseProvider):
  """Provider for TIMM (PyTorch Image Models)."""

  def __init__(self):
    weights_dir = _HF_TIMM_WEIGHTS_DIR.value
    self._weights_dir = epath.Path(weights_dir) if weights_dir else None

  def list_modules(self) -> list[str]:
    return timm.list_models()

  def get_module_spec(
      self, name: str, *, load_weights: bool = False
  ) -> ModuleSpec:
    """Creates a ModuleSpec for a TIMM model.

    Attempts to determine the default input shape from the model's pretrained
    configuration. Defaults to (3, 224, 224) if configuration is unavailable.

    Args:
      name: Name (str) of the timm model.
      load_weights: If True, loads pretrained weights. If False, initializes
        with random weights.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    try:
      config = timm.models.get_pretrained_cfg(name)
    except RuntimeError:
      config = None
      logging.warning(
          "Couldn't find config for %s.",
          name,
      )

    def _module_factory():
      if load_weights and self._weights_dir:
        local_checkpoint = self._weights_dir / f"{name}.pth"

        if local_checkpoint.exists():
          model = timm.create_model(name, pretrained=False)
          #  local checkpoint can't be loaded by timm.create_model directly.
          with local_checkpoint.open("rb") as f:
            state_dict = torch.load(f)
          model.load_state_dict(state_dict)
          return model
        else:
          logging.warning(
              "Checkpoint %s not found. Using random init.", local_checkpoint
          )

      return timm.create_model(name, pretrained=False)

    def _input_factory(shape=None, device="cpu"):
      input_size = config.input_size if config else (3, 224, 224)
      final_shape = shape if shape else (1, *input_size)
      return ((torch.randn(final_shape, device=device),), {})

    def _preprocessor_factory():
      if config:
        data_config = timm.data.resolve_data_config(
            {}, pretrained_cfg=config.to_dict()
        )
        transform = timm.data.create_transform(**data_config)
        return lambda img: transform(img).unsqueeze(0)
      return None

    return ModuleSpec(
        _module_factory, _input_factory, _preprocessor_factory, config
    )


def _get_max_seq_len(
    config: transformers.AutoConfig, default: int = 512
) -> int:
  """Heuristically determines the maximum sequence length from a config.

  Checks multiple attributes (e.g., `max_position_embeddings`, `n_positions`)
  and applies a cap to avoid unreasonably large allocations (e.g., for models
  using relative embeddings or integer-limit defaults).

  Args:
    config: The Transformers configuration object.
    default: The fallback sequence length if no attribute is found.

  Returns:
    The determined maximum sequence length.
  """
  for attr in [
      "max_position_embeddings",
      "n_positions",
      "seq_length",
      "max_seq_len",
  ]:
    if hasattr(config, attr):
      value = getattr(config, attr)
      # Some configs have this set to huge numbers (e.g. integer limit)
      # or None. We cap it to an arbitrary value.
      if value is not None and value < _MAX_SEQ_LEN_HEURISTIC_CAP:
        return value

    # Check tokenizer-specific max length if available in config
    # (Sometimes 'model_max_length' is injected into config).
    if hasattr(config, "model_max_length"):
      value = config.model_max_length
      if value is not None and value < _MAX_SEQ_LEN_HEURISTIC_CAP:
        return value

  # Fallback for relative position models (T5, etc.) or missing data.
  return default


def _walk_package_resources(
    traversable: resources_abc.Traversable,
) -> Iterator[resources_abc.Traversable]:
  """Recursively yields Traversable objects for all files in a directory/package."""
  for path in traversable.iterdir():
    if path.is_file():
      yield path
    elif path.is_dir():
      # Recursively call the function for subdirectories
      yield from _walk_package_resources(path)


class TransformersProvider(BaseProvider):
  """Provider for Hugging Face Transformers models.

  Note:
    This provider relies on local resource files for model configurations
    rather than downloading directly from the Hugging Face Hub.
  """

  _FILES = resources.files("torch_tpu").joinpath(
      "examples/huggingface_transformers/model_configs"
  )

  def __init__(self):
    weights_dir = _HF_TRANSFORMERS_WEIGHTS_DIR.value
    if weights_dir:
      self._weights_dir = epath.Path(weights_dir)
    else:
      self._weights_dir = None

  def list_modules(self) -> list[str]:
    modules = []
    for resource in _walk_package_resources(self._FILES):
      if resource.is_file() and "config.json" in resource.name:
        with resources.as_file(resource) as f:
          modules.append(f"{f.parts[-3]}/{f.parts[-2]}")
    return modules

  def get_module_spec(
      self, name: str, *, load_weights: bool = False
  ) -> ModuleSpec:
    """Creates a ModuleSpec for a Transformer model.

    Automatically detects the architecture from the config and prepares
    integer-based input tensors (token IDs).

    Args:
      name: Name (str) of the hf transformer model.
      load_weights: If True, loads pretrained weights. If False, initializes
        with random weights.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    if load_weights:
      model_dir_or_repo_id = name
      if self._weights_dir:
        model_dir_or_repo_id = self._weights_dir / name

      model_fn = lambda: transformers.AutoModelForCausalLM.from_pretrained(
          model_dir_or_repo_id
      )
      preprocessor_fn = lambda: (
          transformers.AutoTokenizer.from_pretrained(model_dir_or_repo_id)
      )
      config = transformers.AutoConfig.from_pretrained(model_dir_or_repo_id)
    else:
      with resources.as_file(
          self._FILES.joinpath(str(pathlib.Path(name) / "config.json"))
      ) as f:
        config = transformers.AutoConfig.from_pretrained(str(f))
      architectures = getattr(config, "architectures", [])
      model_cls = transformers.AutoModel

      if architectures:
        try:
          model_cls = getattr(transformers, architectures[0])
        except AttributeError:
          logging.warning(
              "Could not find architecture %s, falling back to AutoModel.",
              architectures[0],
          )

      model_fn = lambda: model_cls(config)
      preprocessor_fn = None

    safe_seq_len = min(_get_max_seq_len(config), 512)

    def _input_fn(
        shape=None, device="cpu"
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
      actual_shape = shape if shape is not None else (1, safe_seq_len)
      return (
          (),
          {
              "input_ids": torch.randint(
                  0, config.vocab_size, actual_shape, device=device
              ),
              "attention_mask": torch.ones(
                  actual_shape, device=device, dtype=torch.long
              ),
          },
      )

    return ModuleSpec(model_fn, _input_fn, preprocessor_fn, config)


class ModuleRegistry:
  """Central registry for managing multiple model providers."""

  def __init__(self):
    self._providers: dict[str, BaseProvider] = {
        "torchvision": TorchvisionProvider(),
        "timm": TimmProvider(),
        "transformers": TransformersProvider(),
    }

  def list_all_sources(self) -> list[str]:
    """Returns a list of registered provider keys (e.g., 'torchvision')."""
    return list(self._providers.keys())

  def list_all_modules(self) -> list[str]:
    """Returns a unified list of all available modules across all providers.

    Returns:
      A list of strings formatted as '{source}/{model_name}'.
    """
    modules = []
    for key, p in self._providers.items():
      for m in p.list_modules():
        modules.append(f"{key}/{m}")
    return modules

  def list_modules(self, source: str) -> list[str]:
    """Lists available models for a specific source.

    Args:
      source: The provider key (e.g., 'timm', 'transformers').

    Returns:
      A list of model names.

    Raises:
      ValueError: If the source is not found in the registry.
    """
    if source not in self._providers:
      raise ValueError(f"Source '{source}' not supported.")
    return self._providers[source].list_modules()

  def get_module_spec(
      self, source: str, name: str, *, load_weights: bool = False
  ) -> ModuleSpec:
    """Instantiates and returns the ModuleSpec for a specific model.

    Args:
      source: The provider key.
      name: The name of the model within that provider.
      load_weights: Whether to load pre-trained weights.

    Returns:
      A ModuleSpec containing the model factory and input factory.

    Raises:
      ValueError: If the source is not found in the registry.
    """
    if source not in self._providers:
      raise ValueError(f"Source '{source}' not supported.")

    return self._providers[source].get_module_spec(
        name, load_weights=load_weights
    )
