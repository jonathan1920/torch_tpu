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

import os
import pathlib
from absl import app
from PIL import Image
import torch
from examples import paths
import transformers
from rules_python.python.runfiles import runfiles


def main(argv):
  del argv
  config_file = runfiles.Create().Rlocation(
      "__main__/examples/huggingface_transformers/model_configs/Qwen/Qwen3-VL-2B-Instruct/config.json"
  )
  model_path = os.path.dirname(config_file)

  print("Loading config...", flush=True)
  config = transformers.AutoConfig.from_pretrained(model_path)
  print(f"Config loaded: {type(config)}", flush=True)

  print("Loading processor...", flush=True)
  tokenizer_path = (
      pathlib.Path(paths.XM_HOME)
      / "weights"
      / "huggingface"
      / "Qwen"
      / "Qwen3-0.6B"
  )
  tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
  image_processor = transformers.AutoImageProcessor.from_pretrained(model_path)
  try:
    video_processor = transformers.AutoVideoProcessor.from_pretrained(
        model_path
    )
  except Exception:
    video_processor = None

  chat_template_file = os.path.join(model_path, "chat_template.json")
  if os.path.exists(chat_template_file):
    with open(chat_template_file, "r") as f:
      chat_template = f.read()
  else:
    chat_template = None

  processor = transformers.Qwen3VLProcessor(
      image_processor=image_processor,
      video_processor=video_processor,
      tokenizer=tokenizer,
      chat_template=chat_template,
  )
  print(f"Processor loaded: {type(processor)}", flush=True)

  print("Instantiating model with random weights...", flush=True)
  try:
    model = transformers.AutoModelForConditionalGeneration.from_config(config)
    print(
        "Model instantiated via AutoModelForConditionalGeneration:"
        f" {type(model)}",
        flush=True,
    )
  except AttributeError:
    print(
        "AutoModelForConditionalGeneration not found, trying"
        " Qwen3VLForConditionalGeneration",
        flush=True,
    )
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration(config)
    print(
        "Model instantiated via Qwen3VLForConditionalGeneration:"
        f" {type(model)}",
        flush=True,
    )

  model = model.to(torch.bfloat16)

  # Create dummy input
  image = Image.new("RGB", (224, 224), color="red")
  messages = [{
      "role": "user",
      "content": [
          {"type": "image", "image": image},
          {"type": "text", "text": "Describe this image."},
      ],
  }]

  text = processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )

  print("Applying chat template...", flush=True)
  inputs = processor(
      text=[text],
      images=[image],
      padding=True,
      return_tensors="pt",
  )
  print(f"Inputs keys: {inputs.keys()}", flush=True)
  for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
      print(f"  {k}: shape {v.shape}, dtype {v.dtype}", flush=True)
    else:
      print(f"  {k}: {type(v)}", flush=True)

  print("Running forward pass...", flush=True)
  with torch.no_grad():
    outputs = model(**inputs)
  print("Forward pass done.", flush=True)
  print(f"Outputs keys: {outputs.keys()}", flush=True)
  if "logits" in outputs:
    print(f"Logits shape: {outputs.logits.shape}", flush=True)


if __name__ == "__main__":
  app.run(main)
