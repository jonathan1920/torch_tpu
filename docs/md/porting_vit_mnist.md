---
title: Porting Vit Mnist
marimo-version: 0.19.9
width: medium
---

```python {.marimo hide_code="true"}
import marimo as mo
```

# Porting a Vision Transformer (ViT) to TorchTPU

This tutorial demonstrates how to port a Vision Transformer (ViT) model to the TorchTPU backend. We'll implement a "Tiny ViT" optimized for TPU performance and train it on the MNIST dataset.
<!---->
## 1. The Dispatcher Shim
First, we implement a shim to prevent crashes when libraries attempt to look up `torchvision` operators that might not be present in the environment.

```python {.marimo}
import torch
# Prevents crashes when libraries attempt to look up torchvision operators
try:
    torch.library.Library("torchvision", "DEF").define(
        "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
    )
except Exception:
    pass
```

## 2. Model Architecture (Optimized for TPU)
We define the `TpuViT` architecture using a `Conv2d` patch embedding — the standard ViT approach where a single convolution with `kernel_size=patch_size` and `stride=patch_size` extracts and projects all patches in one operation.

```python {.marimo}
import torch.nn as nn
import torch.nn.functional as F
import math

class TpuAttentionBlock(nn.Module):
    """Manual multi-head attention — avoids SDP MATH fallback on TPU."""
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x):
        B, N, C = x.shape
        # Self-attention with residual
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.out_proj(h)
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        return x

class TpuViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_layers=4, hidden_dim=128, num_heads=4):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2

        # Conv2d patch embedding: kernel_size=stride=patch_size
        self.patch_embed = nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        self.blocks = nn.Sequential(*[
            TpuAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp_head = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        # x shape: [B, 1, 32, 32]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

        x = x + self.pos_embedding
        x = self.blocks(x)
        x = self.norm(x)

        return self.mlp_head(x.mean(dim=1))
```

## 3. Pure Tensor Data Preparation
We fetch the MNIST dataset and prepare it as pure Torch tensors. We resize the images to 32x32 to match our model's expected input size.

```python {.marimo}
from sklearn.datasets import fetch_openml
import numpy as np

print("Fetching MNIST data...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

# Train: first 1000, Test: next 200
X_train = mnist.data[:1000].astype(np.float32) / 255.0
Y_train = mnist.target[:1000].astype(np.int64)
X_test = mnist.data[1000:1200].astype(np.float32) / 255.0
Y_test = mnist.target[1000:1200].astype(np.int64)

# Resize 28x28 -> 32x32 using core functional ops
X_tensor = torch.from_numpy(X_train).view(-1, 1, 28, 28)
X_tensor = F.interpolate(X_tensor, size=(32, 32))
Y_tensor = torch.from_numpy(Y_train)

X_test_tensor = torch.from_numpy(X_test).view(-1, 1, 28, 28)
X_test_tensor = F.interpolate(X_test_tensor, size=(32, 32))
Y_test_tensor = torch.from_numpy(Y_test)

print(f"Train: {len(X_tensor)} samples, Test: {len(X_test_tensor)} samples")
```

## 4. Training Loop
Finally, we initialize the TPU device using `safe_init`, move the model to the device, and run the training loop.

```python {.marimo}
from tpu_utils import safe_init

# Self-healing hardware initialization
device = safe_init()
print(f"Executing on: {device}")
```

```python {.marimo}
# Move model to TPU (float32 — required by SDP attention fast path)
model = TpuViT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Starting training loop...")
model.train()
for epoch in range(4):   # Change to at least 15 to see the model start to converge. 
    idx = 0
    batch_size = 32
    while idx < len(X_tensor):
        inputs = X_tensor[idx : idx + batch_size].to(device)
        targets = Y_tensor[idx : idx + batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        idx += batch_size

    print(f"Epoch {epoch} complete. Last Loss: {loss.item():.4f}")

print("\n" + "=" * 40)
print("Verification complete: torchtpu is functional")
print("=" * 40)
```

## 5. Inference Demo
Now that the model is trained, let's run inference on a single test image and see what the model predicts.

```python {.marimo}
import matplotlib.pyplot as plt
import io, base64, random

# Pick a random test image
test_idx = random.randint(0, len(X_test_tensor) - 1)
test_img = X_test_tensor[test_idx:test_idx+1]  # [1, 1, 32, 32]
true_label = Y_test_tensor[test_idx].item()

# Run inference on TPU
model.eval()
with torch.no_grad():
    logits = model(test_img.to(device))
    predicted = logits.argmax(dim=1).cpu().item()

# Render the image
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(test_img[0, 0].numpy(), cmap="gray")
color = "green" if predicted == true_label else "red"
ax.set_title(f"Predicted: {predicted}  |  Actual: {true_label}", fontsize=14, color=color)
ax.axis("off")
plt.tight_layout()

# Convert to embeddable image
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
plt.close(fig)
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode()

result_text = "✅ Correct!" if predicted == true_label else "❌ Incorrect"
mo.output.replace(
    mo.vstack([
        mo.md(f"### {result_text}"),
        mo.image(src=f"data:image/png;base64,{img_b64}"),
    ])
)
```