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

# pylint: skip-file
import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.nav_menu(
      {
          "#tutorials": f"{mo.icon('lucide:graduation-cap')} Tutorials",
          "#guides": f"{mo.icon('lucide:wrench')} How-to Guides",
          "#reference": f"{mo.icon('lucide:book-open')} Reference",
          "#explanation": f"{mo.icon('lucide:brain')} Explanation",
      },
      orientation="horizontal",
  )
  return


@app.cell(hide_code=True)
def _(mo):
  mo.Html(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
.portal-hero{{
    background:linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);
    color:#fff;padding:52px 48px 44px;border-radius:20px;
    text-align:center;position:relative;overflow:hidden;margin-bottom: 32px;
}}
.portal-hero::before{{
    content:'';position:absolute;inset:-50%;
    background:radial-gradient(ellipse at 40% 50%,rgba(99,102,241,.2),transparent 60%);
    animation:heroGlow 8s ease-in-out infinite alternate;pointer-events:none;
}}
@keyframes heroGlow{{
    0%{{opacity:.4;transform:scale(1)}} 100%{{opacity:1;transform:scale(1.12)}}
}}
.portal-hero h1{{
    font-family:'Inter',system-ui,sans-serif;
    font-size:2.6em;font-weight:800;margin:0 0 10px;
    position:relative;letter-spacing:-.03em;color:#fff;text-align:center;
}}
.portal-hero .subtitle{{
    font-family:'Inter',system-ui,sans-serif;
    font-size:1.15em;font-weight:300;opacity:.75;margin:0;position:relative;
}}
.portal-hero .tip{{
    display:inline-block;margin-top:22px;padding:9px 20px;
    background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);
    border-radius:999px;font-family:'Inter',system-ui,sans-serif;
    font-size:.88em;position:relative;backdrop-filter:blur(6px);
}}
.portal-hero .tip code{{
    background:rgba(255,255,255,.14);padding:2px 8px;border-radius:5px;font-size:.92em;
}}
.portal-section{{margin:48px 0}}
.section-header{{display:flex;align-items:center;gap:14px;margin-bottom:12px}}
.section-badge{{
    width:46px;height:46px;border-radius:13px;
    display:flex;align-items:center;justify-content:center;
    font-size:1.4em;flex-shrink:0;
}}
.section-badge.blue{{background:rgba(66,133,244,.10)}}
.section-badge.green{{background:rgba(52,168,83,.10)}}
.section-badge.yellow{{background:rgba(251,188,4,.10)}}
.section-badge.red{{background:rgba(234,67,53,.10)}}
.section-title{{
    font-family:'Inter',system-ui,sans-serif;
    font-size:1.5em;font-weight:700;margin:0;letter-spacing:-.01em;
}}
.section-subtitle{{
    font-family:'Inter',system-ui,sans-serif;
    font-size:.95em;opacity:.6;margin:4px 0 20px 60px;line-height:1.5;
}}
.card-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}
@media(max-width:680px){{
    .card-grid{{grid-template-columns:1fr}}
    .portal-hero{{padding:40px 24px}}
    .portal-hero h1{{font-size:2em}}
}}
.portal-card{{
    display:flex;align-items:flex-start;gap:14px;padding:24px 26px;
    border-radius:15px;border:1px solid rgba(0,0,0,.06);
    text-decoration:none !important;color:inherit !important;
    transition:transform .25s cubic-bezier(.4,0,.2,1),
               box-shadow .25s cubic-bezier(.4,0,.2,1),
               background .25s ease;
    background:rgba(0,0,0,.02);cursor:pointer;
}}
.portal-card:hover{{transform:translateY(-4px);box-shadow:0 12px 32px rgba(0,0,0,.08)}}
.portal-card.blue{{border-left:5px solid #4285F4}}
.portal-card.green{{border-left:5px solid #34A853}}
.portal-card.yellow{{border-left:5px solid #FBBC04}}
.portal-card.red{{border-left:5px solid #EA4335}}
.portal-card.blue:hover{{background:rgba(66,133,244,.05)}}
.portal-card.green:hover{{background:rgba(52,168,83,.05)}}
.portal-card.yellow:hover{{background:rgba(251,188,4,.05)}}
.portal-card.red:hover{{background:rgba(234,67,53,.05)}}
.card-num{{
    font-family:'Inter',system-ui,sans-serif;
    font-size:.85em;font-weight:700;width:32px;height:32px;border-radius:10px;
    display:flex;align-items:center;justify-content:center;flex-shrink:0;
    background:rgba(66,133,244,.1);color:#4285F4;
}}
.card-body{{flex:1}}
.card-title{{
    font-family:'Inter',system-ui,sans-serif;
    font-weight:600;font-size:1.1em;margin-bottom:6px;
}}
.card-desc{{
    font-family:'Inter',system-ui,sans-serif;
    font-size:.92em;opacity:.65;line-height:1.6;margin:0;
}}
</style>
<script>
(function() {{
    const isStatic = () => {{
        return window.location.pathname.endsWith('.html') || 
               window.location.protocol === 'file:' ||
               (!window.location.search.includes('access_token') && window.location.hostname !== 'localhost');
    }};

    if (isStatic()) {{
        const rewrite = (a) => {{
            const href = a.getAttribute('href');
            if (href && href.startsWith('?file=') && href.endsWith('.py')) {{
                a.href = href.replace('?file=', '').replace('.py', '.html');
            }}
        }};
        
        const observer = new MutationObserver((mutations) => {{
            mutations.forEach(m => {{
                m.addedNodes.forEach(node => {{
                    if (node.nodeType === 1) {{
                        if (node.tagName === 'A') rewrite(node);
                        node.querySelectorAll('a[href^="?file="]').forEach(rewrite);
                    }}
                }});
            }});
        }});
        
        observer.observe(document.body, {{ childList: true, subtree: true }});
        document.querySelectorAll('a[href^="?file="]').forEach(rewrite);
        // Interval fallback for extremely late renders
        setInterval(() => {{
            document.querySelectorAll('a[href^="?file="]').forEach(rewrite);
        }}, 2000);
    }}
}})();
</script>
""")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
<div class="portal-hero">
    <h1 style="text-align: center;">🚀 TorchTPU Interactive Portal</h1>
    <p class="subtitle" style="text-align: center;">Your hub for using PyTorch on Google's TPUs.</p>
    <div class="tip">💡 Run <code>marimo edit tutorials/</code> to browse all notebooks</div>
</div>
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
<details style="margin-bottom: 32px;">
<summary><strong>⚙️ Installation — Set up TorchTPU on your TPU VM</strong></summary>

Install TorchTPU and its dependencies on your TPU VM. Components must be installed in this order to ensure the PyTorch dispatcher and TPU drivers align correctly.

### Step 1: Initialize Python 3.12

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv env
source env/bin/activate
```

### Step 2: Install TorchTPU & Dependencies

```bash
# 1. Authenticate to get access to whl
pip install keyrings.google-artifactregistry-auth
gcloud auth login
gcloud auth application-default login

# 2. Install torch_tpu, PyTorch 2.10 CPU will install automatically, do not install manually.
pip install --pre --index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" torch_tpu

# 3. Install optional utilities
pip install portpicker marimo scikit-learn pandas

#4. Optional for Pallas kernels
pip install jax

#5. Optional for xProf
pip install tensorboard
```

</details>
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
<div id="tutorials" class="portal-section">
    <div class="section-header">
        <div class="section-badge blue">🎓</div>
        <h2 class="section-title">Tutorials</h2>
    </div>
    <p class="section-subtitle">Step-by-step lessons that teach core concepts hands-on.</p>
    <div class="card-grid">
        <a href="?file=get_started_tpu.py" class="portal-card blue">
            <span class="card-num">1</span>
            <div class="card-body">
                <div class="card-title">Get Started with TorchTPU</div>
                <div class="card-desc">Device init, deferred execution, and materialization</div>
            </div>
        </a>
        <a href="?file=porting_vit_mnist.py" class="portal-card blue">
            <span class="card-num">2</span>
            <div class="card-body">
                <div class="card-title">Your First Port (ViT on MNIST)</div>
                <div class="card-desc">Ghosting shims, Linear patching, and pure-tensor pipelines</div>
            </div>
        </a>
        <a href="?file=debugger_toolkit.py" class="portal-card blue">
            <span class="card-num">3</span>
            <div class="card-body">
                <div class="card-title">The Debugger's Toolkit</div>
                <div class="card-desc">OpTracer, format_model, and graph visualization</div>
            </div>
        </a>
        <a href="?file=distributed_training_101.py" class="portal-card blue">
            <span class="card-num">4</span>
            <div class="card-body">
                <div class="card-title">Distributed Training 101</div>
                <div class="card-desc">Multi-core scaling (DDP) and the TPU environment orchestrator</div>
            </div>
        </a>
    </div>
</div>
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
<div id="guides" class="portal-section">
    <div class="section-header">
        <div class="section-badge green">🛠️</div>
        <h2 class="section-title">How-to Guides</h2>
    </div>
    <p class="section-subtitle">Focused recipes for solving specific problems.</p>
    <div class="card-grid">
        <a href="?file=numerical_parity.py" class="portal-card green">
            <div class="card-body">
                <div class="card-title">Numerical Parity Workflow</div>
                <div class="card-desc">Validating TPU outputs match CPU within tolerance</div>
            </div>
        </a>
        <a href="?file=scaling_fsdp.py" class="portal-card green">
            <div class="card-body">
                <div class="card-title">Scaling to Large Models (FSDP)</div>
                <div class="card-desc">Sharding models across TPU chips with FSDP</div>
            </div>
        </a>
        <a href="?file=advanced_compile.py" class="portal-card green">
            <div class="card-body">
                <div class="card-title">Advanced torch.compile on TPU</div>
                <div class="card-desc">Graph breaks, backends, and compilation debugging</div>
            </div>
        </a>
        <a href="?file=mixed_precision.py" class="portal-card green">
            <div class="card-body">
                <div class="card-title">Mixed Precision (BFloat16 & AMP)</div>
                <div class="card-desc">Casting, autocasting, and precision diagnostics</div>
            </div>
        </a>
        <a href="?file=custom_pallas_kernels.py" class="portal-card green">
            <div class="card-body">
                <div class="card-title">Custom Kernels with Pallas</div>
                <div class="card-desc">Writing custom TPU kernels with JAX Pallas</div>
            </div>
        </a>
        <a href="?file=performance_profiling.py" class="portal-card green">
            <div class="card-body">
                <div class="card-title">Performance Profiling</div>
                <div class="card-desc">Capturing and analyzing TPU performance traces</div>
            </div>
        </a>
    </div>
</div>
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
<div id="reference" class="portal-section">
    <div class="section-header">
        <div class="section-badge yellow">📖</div>
        <h2 class="section-title">Reference</h2>
    </div>
    <p class="section-subtitle">Lookup tables and quick-reference materials.</p>
    <div class="card-grid">
        <a href="?file=api_mapping.py" class="portal-card yellow">
            <div class="card-body">
                <div class="card-title">API Mapping Table</div>
                <div class="card-desc">PyTorch ↔ TorchTPU API equivalents</div>
            </div>
        </a>
        <a href="?file=environment_variables.py" class="portal-card yellow">
            <div class="card-body">
                <div class="card-title">Environment Variables</div>
                <div class="card-desc">All env vars that control TorchTPU behavior</div>
            </div>
        </a>
        <a href="?file=error_glossary.py" class="portal-card yellow">
            <div class="card-body">
                <div class="card-title">Error Code Glossary</div>
                <div class="card-desc">Common errors, what they mean, and how to fix them</div>
            </div>
        </a>
        <a href="?file=hardware_alignment.py" class="portal-card yellow">
            <div class="card-body">
                <div class="card-title">Hardware Alignment Cheat Sheet</div>
                <div class="card-desc">MXU tile sizes, padding rules, and performance cliffs</div>
            </div>
        </a>
    </div>
</div>
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
<div id="explanation" class="portal-section">
    <div class="section-header">
        <div class="section-badge red">🧠</div>
        <h2 class="section-title">Explanation</h2>
    </div>
    <p class="section-subtitle">Deep dives into how TorchTPU works under the hood.</p>
    <div class="card-grid">
        <a href="?file=xla_stablehlo_pipeline.py" class="portal-card red">
            <div class="card-body">
                <div class="card-title">The XLA & StableHLO Pipeline</div>
                <div class="card-desc">How PyTorch ops become TPU machine code</div>
            </div>
        </a>
        <a href="?file=compilation_cache.py" class="portal-card red">
            <div class="card-body">
                <div class="card-title">The Compilation Cache</div>
                <div class="card-desc">How compiled graphs are cached and reused</div>
            </div>
        </a>
        <a href="?file=deferred_execution.py" class="portal-card red">
            <div class="card-body">
                <div class="card-title">The Deferred Execution Model</div>
                <div class="card-desc">DeviceBufferRef, promises, and materialization triggers</div>
            </div>
        </a>
        <a href="?file=spmd_synchronization.py" class="portal-card red">
            <div class="card-body">
                <div class="card-title">Strict SPMD Synchronization</div>
                <div class="card-desc">Why all ranks must execute the same graph</div>
            </div>
        </a>
    </div>
</div>
    """)
  return


if __name__ == "__main__":
  app.run()
