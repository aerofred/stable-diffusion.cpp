# Backend selection

`stable-diffusion.cpp` has two backend assignments:

- `--backend` selects the runtime backend used to execute model graphs.
- `--params-backend` selects where model parameters are kept.

If `--params-backend` is not set, auto-fit chooses parameter placement. With
`--auto-fit off`, parameters use the same backend as their module runtime backend.

## Syntax

A backend assignment can be a single backend name:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cpu
```

This applies to every module that does not have a more specific assignment.

Assignments can also target individual modules:

```shell
sd-cli -m model.safetensors -p "a cat" --backend te=cpu,vae=cuda0,diffusion=vulkan0
```

The same syntax is used for parameter placement:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend te=cpu,vae=cpu
```

`--params-backend` also accepts the special value `disk`:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend disk
```

`--max-vram` can target resolved backend/device names:

```shell
sd-cli -m model.safetensors -p "a cat" --backend diffusion=cuda0,vae=vulkan0 --max-vram cuda0=6,vulkan0=2
```

The value is a shared per-device budget for managed weights and registered
runner compute/cache buffers. Live free memory can lower the effective limit
for each graph run. Driver contexts and allocations made outside the managed
model runners are not part of this accounting, so it is not a hard physical
VRAM cap.

Module names are case-insensitive. Hyphens and underscores in module names are ignored, so `clip_vision`, `clip-vision`, and `clipvision` are equivalent.

`all=`, `default=`, and `*=` can be used to set the default backend inside a mixed assignment:

```shell
sd-cli -m model.safetensors -p "a cat" --backend all=cuda0,te=cpu
```

## Multiple devices per module (layer split)

A `--backend` module assignment can list several devices separated by `&`:

```shell
sd-cli -m model.safetensors -p "a cat" --backend "diffusion=cuda0&cuda1"
```

The module's transformer blocks are then distributed across the listed devices
in contiguous ranges: the first device is filled up to its free memory (minus a
compute-buffer headroom of about 2 GiB per device) before the next one receives
blocks, so a model that fits on the first device stays there. Only when the
weights exceed the combined budget are they spread proportionally to each
device's budget. The module's graphs are executed with a `ggml_backend_sched`
that runs each block on the device holding its weights, copying the residual
stream at the range boundaries. Execution is sequential across devices: a
split does not run blocks on two GPUs at the same time. The first device in the list is the module's main device: it also
holds the non-block tensors (embeddings, final norms, small sub-runners such as
CLIP models or projectors) and the graph inputs/outputs.

Layer split is supported for the `diffusion` and `te` modules. For `te` it
applies to the dominant text encoder (`t5xxl` or the LLM); other modules accept
only a single device. If the module has no recognizable transformer blocks, the
assignment falls back to the first listed device.

`--params-backend` accepts no device lists. If the module has no explicit
params assignment, each block range's parameters are loaded directly to (and,
with `--params-backend diffusion=disk`, released directly from) its own device;
an explicit assignment such as `te=cpu` keeps the parameters on that backend
and stages each range to its device on demand.

Layer split uses the fixed graph-cut plan to assign blocks across devices. When
the module's weights and compute buffers fit resident on the listed devices, the
whole graph runs in one scheduler pass. Otherwise the graph runs segment by
segment, exactly like single-device segmented execution: each block's weights are
staged to (and prefetched on) the device that owns it, the residual stream is
copied across devices only at range boundaries, and the resident weight cache
spans every listed device. Weights that exceed the combined device budgets are
spread across the devices in proportion to their budgets and streamed per segment
instead of being rejected. `--max-vram` can still provide the per-device limits
used by layer split and auto-fit.

Note that layer split pools the devices' memory for *weights*, not for
activations: every block still executes on a single device, so a module's
compute buffer (which grows with resolution and frame count) must fit on each
device individually. Splitting a module across two GPUs therefore does not raise
the maximum resolution or frame count; it lets more of a large model stay
resident instead of being re-staged from RAM at every step.

### Placement policy (`--split-ratio`)

`--split-ratio` chooses how a layer split places blocks on the listed devices:

- `auto` (default): on the first split graph each device is benchmarked once
  (an f16 matmul and a pinned host-to-device upload, about 100 ms per device).
  Each block then goes where its estimated cost is lowest: compute time on the
  device, plus the weight transfer when the block does not fit resident, plus
  the residual-stream copy when the block changes device. A block is streamed
  from RAM on a fast device whenever that beats running it resident on a slower
  one, so a GPU on a narrow PCIe link only receives blocks it can keep resident
  and only when that is faster than streaming. The log prints the measured
  profile and the resulting placement with its estimated time per graph.
- `vram`: fill each device by free memory in `--backend` order (previous behavior).
- `r0,r1,...`: fixed fractions of the parameter-bearing blocks per device, in
  `--backend` order, e.g. `--split-ratio 0.7,0.3`.

```shell
sd-cli -m model.safetensors -p "a cat" --backend "diffusion=cuda0&cuda1" --split-ratio 0.7,0.3
```

### Parallel classifier-free guidance (`--backend cfg=<device>`)

With `cfg=<device>`, a second copy of the diffusion model is built on that GPU
and runs the unconditional pass of every sampling step concurrently with the
conditional pass on the diffusion device:

```shell
sd-cli -m model.safetensors -p "a cat" --cfg-scale 3.5 --backend diffusion=cuda0,cfg=cuda1
```

Both devices hold or stream their own copy of the weights (with
`--offload-to-cpu` this also doubles the diffusion model's RAM copy), so the
step time becomes the slower of the two passes instead of their sum. This only
helps when the model fits resident on both devices, or streams fast enough on
both. The replica follows the `diffusion` params backend and LoRA settings. The
sequential path is kept when there is no unconditional pass (`--cfg-scale 1`),
for the high-noise model of two-stage models, with ControlNet, with step
caches (EasyCache and friends), with generation extensions such as PhotoMaker,
and when `cfg` names the diffusion device itself.

Use `--list-devices` to see the device names available on the system.

### Row split (`--split-mode row`)

`--split-mode` selects how a multi-device module distributes its weights:
`layer` (the default, described above) or `row`. It accepts a single mode or
per-module assignments:

```shell
sd-cli -m model.safetensors -p "a cat" --backend "diffusion=cuda0&cuda1" --split-mode row
sd-cli -m model.safetensors -p "a cat" --backend "diffusion=cuda0&cuda1,te=cuda0&cuda1" --split-mode diffusion=row,te=layer
```

In row mode the module keeps executing on its main (first listed) device, but
its transformer-block matmul weights are allocated in the backend's row-split
buffer type, which slices each weight's rows across the listed devices in
proportion to free memory and runs those matmuls on all devices in parallel.
Compared to a layer split this uses all GPUs within every layer (instead of
sequentially device by device) at the cost of a cross-device reduction per
matmul - usually the faster option when the devices have fast interconnect.

Row split requires a compatible split-buffer export from the linked GGML
backend. If it is unavailable (or the listed devices belong to different backend
registries), the module falls back to a layer split.
Embeddings, normalization weights, biases and other non-block tensors stay in
regular buffers on the main device.

Row-split execution can use graph segments, but split weights are loaded
synchronously instead of using the normal single-device prefetch path. Because
GGML does not expose exact shard allocation sizes, the managed budget currently
counts a split buffer's full size on each participating device. This is a
conservative bound and can reject otherwise feasible layouts.

Direct ("immediately") LoRA application cannot patch row-split tensors; with
`--split-mode row` the automatic LoRA mode selects runtime application, and an
explicit `--lora-apply-mode immediately` skips the split tensors with a
warning.

## Automatic placement (`--auto-fit on|off`)

`--auto-fit` requires `on` or `off` and defaults to `on` when omitted.
Explicit `--params-backend` assignments disable auto-fit,
regardless of argument order, even with `--auto-fit on`.

Auto-fit preserves explicit `--backend` assignments, including per-module
assignments and device lists. For modules without a runtime assignment, it chooses
the GPU with the largest available memory budget (the first device on a tie).
It then derives parameter placements from the model metadata, each module's
compute devices, and the remaining memory budgets. The chosen backend
specifications are printed.

```shell
sd-cli -m model.safetensors -p "a cat" --auto-fit on
sd-cli -m model.safetensors -p "a cat" --auto-fit on --max-vram cuda0=8,cuda1=14
sd-cli -m model.safetensors -p "a cat" --backend cuda0
sd-cli -m model.safetensors -p "a cat" --backend diffusion=cuda0,te=cpu,vae=cuda1
sd-cli -m model.safetensors -p "a cat" --auto-fit off
```

Budgets reuse `--max-vram`: a positive per-device value caps what auto-fit
plans with on that device, a negative value means "free memory minus that many
GiB", and with no budget set each device's free memory minus a 512 MiB margin
is used. These resolved GPU budgets, including the safety margin, also drive
the runner's graph-cut capacity checks.

Runtime capacity checks also leave 512 MiB of currently free device memory for
backend scratch buffers and pipelines, including with explicit backend assignments.
They cap free-memory reports by the device's total memory minus tracked
resident allocations. Vulkan reports exceeding total memory are rejected because
its heap-budget subtraction can underflow. Other backends use the cap instead of
treating such reports as zero free memory. Failed checks log the reported free and
total memory alongside tracked weight and runtime allocations.

Components are considered in `diffusion`, `te`, `vae` order so that repeatedly
used diffusion weights have priority. Each component's weights use the first
storage location with enough remaining budget:

1. The component's compute GPU, leaving estimated space for computation and weight staging.
2. CPU RAM, reserving the larger of 2 GiB or 10% of available RAM for other work.
3. Another GPU, choosing the one with the largest remaining budget that fits.
4. Disk, reloading weights on demand.

GPU cache space follows the same component priority. Before a lower-priority
component can become permanently resident, the planner leaves room for the full
weights and estimated compute space of higher-priority offloaded components.
If offloaded diffusion already needs the entire main GPU budget, TE and VAE also
use offloaded parameters. Their GPU copies can then be released after their
phases, leaving more room to reuse diffusion weights across sampling steps.
CPU parameter residency allows GPU weight caching; it does not force every
weight to be copied again at every step.

RAM and GPU budgets are shared across components. Each component uses a single
parameter backend; several other GPUs' capacities are not combined to store
one component. If available RAM cannot be queried, RAM residency is skipped.
Weights stored on another GPU are copied to the component's compute devices for
execution. CPU modules use RAM or disk. Compute reserves and cache priority are
accounted for separately on each device, so a CPU module does not reserve GPU
space. Storage on another module's GPU also leaves room for that module's work.

Auto-fit does not select multi-GPU layer/row computation itself. Explicit device
lists and `--split-mode` still control that computation. Before the runners have
built their split plans, auto-fit conservatively counts the full component size
on each listed GPU when checking residency and cache space. This can offload
parameters even when a split layout would fit; use `--auto-fit off` to keep the
default split-device parameter placement.

For example, a diffusion model whose full weights exceed the main GPU's budget
can use `--backend diffusion=cuda0 --params-backend diffusion=cpu` when RAM is
sufficient. Automatic graph segmentation can then load the required weights
for each segment and reclaim idle GPU copies. `--disable-segmented-compute`
still disables segmentation.

Initial compute reserves are estimates (2 GiB for diffusion and text encoders,
1 GiB for VAE); higher-priority placements also leave staging space for the
largest weight tensor of each lower-priority offloaded component. Actual segment
weights, compute buffers and caches must
still fit the runner's capacity checks. Offloading weights does not guarantee
that every resolution or frame count will fit, and auto-fit does not change a
component to CPU computation solely because its full weights exceed VRAM.
If a VAE decode fails, decoding retries with spatial tiling even when `--auto-fit`
is off; supported video decoders try temporal tiling first and can then add
spatial tiling. Spatial retries use half-size tiles along each latent dimension.

## Modules

| Module | Purpose | Accepted names |
| --- | --- | --- |
| `diffusion` | UNet, DiT, MMDiT, Flux, Wan, Qwen Image, and other diffusion models | `diffusion`, `model`, `unet`, `dit` |
| `te` | Text encoders and conditioners | `te`, `clip`, `text`, `textencoder`, `textencoders`, `conditioner`, `cond`, `llm`, `t5`, `t5xxl` |
| `clip_vision` | CLIP vision encoder | `clip_vision`, `clipvision`, `clip-vision`, `vision` |
| `vae` | VAE and TAE | `vae`, `firststage`, `autoencoder`, `tae` |
| `controlnet` | ControlNet | `controlnet`, `control` |
| `photomaker` | PhotoMaker ID encoder and PhotoMaker LoRA | `photomaker`, `photomakerid`, `pmid`, `photo` |
| `upscaler` | ESRGAN upscaler | `upscaler`, `esrgan`, `hires` |
| `detector` | ADetailer YOLOv8 detector | `detector`, `adetailer`, `yolo` |

`te` is the preferred module name for text encoders. `clip` is kept as an accepted alias because many existing commands and model names use CLIP terminology.

## Backend names

Backend names are resolved against the GGML backend device list. Matching is case-insensitive and accepts exact names or unique prefixes, so common values include names such as:

- `cpu`
- `cuda0`
- `vulkan0`
- `metal`

The special values `auto`, `default`, and an empty backend name select the default backend. The default preference is GPU, then integrated GPU, then CPU.

The special value `gpu` selects the first GPU backend, falling back to the first integrated GPU backend.

The special value `disk` is accepted only by `--params-backend`. `--backend disk` is invalid because `disk` is a parameter residency mode, not a runtime compute backend.

## Runtime backend vs. parameter backend

The runtime backend controls where graph execution runs. The parameter backend controls where model weights are allocated or whether they are reloaded from disk on demand.

For example:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend cpu
```

This runs all modules on `cuda0`, but stores parameters in CPU RAM. During execution, parameters are moved to the runtime backend as needed.

For example:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend disk
```

This runs all modules on `cuda0`, reloads parameters from the model file as needed, and releases those parameter buffers after use.

Outside `--auto-fit`, `disk` is never selected implicitly. If `--params-backend` is not set, parameters use the runtime backend.

Per-module assignments can be mixed:

```shell
sd-cli -m model.safetensors -p "a cat" --backend diffusion=cuda0,te=cpu,vae=cpu --params-backend diffusion=cuda0,te=cpu,vae=cpu
```

This keeps text encoding and VAE execution on CPU while the diffusion model runs on GPU.

## Backend sharing and lifetime

Backends are managed by `SDBackendManager`.

Within one manager, backend instances are cached by resolved backend device name. If multiple modules request the same backend, they share the same `ggml_backend_t`.

For example:

```shell
--backend te=cpu,vae=cpu
```

uses one shared CPU backend for both `te` and `vae` runtime execution.

Runtime and parameter assignments also share the same backend cache. If `--backend diffusion=cuda0` and `--params-backend diffusion=cuda0` resolve to the same device, both use the same backend instance.

`--params-backend disk` does not create a separate backend instance. Parameters are loaded lazily using the module runtime backend.

`SDBackendManager` owns the backend instances and frees them when the context or upscaler is destroyed. Model runners receive non-owning runtime and parameter backend pointers and do not free them.

## Compatibility flags

The example CLI/server still accepts these older CPU placement flags as compatibility aliases:

- `--clip-on-cpu`
- `--vae-on-cpu`
- `--control-net-cpu`
- `--offload-to-cpu`

`--clip-on-cpu`, `--vae-on-cpu`, and `--control-net-cpu` are deprecated. The example argument layer prepends `te=cpu`, `vae=cpu`, and `controlnet=cpu` to `--backend` before creating the context.

`--offload-to-cpu` prepends a CPU default to the parameter assignment in the caller before creating the context:

```shell
--params-backend '*=cpu'
```

Because this default is inserted first, later explicit `--params-backend` entries can still override it, for example `--offload-to-cpu --params-backend te=disk` keeps non-TE parameters on CPU and reloads TE parameters from disk.

Library callers should set `backend` and `params_backend` directly. `sd_ctx_params_init()`
enables `auto_fit` by default; a nonempty `params_backend` assignment disables it.
The `backend` assignment constrains auto-fit's compute placement.
The old CPU/offload fields are no longer part of the C API. Explicit `--backend` and
`--params-backend` assignments are preferred for new commands.
