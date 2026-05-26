# MMD Guidance

Code for **Training-Free Distribution Adaptation for Diffusion Models via Maximum Mean Discrepancy Guidance**.

Paper: https://arxiv.org/abs/2601.08379

This repository contains two parts:

- `Unconditional_MMD`: MMD Guidance for unconditional diffusion models.
- `Prompt_Aware_MMD`: prompt-aware MMD Guidance for conditional/text-guided generation.

The current implementation for unconditional MMD is based on the CompVis latent-diffusion codebase, with targeted changes that add MMD guidance during sampling.

## Unconditional MMD

The unconditional code is in:

```text
Unconditional_MMD/latent-diffusion
```

The main modified files are:

```text
Unconditional_MMD/latent-diffusion/scripts/sample_diffusion.py
Unconditional_MMD/latent-diffusion/ldm/models/diffusion/ddpm.py
```

`scripts/sample_diffusion.py` loads the LDM checkpoint, prepares reference features, and launches sampling. `ldm/models/diffusion/ddpm.py` contains the MMD guidance functions and applies the MMD gradient inside the reverse diffusion loop.

### Setup

Create the latent-diffusion environment from the provided environment file:

```bash
cd Unconditional_MMD/latent-diffusion
conda env create -f environment.yaml
conda activate ldm
```

Use the same checkpoint and model layout expected by latent-diffusion. For example:

```text
models/ldm/ffhq256/model.ckpt
models/ldm/ffhq256/config.yaml
```

### Run Unconditional Sampling

Example command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py \
  -r models/ldm/ffhq256/model.ckpt \
  -l log_output \
  --pickle_file precomputed.pkl \
  --dataset_metadata dataset_path \
  -n 500 \
  --batch_size 1 \
  -c 50 \
  -e 1 \
  -v \
  --n_references 150 \
  --alpha 0.002
```

Arguments:

- `-r`, `--resume`: path to the LDM checkpoint or checkpoint directory.
- `-l`, `--logdir`: output directory for generated samples.
- `--dataset_metadata`: directory containing the reference images used for MMD guidance.
- `--pickle_file`: path for cached reference features. If the file exists, it is loaded. Otherwise, `precompute_F_M_real` computes and saves it.
- `--n_references`: number of reference images used to compute the MMD reference set.
- `--alpha`: MMD guidance strength.
- `-v`, `--vanilla_sample`: use vanilla DDPM sampling. In this implementation, MMD guidance is applied in the DDPM sampling loop.
- `-n`, `--n_samples`: number of generated samples.
- `--batch_size`: sampling batch size.
- `-c`, `--custom_steps`: number of DDIM steps when DDIM sampling is used.
- `-e`, `--eta`: DDIM eta when DDIM sampling is used.

### Reference Feature Precomputation

MMD guidance compares generated samples against a reference set. In this implementation, reference images are encoded into the latent space of the LDM first-stage model.

The precomputation function is in:

```text
Unconditional_MMD/latent-diffusion/scripts/sample_diffusion.py
```

Core logic:

```python
def precompute_F_M_real(
    model, image_dir, device="cuda", output_file="features.pkl", n_references=500
):
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            F_M_real_all = pickle.load(f)
        return F_M_real_all

    image_paths = list(Path(image_dir).rglob("*.png"))[:n_references]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    F_M_real_all = None

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image)
        img_tensor = (img_tensor - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0).to(device)

        encoder_posterior = model.encode_first_stage(img_tensor)
        features = model.get_first_stage_encoding(encoder_posterior).detach()
        features = features * model.scale_factor

        if F_M_real_all is None:
            F_M_real_all = features.detach()
        else:
            F_M_real_all = torch.cat((F_M_real_all, features.detach()), dim=0)

    with open(output_file, "wb") as f:
        pickle.dump(F_M_real_all, f)

    return F_M_real_all
```

For a new diffusion model, this step should be adapted to the model's representation space. If your model samples directly in pixel space, `F_M_real` can be image-space tensors. If your model samples in latent space, encode the reference images using the same encoder/scale used by the sampler.

### MMD Guidance Hook

The MMD guidance logic is in:

```text
Unconditional_MMD/latent-diffusion/ldm/models/diffusion/ddpm.py
```

The connected functions are:

```python
def _flatten_feats(self, t):
    return t.reshape(t.size(0), -1)


def compute_kid_grad_poly3(self, real_img, gen_img, l2norm_img=False, real_weights=None, chunk=0):
    X = self._flatten_feats(real_img)
    Y = self._flatten_feats(gen_img)

    if l2norm_img:
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        Y = torch.nn.functional.normalize(Y, p=2, dim=1)

    Nr, D = X.size()
    Ng = Y.size(0)
    k = Ng - 1
    yk = Y[k]

    if real_weights is None:
        W = torch.ones(Nr, device=X.device, dtype=X.dtype)
    else:
        W = real_weights.to(device=X.device, dtype=X.dtype)
    W = (W / W.sum().clamp_min(1e-12)).detach()

    s_yy_k = Y.matmul(yk) / D + 1.0
    s_xy_k = X.matmul(yk) / D + 1.0

    c_yy = 3.0 * (s_yy_k * s_yy_k)
    c_xy = 3.0 * (s_xy_k * s_xy_k) * W

    grad_gg_k = (2.0 / (Ng * Ng * D)) * (Y.t().matmul(c_yy))
    grad_rg_k = (1.0 / (Ng * D)) * (X.t().matmul(c_xy))

    gk = grad_gg_k - 2.0 * grad_rg_k
    return gk.view_as(gen_img[-1])


def get_F_M(self, F, f):
    return torch.cat((F, f), dim=0)


def cond_fn(self, img, F_M_Samples, F_M_real, ts, last_step, criteria_guidance_scale=0.1):
    features_ = img.detach()

    if F_M_Samples is not None:
        _F = self.get_F_M(F_M_Samples, features_)
        grads = self.compute_kid_grad_poly3(F_M_real, _F)
        grads_same_scale = grads / (grads.norm(2).detach() + 1e-5) * img.norm(2).detach()
        grads = grads_same_scale * criteria_guidance_scale
    else:
        grads = torch.zeros_like(img)

    if last_step == 0:
        if F_M_Samples is None:
            F_M_Samples = features_
        else:
            F_M_Samples = self.get_F_M(F_M_Samples, features_)

    return grads, F_M_Samples
```

During sampling, `cond_fn` is called after the model denoising step, and the MMD gradient is subtracted from the current sample:

```python
img = self.p_sample(
    img,
    torch.full((b,), i, device=device, dtype=torch.long),
    clip_denoised=self.clip_denoised,
)

grad, F_M_Samples = self.cond_fn(
    img,
    F_M_Samples,
    F_M_real,
    torch.full((b,), i, device=device, dtype=torch.long),
    i,
    criteria_guidance_scale=guidance_scale,
)

img -= grad
```

### Plugging In Your Own Diffusion Model

To use MMD Guidance with another diffusion model, keep the model's original denoising logic unchanged and add MMD as an inference-time guidance hook.

1. Precompute reference features:

```python
F_M_real = precompute_F_M_real(
    model,
    image_dir="path/to/reference/images",
    output_file="precomputed.pkl",
    n_references=150,
)
```

2. Keep a generated-sample memory during sampling:

```python
F_M_Samples = None
```

3. At each reverse diffusion step, run the original sampler first:

```python
x = original_denoising_step(x, t, model, ...)
```

4. Apply the MMD guidance update:

```python
grad, F_M_Samples = cond_fn(
    x,
    F_M_Samples,
    F_M_real,
    t,
    last_step,
    criteria_guidance_scale=alpha,
)

x = x - grad
```

5. Make sure `F_M_real`, `F_M_Samples`, and `x` live in the same space:

- latent-space diffusion: use latent reference features.
- pixel-space diffusion: use image tensors or features from a chosen feature extractor.
- conditional diffusion: include conditioning normally in the original denoising step, then apply MMD guidance to the sampled representation.

The key requirement is that `cond_fn` receives comparable generated and reference tensors. The diffusion model itself does not need to be retrained.

## Prompt-Aware MMD

Prompt-aware MMD extends the same idea to conditional generation by making the MMD objective aware of both generated/reference samples and their associated prompts or conditioning information.

The prompt-aware implementation belongs in:

```text
Prompt_Aware_MMD
```

This section will contain the prompt-aware running instructions, reference-feature preparation, and product-kernel guidance details.

## License

This repository builds on the CompVis latent-diffusion codebase. The latent-diffusion components retain their original license, which is included in:

```text
Unconditional_MMD/latent-diffusion/LICENSE
```

The MMD Guidance modifications and additional code in this repository should be used under the top-level repository license. If you use this code, please also respect the licenses of the upstream latent-diffusion project and any pretrained checkpoints or datasets used in your experiments.

## Citation

```bibtex
@misc{mmd_guidance_2026,
  title={Training-Free Distribution Adaptation for Diffusion Models via Maximum Mean Discrepancy Guidance},
  author={Matina Mahdizadeh Sani and Nima Jamali and Mohammad Jalali and Farzan Farnia},
  year={2026},
  eprint={2601.08379},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
