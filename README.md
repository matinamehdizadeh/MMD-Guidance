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

Use the same checkpoint and model layout expected by latent-diffusion.

### Run Unconditional Sampling

Example command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py \
  -r {your_model.ckpt} \
  -l {log_output} \
  --pickle_file {precomputed.pkl} \
  --dataset_metadata {path_to_reference_images} \
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
The connected functions are:
  -precompute_F_M_real

```


For a new diffusion model, this step should be adapted to the model's representation space. If your model samples directly in pixel space, `F_M_real` can be image-space tensors. If your model samples in latent space, encode the reference images using the same encoder/scale used by the sampler.

### MMD Guidance Hook

The MMD guidance logic is in:

```text
Unconditional_MMD/latent-diffusion/ldm/models/diffusion/ddpm.py
The connected functions are:

  _flatten_feats
  compute_kid_grad_poly3
  get_F_M
  cond_fn
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
    n_references=n,
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



## Prompt-Aware MMD

Prompt-aware MMD extends the same idea to conditional generation by making the MMD objective aware of both generated/reference samples and their associated prompts or conditioning information.

The prompt-aware implementation belongs in:

```text
Prompt_Aware_MMD
```



## License

For the unconditional MMD guidance, this repository builds on the CompVis latent-diffusion codebase. The latent-diffusion components retain their original license, which is included in:

```text
Unconditional_MMD/latent-diffusion/LICENSE
```

The MMD Guidance modifications and additional code in this repository should be used under the top-level repository license. If you use this code.

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
