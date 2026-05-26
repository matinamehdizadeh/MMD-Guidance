import argparse, os, sys, glob, datetime, yaml
import gc
import pickle
import time

import numpy as np
import PIL
import torch
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import trange

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, F_M_Samples=None, F_M_real=None, return_intermediates=True, guidance_scale=0.0):
    return model.p_sample_loop(shape,
                               return_intermediates=return_intermediates,
                               F_M_Samples=F_M_Samples, F_M_real=F_M_real,
                               guidance_scale=guidance_scale)


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, F_M_Samples=None, F_M_real=None, batch_size=1, vanilla=False,
                               custom_steps=None, eta=1.0, guidance_scale=0.0):
    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow, F_M_Samples = convsample(model, shape, F_M_Samples, F_M_real,
                                                      guidance_scale=guidance_scale)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)
        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log, F_M_Samples

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None,
        F_M_real=None, guidance_scale=0.0):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    F_M_Samples = None

    time_range = [x for x in range(50, 550, 50)]

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
            
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for i in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            if i in time_range:
                end = time.time()
                print(f"Elapsed time for generating {i} samples: {end - tstart:.4f} seconds")
            
            if F_M_Samples is not None and len(F_M_Samples) == 100:
                F_M_Samples = F_M_Samples[-1].unsqueeze(0)

            logs, F_M_Samples = make_convolutional_sample(model, F_M_Samples, F_M_real, batch_size=batch_size,
                                                          vanilla=vanilla, custom_steps=custom_steps,
                                                          eta=eta, guidance_scale=guidance_scale)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        help="MMD Guidance weight",
        default=0.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "--pickle_file",
        type=str,
        nargs="?",
        help="features",
        default=""
    )
    parser.add_argument(
        "--dataset_metadata",
        type=str,
        nargs="?",
        help="user data",
        default=""
    )
    parser.add_argument(
        "-s",
        "--n_references",
        type=int,
        nargs="?",
        help="number of references",
        default=500
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        # print("Checkpoint path:", ckpt)
        # print("Model keys:", pl_sd.keys())
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step



def precompute_F_M_real(
    model, image_dir, device="cuda", output_file='features.pkl', n_references=500
):
    if os.path.exists(output_file):
        print(f"Loading precomputed features from {output_file}...")
        with open(output_file, "rb") as f:
            F_M_real_all = pickle.load(f)
        return F_M_real_all

    image_paths = list(Path(image_dir).rglob('*.png'))[:n_references]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 299x299
        transforms.ToTensor(),  # Convert the image to a tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    print(len(image_paths))
    F_M_real_all = None

    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image)
            img_tensor = (img_tensor - 0.5) * 2
            img_tensor = img_tensor.unsqueeze(0).to(device)  # Add a batch dimension
            encoder_posterior = model.encode_first_stage(img_tensor)
            features = model.get_first_stage_encoding(encoder_posterior).detach()

            features = features * model.scale_factor

            if F_M_real_all is None:
                F_M_real_all = features.detach()
            else:
                F_M_real_all = torch.cat((F_M_real_all, features.detach()), dim=0)  # Concatenate features

            del image, img_tensor, features  # Free memory
            torch.cuda.empty_cache()
            gc.collect()

        except (PIL.UnidentifiedImageError, OSError) as e:
            print(f"Skipping problematic file: {img_path}. Error: {e}")

    print(f"Saving precomputed features to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(F_M_real_all, f)

    return F_M_real_all


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    dataset_metadata = opt.dataset_metadata
    print("Starting ...")

    F_M_real = precompute_F_M_real(model, dataset_metadata, output_file=opt.pickle_file, n_references=opt.n_references)


    run(model, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=numpylogdir, F_M_real=F_M_real,
        guidance_scale=opt.alpha)

    print("done.")
