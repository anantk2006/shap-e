import torch
from IPython import display
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.diffusion.k_diffusion import sample_heun, GaussianToKarrasDenoiser
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type = str, default="a banana ", help="pretty self explanatory dumbass")
parser.add_argument("--second", type = str, default="that has been peeled", help="second prompt for editing")

args = parser.parse_args()
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

batch_size = 4
guidance_scale = 15.0



latents_old = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[args.prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)
model2 = GaussianToKarrasDenoiser(model, diffusion)
def denoiser(x_t, sigma):
            _, denoised = model2.denoise(
                x_t, sigma, clip_denoised=True, model_kwargs = dict(texts=[args.prompt] * batch_size)
            )
            return denoised

latents = sample_heun(denoiser, latents_old, torch.Tensor([10]*32).cuda(), s_churn=10e3)

for x in latents:
      lat = x["x"]
      continue


latents_new = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[args.prompt+args.second] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=32,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
    x_T = lat
)
render_mode = 'nerf' # you can change this to 'stf'
size = 64 # this is the size of the renders; higher values take longer to render.

# cameras = create_pan_cameras(size, device)
# for i, latent in enumerate(latents):
#     images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
#     display(gif_widget(images))

# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh

for i, latent in enumerate(latents_old):
    
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'old_{i}.ply', 'wb') as f:
        t.write_ply(f)

for i, latent in enumerate(lat):
    
    t = decode_latent_mesh(xm, latent).tri_mesh()
    
    with open(f'med_{i}.ply', 'wb') as f:
        t.write_ply(f)

for i, latent in enumerate(latents_new):
    
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'new_{i}.ply', 'wb') as f:
        t.write_ply(f)

# cameras = create_pan_cameras(size, device)
# for i, latent in enumerate(latents_new):
#     images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
#     print(gif_widget(images))
    