import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# Get the device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Set the device
DEVICE = get_device()
print("Using device:", DEVICE)

def create_pipeline():
    """
    Load the SDXL base model
    """
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    # Force VAE to be float32 to prevent black images on MPS
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float32,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        vae=vae,
    )

    # Move pipeline to the selected device
    pipe.to(DEVICE)
    pipe.unet.to(dtype=torch.float32)
    pipe.vae.to(dtype=torch.float32)
    pipe.text_encoder.to(dtype=torch.float32)
    pipe.text_encoder_2.to(dtype=torch.float32)

    # Reduce VRAM / RAM usage
    if DEVICE != "cpu":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    pipe.set_progress_bar_config(disable=False)
    return pipe

# Create pipeline
pipe = create_pipeline()

# Create txt2img function
def txt2img(
    prompt: str,
    negative_prompt: str | None = None,
    steps: int = 20,
    guidance_scale: float = 7.0,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
):
    # Create generator
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Create image
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    image = result.images[0]
    return image
