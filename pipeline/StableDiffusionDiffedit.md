# DiffEdit with StableDiffusion
## Overview
Image editing typically requires providing a mask of the area to be edited. DiffEdit automatically generates the mask for you based on a text query, making it easier overall to create a mask without image editing software. The DiffEdit algorithm works in three steps:

1. the diffusion model denoises an image conditioned on some query text and reference text which produces different noise estimates for different areas of the image; the difference is used to infer a mask to identify which area of the image needs to be changed to match the query text
2. the input image is encoded into latent space with DDIM
3. the latents are decoded with the diffusion model conditioned on the text query, using the mask as a guide such that pixels outside the mask remain the same as in the input image

This guide will show you how to use DiffEdit to edit images without manually creating a mask.


## How to use

```pycon
    import PIL
    import requests
    import mindspore as ms
    from io import BytesIO

    from mindone.diffusers import StableDiffusionDiffEditPipeline


    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")


    img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"

    init_image = download_image(img_url).resize((768, 768))

    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", mindspore_dtype=ms.float16
    )

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)

    prompt = "A bowl of fruits"

    inverted_latents = pipe.invert(image=init_image, prompt=prompt)[0]
```