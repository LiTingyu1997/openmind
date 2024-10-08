# StableDiffusion
## Overview
The Stable Diffusion upscaler diffusion model was created by the researchers and engineers from CompVis, Stability AI, and LAION. It is used to enhance the resolution of input images by a factor of 4.

Make sure to check out the Stable Diffusion Tips section to learn how to explore the tradeoff between scheduler speed and quality, and how to reuse pipeline components efficiently!

If you're interested in using one of the official checkpoints for a task, explore the CompVis, Runway, and Stability AI Hub organizations!


## How to use

```pycon
    import requests
    from PIL import Image
    from io import BytesIO
    from mindone.diffusers import StableDiffusionUpscalePipeline
    import mindspore as ms

    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, revision="fp16", mindspore_dtype=ms.float16
    )

    # let's download an  image
    url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((128, 128))
    prompt = "a white cat"

    upscaled_image = pipeline(prompt=prompt, image=low_res_img)[0][0]
    upscaled_image.save("upsampled_cat.png")
```