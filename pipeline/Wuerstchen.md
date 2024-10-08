# Würstchen Text-to-image
## Overview
Würstchen is a diffusion model, whose text-conditional model works in a highly compressed latent space of images. Why is this important? Compressing data can reduce computational costs for both training and inference by magnitudes. Training on 1024x1024 images is way more expensive than training on 32x32. Usually, other works make use of a relatively small compression, in the range of 4x - 8x spatial compression. Würstchen takes this to an extreme. Through its novel design, we achieve a 42x spatial compression. This was unseen before because common methods fail to faithfully reconstruct detailed images after 16x spatial compression. Würstchen employs a two-stage compression, what we call Stage A and Stage B. Stage A is a VQGAN, and Stage B is a Diffusion Autoencoder (more details can be found in the paper). A third model, Stage C, is learned in that highly compressed latent space. This training requires fractions of the compute used for current top-performing models, while also allowing cheaper and faster inference.


## How to use

```pycon
import mindspore as ms
from mindone.diffusers import WuerstchenPriorPipeline, WuerstchenDecoderPipeline

prior_pipe = WuerstchenPriorPipeline.from_pretrained(
    "warp-ai/wuerstchen-prior", mindspore_dtype=ms.float16
)
gen_pipe = WuerstchenDecoderPipeline.from_pretrained("warp-ai/wuerstchen", mindspore_dtype=ms.float16)

prompt = "an image of a shiba inu, donning a spacesuit and helmet"
prior_output = pipe(prompt)
images = gen_pipe(prior_output[0], prompt=prompt)
```
![alt text](<image (27).jpeg>)