# UnCLIPPipeline Text-to-image
## Overview

[Hierarchical Text-Conditional Image Generation with CLIP Latents](https://huggingface.co/papers/2204.06125) is by Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen. The unCLIP model in 🤗 Diffusers comes from kakaobrain's [karlo](https://github.com/kakaobrain/karlo).

The abstract from the paper is following:

*Contrastive models like CLIP have been shown to learn robust representations of images that capture both semantics and style. To leverage these representations for image generation, we propose a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding. We show that explicitly generating image representations improves image diversity with minimal loss in photorealism and caption similarity. Our decoders conditioned on image representations can also produce variations of an image that preserve both its semantics and style, while varying the non-essential details absent from the image representation. Moreover, the joint embedding space of CLIP enables language-guided image manipulations in a zero-shot fashion. We use diffusion models for the decoder and experiment with both autoregressive and diffusion models for the prior, finding that the latter are computationally more efficient and produce higher-quality samples.*

## How to use

```pycon
import mindspore as ms
from mindone.diffusers import UnCLIPPipeline

pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", mindspore_type=ms.float16)

prompt = "a high-resolution photograph of a big red frog on a green leaf"
image = pipe([prompt])[0][0]
image.save("./frog.png")

```
![alt text](4C239D89-2E8C-43E4-9794-959BCCE80114.png)