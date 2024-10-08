# DiT
## Overview
[Scalable Diffusion Models with Transformers](https://huggingface.co/papers/2212.09748) (DiT) is by William Peebles and Saining Xie.

The abstract from the paper is:

*We explore a new class of diffusion models based on the transformer architecture. We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches. We analyze the scalability of our Diffusion Transformers (DiTs) through the lens of forward pass complexity as measured by Gflops. We find that DiTs with higher Gflops -- through increased transformer depth/width or increased number of input tokens -- consistently have lower FID. In addition to possessing good scalability properties, our largest DiT-XL/2 models outperform all prior diffusion models on the class-conditional ImageNet 512x512 and 256x256 benchmarks, achieving a state-of-the-art FID of 2.27 on the latter.*

The original codebase can be found at [facebookresearch/dit](https://github.com/facebookresearch/dit).

<Tip>


## How to use

```pycon
    from mindone.diffusers import DiTPipeline, DPMSolverMultistepScheduler
    import mindspore as ms

    import numpy as np

    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", mindspore_dtype=ms.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # pick words from Imagenet class labels
    pipe.labels  # to print all available words

    # pick words that exist in ImageNet
    words = ["white shark", "umbrella"]

    class_ids = pipe.get_label_ids(words)

    generator = np.random.default_rng(33)
    output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

    image = output[0][0]  # label 'white shark'
```