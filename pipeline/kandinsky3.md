# Kandinsky3
## Overview
Kandinsky 3 is created by [Vladimir Arkhipkin](https://github.com/oriBetelgeuse),[Anastasia Maltseva](https://github.com/NastyaMittseva),[Igor Pavlov](https://github.com/boomb0om),[Andrei Filatov](https://github.com/anvilarth),[Arseniy Shakhmatov](https://github.com/cene555),[Andrey Kuznetsov](https://github.com/kuznetsoffandrey),[Denis Dimitrov](https://github.com/denndimitrov), [Zein Shaheen](https://github.com/zeinsh)

The description from it's Github page:

*Kandinsky 3.0 is an open-source text-to-image diffusion model built upon the Kandinsky2-x model family. In comparison to its predecessors, enhancements have been made to the text understanding and visual quality of the model, achieved by increasing the size of the text encoder and Diffusion U-Net models, respectively.*

Its architecture includes 3 main components:
1. [FLAN-UL2](https://huggingface.co/google/flan-ul2), which is an encoder decoder model based on the T5 architecture.
2. New U-Net architecture featuring BigGAN-deep blocks doubles depth while maintaining the same number of parameters.
3. Sber-MoVQGAN is a decoder proven to have superior results in image restoration.



The original codebase can be found at [ai-forever/Kandinsky-3](https://github.com/ai-forever/Kandinsky-3).

<Tip>

Check out the [Kandinsky Community](https://huggingface.co/kandinsky-community) organization on the Hub for the official model checkpoints for tasks like text-to-image, image-to-image, and inpainting.

</Tip>

## How to use

```pycon
    from diffusers import Kandinsky3Pipeline
    import mindspore as ms
    import numpy as np

    pipe = Kandinsky3Pipeline.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", mindspore_dtype=ms.float16)

    prompt = "A photograph of the inside of a subway train. There are raccoons sitting on the seats.
        One of them is reading a newspaper. The window shows the city in the background."

    generator = np.random.Generator(np.random.PCG64(43))
    image = pipe(prompt, num_inference_steps=25, generator=generator)[0][0]
```