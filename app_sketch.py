#!/usr/bin/env python

import gradio as gr
import PIL.Image
import torch
import torchvision.transforms.functional as TF

from model import Model
from utils import (
    DEFAULT_STYLE_NAME,
    MAX_SEED,
    STYLE_NAMES,
    apply_style,
    randomize_seed_fn,
)


def create_demo(model: Model) -> gr.Blocks:
    def run(
        image: PIL.Image.Image,
        prompt: str,
        negative_prompt: str,
        style_name: str = DEFAULT_STYLE_NAME,
        num_steps: int = 25,
        guidance_scale: float = 5,
        adapter_conditioning_scale: float = 0.8,
        adapter_conditioning_factor: float = 0.8,
        seed: int = 0,
        progress=gr.Progress(track_tqdm=True),
    ) -> PIL.Image.Image:
        image = image.convert("RGB")
        image = TF.to_tensor(image) > 0.5
        image = TF.to_pil_image(image.to(torch.float32))

        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        return model.run(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            adapter_name="sketch",
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=adapter_conditioning_scale,
            adapter_conditioning_factor=adapter_conditioning_factor,
            seed=seed,
            apply_preprocess=False,
        )[1]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    image = gr.Image(
                        source="canvas",
                        tool="sketch",
                        type="pil",
                        image_mode="L",
                        invert_colors=True,
                        shape=(1024, 1024),
                        brush_radius=4,
                        height=600,
                    )
                    with gr.Row():
                        adapter_name = gr.Dropdown(label="Adapter name", choices=ADAPTER_NAMES, value=ADAPTER_NAMES[0])
                        style = gr.Dropdown(label="Style", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
                    run_button = gr.Button("Run")
                with gr.Accordion("Advanced options", open=False):
                    negative_prompt = gr.Textbox(label="Negative prompt")
                    num_steps = gr.Slider(
                        label="Number of steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=25,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=10.0,
                        step=0.1,
                        value=5,
                    )
                    adapter_conditioning_scale = gr.Slider(
                        label="Adapter conditioning scale",
                        minimum=0.5,
                        maximum=1,
                        step=0.1,
                        value=0.8,
                    )
                    adapter_conditioning_factor = gr.Slider(
                        label="Adapter conditioning factor",
                        info="Fraction of timesteps for which adapter should be applied",
                        minimum=0.5,
                        maximum=1,
                        step=0.1,
                        value=0.8,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Column():
                result = gr.Image(label="Result", height=600)

        inputs = [
            image,
            prompt,
            negative_prompt,
            style,
            num_steps,
            guidance_scale,
            adapter_conditioning_scale,
            adapter_conditioning_factor,
            seed,
        ]
        prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=run,
            inputs=inputs,
            outputs=result,
            api_name=False,
        )
        negative_prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=run,
            inputs=inputs,
            outputs=result,
            api_name=False,
        )
        run_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=run,
            inputs=inputs,
            outputs=result,
            api_name=False,
        )

    return demo


if __name__ == "__main__":
    model = Model("sketch")
    demo = create_demo(model)
    demo.queue(max_size=20).launch()
