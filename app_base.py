#!/usr/bin/env python

import os

import gradio as gr
import PIL.Image
from diffusers.utils import load_image

from model import ADAPTER_NAMES, Model
from utils import (
    DEFAULT_STYLE_NAME,
    MAX_SEED,
    STYLE_NAMES,
    apply_style,
    randomize_seed_fn,
)

CACHE_EXAMPLES = os.environ.get("CACHE_EXAMPLES") == "1"


def create_demo(model: Model) -> gr.Blocks:
    def run(
        image: PIL.Image.Image,
        prompt: str,
        negative_prompt: str,
        adapter_name: str,
        style_name: str = DEFAULT_STYLE_NAME,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        adapter_conditioning_scale: float = 1.0,
        adapter_conditioning_factor: float = 1.0,
        seed: int = 0,
        apply_preprocess: bool = True,
        progress=gr.Progress(track_tqdm=True),
    ) -> list[PIL.Image.Image]:
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        return model.run(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            adapter_name=adapter_name,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=adapter_conditioning_scale,
            adapter_conditioning_factor=adapter_conditioning_factor,
            seed=seed,
            apply_preprocess=apply_preprocess,
        )

    def process_example(
        image_url: str,
        prompt: str,
        adapter_name: str,
        style_name: str,
        guidance_scale: float,
        adapter_conditioning_scale: float,
        seed: int,
        apply_preprocess: bool,
    ) -> list[PIL.Image.Image]:
        image = load_image(image_url)
        return run(
            image=image,
            prompt=prompt,
            negative_prompt="",
            adapter_name=adapter_name,
            style_name=style_name,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=adapter_conditioning_scale,
            seed=seed,
            apply_preprocess=apply_preprocess,
        )

    examples = [
        [
            "assets/org_canny.jpg",
            "Mystical fairy in real, magic, 4k picture, high quality",
            "canny",
            "Photographic",
            5.0,
            1.0,
            0,
            True,
        ],
        [
            "assets/org_sketch.png",
            "a robot, mount fuji in the background, 4k photo, highly detailed",
            "sketch",
            "Photographic",
            5.0,
            1.0,
            0,
            True,
        ],
        [
            "assets/org_lin.jpg",
            "Ice dragon roar, 4k photo",
            "lineart",
            "Cinematic",
            7.5,
            0.8,
            0,
            True,
        ],
        [
            "assets/org_mid.jpg",
            "A photo of a room, 4k photo, highly detailed",
            "depth-midas",
            "Photographic",
            5.0,
            1.0,
            0,
            True,
        ],
        [
            "assets/org_zoe.jpg",
            "A photo of a orchid, 4k photo, highly detailed",
            "depth-zoe",
            "Photographic",
            5.0,
            1.0,
            0,
            True,
        ],
        [
            "assets/people.jpg",
            "A couple, 4k photo, highly detailed",
            "openpose",
            "Photographic",
            5.0,
            1.0,
            0,
            True,
        ],
        [
            "assets/depth-midas-image.png",
            "stormtrooper lecture, 4k photo, highly detailed",
            "depth-midas",
            "Photographic",
            5.0,
            1.0,
            0,
            False,
        ],
        [
            "assets/openpose-image.png",
            "spiderman, 4k photo, highly detailed",
            "openpose",
            "Photographic",
            5.0,
            1.0,
            0,
            False,
        ],
    ]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    image = gr.Image(label="Input image", type="pil", height=600)
                    prompt = gr.Textbox(label="Prompt")
                    adapter_name = gr.Dropdown(label="Adapter name", choices=ADAPTER_NAMES, value=ADAPTER_NAMES[0])
                    run_button = gr.Button("Run")
                with gr.Accordion("Advanced options", open=False):
                    apply_preprocess = gr.Checkbox(label="Apply preprocess", value=True)
                    style = gr.Dropdown(label="Style", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
                    negative_prompt = gr.Textbox(label="Negative prompt")
                    num_inference_steps = gr.Slider(
                        label="Number of steps",
                        minimum=1,
                        maximum=Model.MAX_NUM_INFERENCE_STEPS,
                        step=1,
                        value=25,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=30.0,
                        step=0.1,
                        value=5.0,
                    )
                    adapter_conditioning_scale = gr.Slider(
                        label="Adapter conditioning scale",
                        minimum=0.5,
                        maximum=1,
                        step=0.1,
                        value=1.0,
                    )
                    adapter_conditioning_factor = gr.Slider(
                        label="Adapter conditioning factor",
                        info="Fraction of timesteps for which adapter should be applied",
                        minimum=0.5,
                        maximum=1.0,
                        step=0.1,
                        value=1.0,
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
                result = gr.Gallery(label="Result", columns=2, height=600, object_fit="scale-down", show_label=False)

        gr.Examples(
            examples=examples,
            inputs=[
                image,
                prompt,
                adapter_name,
                style,
                guidance_scale,
                adapter_conditioning_scale,
                seed,
                apply_preprocess,
            ],
            outputs=result,
            fn=process_example,
            cache_examples=CACHE_EXAMPLES,
        )

        inputs = [
            image,
            prompt,
            negative_prompt,
            adapter_name,
            style,
            num_inference_steps,
            guidance_scale,
            adapter_conditioning_scale,
            adapter_conditioning_factor,
            seed,
            apply_preprocess,
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
            api_name="run",
        )

    return demo


if __name__ == "__main__":
    model = Model(ADAPTER_NAMES[0])
    demo = create_demo(model)
    demo.queue(max_size=20).launch()
