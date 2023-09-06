#!/usr/bin/env python

import os
import random

import gradio as gr
import numpy as np
import PIL.Image
import torch

from model import ADAPTER_NAMES, Model

DESCRIPTION = "# T2I-Adapter-SDXL"

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


model = Model(ADAPTER_NAMES[0])


def run(
    image: PIL.Image.Image,
    prompt: str,
    negative_prompt: str,
    adapter_name: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    adapter_conditioning_scale: float = 1.0,
    cond_tau: float = 1.0,
    seed: int = 0,
    apply_preprocess: bool = True,
    progress=gr.Progress(track_tqdm=True),
) -> list[PIL.Image.Image]:
    return model.run(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        adapter_name=adapter_name,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        adapter_conditioning_scale=adapter_conditioning_scale,
        cond_tau=cond_tau,
        seed=seed,
        apply_preprocess=apply_preprocess,
    )


with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )

    with gr.Row():
        with gr.Column():
            with gr.Group():
                image = gr.Image(label="Input image", type="pil", height=600)
                prompt = gr.Textbox(label="Prompt")
                adapter_name = gr.Dropdown(label="Adapter", choices=ADAPTER_NAMES, value=ADAPTER_NAMES[0])
                run_button = gr.Button("Run")
            with gr.Accordion("Advanced options", open=False):
                apply_preprocess = gr.Checkbox(label="Apply preprocess", value=True)
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    value="anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
                )
                num_inference_steps = gr.Slider(
                    label="Number of steps",
                    minimum=1,
                    maximum=Model.MAX_NUM_INFERENCE_STEPS,
                    step=1,
                    value=30,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=30.0,
                    step=0.1,
                    value=5.0,
                )
                adapter_conditioning_scale = gr.Slider(
                    label="Adapter Conditioning Scale",
                    minimum=0.5,
                    maximum=1,
                    step=0.1,
                    value=1.0,
                )
                cond_tau = gr.Slider(
                    label="Fraction of timesteps for which adapter should be applied",
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

    inputs = [
        image,
        prompt,
        negative_prompt,
        adapter_name,
        num_inference_steps,
        guidance_scale,
        adapter_conditioning_scale,
        cond_tau,
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

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
