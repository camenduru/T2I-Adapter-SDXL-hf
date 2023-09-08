#!/usr/bin/env python

import gradio as gr
import PIL.Image

from model import ADAPTER_NAMES, Model
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
        adapter_name: str,
        style_name: str = DEFAULT_STYLE_NAME,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        adapter_conditioning_scale: float = 1.0,
        cond_tau: float = 1.0,
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
            cond_tau=cond_tau,
            seed=seed,
            apply_preprocess=apply_preprocess,
        )

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    image = gr.Image(label="Input image", type="pil", height=600)
                    prompt = gr.Textbox(label="Prompt")
                    adapter_name = gr.Dropdown(label="Adapter", choices=ADAPTER_NAMES, value=ADAPTER_NAMES[0])
                    run_button = gr.Button("Run")
                with gr.Accordion("Advanced options", open=False):
                    apply_preprocess = gr.Checkbox(label="Apply preprocess", value=True)
                    negative_prompt = gr.Textbox(label="Negative prompt")
                    style = gr.Dropdown(label="Style", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
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
            style,
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

    return demo


if __name__ == "__main__":
    model = Model(ADAPTER_NAMES[0])
    demo = create_demo(model)
    demo.queue(max_size=20).launch()
