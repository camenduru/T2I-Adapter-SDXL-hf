#!/usr/bin/env python

import gradio as gr
import PIL.Image

from model import ADAPTER_NAMES, Model
from utils import MAX_SEED, randomize_seed_fn

style_list = [
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
default_style_name = "Photographic"
default_style = styles[default_style_name]
style_names = list(styles.keys())


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, default_style)
    return p.replace("{prompt}", positive), n + negative


def create_demo(model: Model) -> gr.Blocks:
    def run(
        image: PIL.Image.Image,
        prompt: str,
        negative_prompt: str,
        adapter_name: str,
        style_name: str = default_style_name,
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
                    negative_prompt = gr.Textbox(
                        label="Negative prompt",
                        value="",
                    )
                    style = gr.Dropdown(choices=style_names, value=default_style_name, label="Style")
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
