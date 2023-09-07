#!/usr/bin/env python

import gradio as gr
import PIL.Image
import torch
import torchvision.transforms.functional as TF

from model import Model
from utils import MAX_SEED, randomize_seed_fn

SKETCH_ADAPTER_NAME = "TencentARC/t2i-adapter-sketch-sdxl-1.0"

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
        style_name: str = default_style_name,
        num_steps: int = 25,
        guidance_scale: float = 5,
        adapter_conditioning_scale: float = 0.8,
        cond_tau: float = 0.8,
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
            adapter_name=SKETCH_ADAPTER_NAME,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=adapter_conditioning_scale,
            cond_tau=cond_tau,
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
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button("Run")
                with gr.Accordion("Advanced options", open=False):
                    style = gr.Dropdown(choices=style_names, value=default_style_name, label="Style")
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
                        label="Adapter Conditioning Scale",
                        minimum=0.5,
                        maximum=1,
                        step=0.1,
                        value=0.8,
                    )
                    cond_tau = gr.Slider(
                        label="Fraction of timesteps for which adapter should be applied",
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
            cond_tau,
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
    model = Model(SKETCH_ADAPTER_NAME)
    demo = create_demo(model)
    demo.queue(max_size=20).launch()
