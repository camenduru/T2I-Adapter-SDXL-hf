#!/usr/bin/env python

import os

import gradio as gr
import torch

from app_base import create_demo as create_demo_base
from app_sketch import create_demo as create_demo_sketch
from model import ADAPTER_NAMES, Model, download_all_adapters

DESCRIPTION = "# T2I-Adapter-SDXL"

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"


download_all_adapters()
model = Model(ADAPTER_NAMES[0])


with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Tabs():
        with gr.Tab(label="Base"):
            create_demo_base(model)
        with gr.Tab(label="Sketch"):
            create_demo_sketch(model)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
