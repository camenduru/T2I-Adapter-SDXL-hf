#!/usr/bin/env python

import os

import gradio as gr
import torch

from app_base import create_demo as create_demo_base
from app_sketch import create_demo as create_demo_sketch
from model import ADAPTER_NAMES, Model, download_all_adapters

DESCRIPTION = "# T2I-Adapter-SDXL"

download_all_adapters()
model = Model(ADAPTER_NAMES[0])

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.Tab(label="Base"):
            create_demo_base(model)
        with gr.Tab(label="Sketch"):
            create_demo_sketch(model)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)
