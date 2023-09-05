from typing import Callable

import PIL.Image
import torch
from controlnet_aux import (
    CannyDetector,
    LineartDetector,
    MidasDetector,
    PidiNetDetector,
    ZoeDetector,
)
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)

ADAPTER_NAMES = [
    "TencentARC/t2i-adapter-canny-sdxl-1.0",
    "TencentARC/t2i-adapter-sketch-sdxl-1.0",
    "TencentARC/t2i-adapter-lineart-sdxl-1.0",
    "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
    "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0",
    "TencentARC/t2i-adapter-recolor-sdxl-1.0",
]


class CannyPreprocessor:
    def __init__(self):
        self.model = CannyDetector()

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=384, image_resolution=1024)


class LineartPreprocessor:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LineartDetector.from_pretrained("lllyasviel/Annotators").to(device)

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=384, image_resolution=1024)


class MidasPreprocessor:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MidasDetector.from_pretrained(
            "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large"
        ).to(device)

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=512, image_resolution=1024)


class PidiNetPreprocessor:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to(device)

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=512, image_resolution=1024, apply_filter=True)


class RecolorPreprocessor:
    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return image.convert("L").convert("RGB")


class ZoePreprocessor:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ZoeDetector.from_pretrained(
            "valhalla/t2iadapter-aux-models", filename="zoed_nk.pth", model_type="zoedepth_nk"
        ).to(device)

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, gamma_corrected=True, image_resolution=1024)


def get_preprocessor(adapter_name: str) -> Callable[[PIL.Image.Image], PIL.Image.Image]:
    if adapter_name == "TencentARC/t2i-adapter-canny-sdxl-1.0":
        return CannyPreprocessor()
    elif adapter_name == "TencentARC/t2i-adapter-sketch-sdxl-1.0":
        return PidiNetPreprocessor()
    elif adapter_name == "TencentARC/t2i-adapter-lineart-sdxl-1.0":
        return LineartPreprocessor()
    elif adapter_name == "TencentARC/t2i-adapter-depth-midas-sdxl-1.0":
        return MidasPreprocessor()
    elif adapter_name == "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0":
        return ZoePreprocessor()
    elif adapter_name == "TencentARC/t2i-adapter-recolor-sdxl-1.0":
        return RecolorPreprocessor()
    else:
        raise ValueError(f"Adapter name must be one of {ADAPTER_NAMES}")


class Model:
    MAX_NUM_INFERENCE_STEPS = 50

    def __init__(self, adapter_name: str):
        if adapter_name not in ADAPTER_NAMES:
            raise ValueError(f"Adapter name must be one of {ADAPTER_NAMES}")

        self.adapter_name = adapter_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.preprocessor = get_preprocessor(adapter_name)

            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            adapter = T2IAdapter.from_pretrained(
                adapter_name,
                torch_dtype=torch.float16,
                varient="fp16",
            ).to(self.device)
            euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
                model_id,
                vae=vae,
                adapter=adapter,
                scheduler=euler_a,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(self.device)
            self.pipe.enable_xformers_memory_efficient_attention()
        else:
            self.pipe = None

    def change_adapter(self, adapter_name: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("This demo does not work on CPU.")
        if adapter_name not in ADAPTER_NAMES:
            raise ValueError(f"Adapter name must be one of {ADAPTER_NAMES}")
        if adapter_name == self.adapter_name:
            return

        self.preprocessor = None  # type: ignore
        torch.cuda.empty_cache()
        self.preprocessor = get_preprocessor(adapter_name)

        self.pipe.adapter = None
        torch.cuda.empty_cache()
        self.pipe.adapter = T2IAdapter.from_pretrained(
            adapter_name,
            torch_dtype=torch.float16,
            varient="fp16",
        ).to(self.device)

    def resize_image(self, image: PIL.Image.Image) -> PIL.Image.Image:
        w, h = image.size
        scale = 1024 / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), PIL.Image.LANCZOS)

    def run(
        self,
        image: PIL.Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        adapter_conditioning_scale: float = 1.0,
        cond_tau: float = 1.0,
        seed: int = 0,
        apply_preprocess: bool = True,
    ) -> list[PIL.Image.Image]:
        if num_inference_steps > self.MAX_NUM_INFERENCE_STEPS:
            raise ValueError(f"Number of steps must be less than {self.MAX_NUM_INFERENCE_STEPS}")

        # Resize image to avoid OOM
        image = self.resize_image(image)

        if apply_preprocess:
            image = self.preprocessor(image)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            adapter_conditioning_scale=adapter_conditioning_scale,
            cond_tau=cond_tau,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images[0]
        return [image, out]
