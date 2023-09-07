import gc
import os
from abc import ABC, abstractmethod

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


class Preprocessor(ABC):
    @abstractmethod
    def to(self, device: torch.device | str) -> "Preprocessor":
        pass

    @abstractmethod
    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        pass


class CannyPreprocessor(Preprocessor):
    def __init__(self):
        self.model = CannyDetector()

    def to(self, device: torch.device | str) -> Preprocessor:
        return self

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=384, image_resolution=1024)


class LineartPreprocessor(Preprocessor):
    def __init__(self):
        self.model = LineartDetector.from_pretrained("lllyasviel/Annotators")

    def to(self, device: torch.device | str) -> Preprocessor:
        return self.model.to(device)

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=384, image_resolution=1024)


class MidasPreprocessor(Preprocessor):
    def __init__(self):
        self.model = MidasDetector.from_pretrained(
            "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large"
        )

    def to(self, device: torch.device | str) -> Preprocessor:
        return self.model.to(device)

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=512, image_resolution=1024)


class PidiNetPreprocessor(Preprocessor):
    def __init__(self):
        self.model = PidiNetDetector.from_pretrained("lllyasviel/Annotators")

    def to(self, device: torch.device | str) -> Preprocessor:
        return self.model.to(device)

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=512, image_resolution=1024, apply_filter=True)


class RecolorPreprocessor(Preprocessor):
    def to(self, device: torch.device | str) -> Preprocessor:
        return self

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return image.convert("L").convert("RGB")


class ZoePreprocessor(Preprocessor):
    def __init__(self):
        self.model = ZoeDetector.from_pretrained(
            "valhalla/t2iadapter-aux-models", filename="zoed_nk.pth", model_type="zoedepth_nk"
        )

    def to(self, device: torch.device | str) -> Preprocessor:
        return self.model.to(device)

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, gamma_corrected=True, image_resolution=1024)


PRELOAD_PREPROCESSORS_IN_GPU_MEMORY = os.getenv("PRELOAD_PREPROCESSORS_IN_GPU_MEMORY", "1") == "1"
PRELOAD_PREPROCESSORS_IN_CPU_MEMORY = os.getenv("PRELOAD_PREPROCESSORS_IN_CPU_MEMORY", "0") == "1"
if PRELOAD_PREPROCESSORS_IN_GPU_MEMORY:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessors_gpu: dict[str, Preprocessor] = {
        "TencentARC/t2i-adapter-canny-sdxl-1.0": CannyPreprocessor().to(device),
        "TencentARC/t2i-adapter-sketch-sdxl-1.0": PidiNetPreprocessor().to(device),
        "TencentARC/t2i-adapter-lineart-sdxl-1.0": LineartPreprocessor().to(device),
        "TencentARC/t2i-adapter-depth-midas-sdxl-1.0": MidasPreprocessor().to(device),
        "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0": ZoePreprocessor().to(device),
        "TencentARC/t2i-adapter-recolor-sdxl-1.0": RecolorPreprocessor().to(device),
    }

    def get_preprocessor(adapter_name: str) -> Preprocessor:
        return preprocessors_gpu[adapter_name]

elif PRELOAD_PREPROCESSORS_IN_CPU_MEMORY:
    preprocessors_cpu: dict[str, Preprocessor] = {
        "TencentARC/t2i-adapter-canny-sdxl-1.0": CannyPreprocessor(),
        "TencentARC/t2i-adapter-sketch-sdxl-1.0": PidiNetPreprocessor(),
        "TencentARC/t2i-adapter-lineart-sdxl-1.0": LineartPreprocessor(),
        "TencentARC/t2i-adapter-depth-midas-sdxl-1.0": MidasPreprocessor(),
        "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0": ZoePreprocessor(),
        "TencentARC/t2i-adapter-recolor-sdxl-1.0": RecolorPreprocessor(),
    }

    def get_preprocessor(adapter_name: str) -> Preprocessor:
        return preprocessors_cpu[adapter_name]

else:

    def get_preprocessor(adapter_name: str) -> Preprocessor:
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

    def download_all_preprocessors():
        for adapter_name in ADAPTER_NAMES:
            get_preprocessor(adapter_name)
        gc.collect()

    download_all_preprocessors()


def download_all_adapters():
    for adapter_name in ADAPTER_NAMES:
        T2IAdapter.from_pretrained(
            adapter_name,
            torch_dtype=torch.float16,
            varient="fp16",
        )
    gc.collect()


download_all_adapters()


class Model:
    MAX_NUM_INFERENCE_STEPS = 50

    def __init__(self, adapter_name: str):
        if adapter_name not in ADAPTER_NAMES:
            raise ValueError(f"Adapter name must be one of {ADAPTER_NAMES}")

        self.preprocessor_name = adapter_name
        self.adapter_name = adapter_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.preprocessor = get_preprocessor(adapter_name).to(self.device)

            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            adapter = T2IAdapter.from_pretrained(
                adapter_name,
                torch_dtype=torch.float16,
                varient="fp16",
            ).to(self.device)
            self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
                model_id,
                vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
                adapter=adapter,
                scheduler=EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(self.device)
            self.pipe.enable_xformers_memory_efficient_attention()
        else:
            self.preprocessor = None  # type: ignore
            self.pipe = None

    def change_preprocessor(self, adapter_name: str) -> None:
        if adapter_name not in ADAPTER_NAMES:
            raise ValueError(f"Adapter name must be one of {ADAPTER_NAMES}")
        if adapter_name == self.preprocessor_name:
            return

        if PRELOAD_PREPROCESSORS_IN_GPU_MEMORY:
            pass
        elif PRELOAD_PREPROCESSORS_IN_CPU_MEMORY:
            self.preprocessor.to("cpu")
        else:
            del self.preprocessor
        self.preprocessor = get_preprocessor(adapter_name).to(self.device)
        self.preprocessor_name = adapter_name
        gc.collect()
        torch.cuda.empty_cache()

    def change_adapter(self, adapter_name: str) -> None:
        if adapter_name not in ADAPTER_NAMES:
            raise ValueError(f"Adapter name must be one of {ADAPTER_NAMES}")
        if adapter_name == self.adapter_name:
            return
        self.pipe.adapter = T2IAdapter.from_pretrained(
            adapter_name,
            torch_dtype=torch.float16,
            varient="fp16",
        ).to(self.device)
        self.adapter_name = adapter_name
        gc.collect()
        torch.cuda.empty_cache()

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
        adapter_name: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        adapter_conditioning_scale: float = 1.0,
        cond_tau: float = 1.0,
        seed: int = 0,
        apply_preprocess: bool = True,
    ) -> list[PIL.Image.Image]:
        if not torch.cuda.is_available():
            raise RuntimeError("This demo does not work on CPU.")
        if num_inference_steps > self.MAX_NUM_INFERENCE_STEPS:
            raise ValueError(f"Number of steps must be less than {self.MAX_NUM_INFERENCE_STEPS}")

        # Resize image to avoid OOM
        image = self.resize_image(image)

        self.change_preprocessor(adapter_name)
        self.change_adapter(adapter_name)

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
