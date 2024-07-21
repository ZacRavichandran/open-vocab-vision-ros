from PIL import Image
import numpy as np

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from enum import Enum

from typing import Optional, Tuple


class SupportedModels(Enum):
    Llava3PhiMini = "xtuner/llava-phi-3-mini-hf"


class VLMWrapper:
    def __init__(
        self,
        model: Optional[str] = SupportedModels.Llava3PhiMini,
        classes="parking lot, sidewalk, road, park, other",
    ) -> None:
        self.model = None

        if model == SupportedModels.Llava3PhiMini.value:
            self.model = LlavaPhi3()
            pass
        else:
            raise ValueError(f"{model} not supported.")

        self.classes = classes.split(",")

        self.scene_prompt = "You are a robot. Where are you currently standing? "

        for i, class_id in enumerate(self.classes):
            self.scene_prompt += f"({i+1}) {class_id}, "

        self.scene_prompt = self.scene_prompt[:-2] + ". "

        self.scene_prompt += "Focus on the nearest parts of the image. Provide a single number. Class id: "

    def open_query(self, prompt: str, image: np.ndarray) -> str:
        img = Image.fromarray(image)
        output = self.model.infer(raw_prompt=prompt, raw_image=img)

        return output

    def classify_scene(self, image: np.ndarray) -> Tuple[str, str]:
        img = Image.fromarray(image)
        output = self.model.infer(raw_prompt=self.scene_prompt, raw_image=img)

        try:
            parsed = int(output.split("Class id:")[-1].strip())
            class_id = self.classes[parsed - 1]
        except:
            class_id = "unkown"

        return output, class_id


class LlavaPhi3:
    def __init__(self) -> None:
        model_id = "xtuner/llava-phi-3-mini-hf"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def infer(self, raw_prompt: str, raw_image: Image) -> str:
        prompt = self.format_prompt(prompt=raw_prompt)
        image_file = "images/sidewalk_2.png"
        # raw_image = Image.open(image_file)

        # raw_image = Image.open(requests.get(image_file, stream=True).raw)

        inputs = self.processor(prompt, raw_image, return_tensors="pt").to(
            0, torch.float16
        )

        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        formatted_output = self.processor.decode(
            output[0][2:], skip_special_tokens=True
        )
        return formatted_output

    def format_prompt(self, prompt: str) -> str:
        return f"<|user|>\n<image>\n{prompt}\n<|assistant|>\n"
