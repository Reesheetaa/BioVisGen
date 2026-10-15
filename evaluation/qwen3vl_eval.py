import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from prompts import SYSTEM_PROMPT, build_user_prompt


class Qwen3VLEvaluator:
    def __init__(self, hf_id: str, device: str = "cuda:0"):
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            hf_id, dtype="auto", device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(hf_id)

    @torch.inference_mode()
    def evaluate(self, description: str, ref_image_path: str, gen_image_path: str,
                 max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        ref_img = Image.open(ref_image_path).convert("RGB")
        gen_img = Image.open(gen_image_path).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reference image (contextual only):"},
                    {"type": "image", "image": ref_img},
                    {"type": "text", "text": "Generated image to evaluate:"},
                    {"type": "image", "image": gen_img},
                    {"type": "text", "text": build_user_prompt(description)},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text
