# import torch
# from PIL import Image
# from transformers import DeepseekVLHybridForConditionalGeneration, AutoProcessor

# from prompts import SYSTEM_PROMPT, build_user_prompt


# class DeepseekVLEvaluator:
#     def __init__(self, hf_id: str, device: str = "cuda:0"):
#         self.device = device
#         self.model = DeepseekVLHybridForConditionalGeneration.from_pretrained(
#             hf_id, torch_dtype=torch.float16, device_map=device, attn_implementation="sdpa"
#         ).eval()
#         self.processor = AutoProcessor.from_pretrained(hf_id)

#     @torch.inference_mode()
#     def evaluate(self, description: str, ref_image_path: str, gen_image_path: str,
#                  max_new_tokens: int = 512, temperature: float = 0.2) -> str:
#         ref_img = Image.open(ref_image_path).convert("RGB")
#         gen_img = Image.open(gen_image_path).convert("RGB")

#         messages = [
#             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "Reference image (contextual only):"},
#                     {"type": "image", "image": ref_img},
#                     {"type": "text", "text": "Generated image to evaluate:"},
#                     {"type": "image", "image": gen_img},
#                     {"type": "text", "text": build_user_prompt(description)},
#                 ],
#             },
#         ]

#         inputs = self.processor.apply_chat_template(
#             messages, add_generation_prompt=True, padding=True, truncation=True,
#             tokenize=True, return_dict=True, return_tensors="pt",
#         ).to(self.model.device)

#         gen_kwargs = {"max_new_tokens": max_new_tokens}
#         if temperature and temperature > 0:
#             gen_kwargs.update(do_sample=True, temperature=temperature)
#         else:
#             gen_kwargs.update(do_sample=False)

#         output_ids = self.model.generate(**inputs, **gen_kwargs)
#         trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
#         text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
#         return text
import torch
from PIL import Image
from transformers import DeepseekVLHybridForConditionalGeneration, AutoProcessor

from prompts import SYSTEM_PROMPT, build_user_prompt


class DeepseekVLEvaluator:
    def __init__(self, hf_id: str, device: str = "cuda:0"):
        self.device = device
        self.model = DeepseekVLHybridForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=torch.float16, device_map=device, attn_implementation="sdpa"
        ).eval()
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
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True, "truncation": True},
        )

        # Move every tensor to the model's device; additionally cast floating-point
        # tensors (pixel_values etc.) to the model's dtype (float16). input_ids /
        # attention_mask are integer tensors and must NOT be cast to float16, so
        # we only convert tensors that are already floating point.
        moved = {}
        for k, v in inputs.items():
            if not torch.is_tensor(v):
                moved[k] = v
                continue
            v = v.to(self.model.device)
            if torch.is_floating_point(v):
                v = v.to(self.model.dtype)
            moved[k] = v
        inputs = moved

        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return text