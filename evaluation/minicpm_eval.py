import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from prompts import SYSTEM_PROMPT, build_user_prompt


class MiniCPMEvaluator:
    def __init__(self, hf_id: str, device: str = "cuda:0"):
        self.device = device
        self.model = AutoModel.from_pretrained(
            hf_id,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        ).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    @torch.inference_mode()
    def evaluate(self, description: str, ref_image_path: str, gen_image_path: str,
                 max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        ref_img = Image.open(ref_image_path).convert("RGB")
        gen_img = Image.open(gen_image_path).convert("RGB")

        user_text = (
            "Reference image (contextual only), followed by the generated image "
            "to evaluate.\n\n" + build_user_prompt(description)
        )

        msgs = [
            {"role": "user", "content": [ref_img, gen_img, user_text]},
        ]

        sampling = temperature is not None and temperature > 0
        text = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            system_prompt=SYSTEM_PROMPT,
            sampling=sampling,
            temperature=temperature if sampling else 1.0,
            max_new_tokens=max_new_tokens,
        )
        return text
