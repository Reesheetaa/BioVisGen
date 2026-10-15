from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from config import DEVICE


class LLMAgent:

    def __init__(self):

        self.model_name = "microsoft/Phi-3-mini-4k-instruct"

        print("Loading Phi-3...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto"
        # )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        )

        self.model = self.model.to("cpu")


        print("Phi-3 loaded successfully.")

    def generate(
        self,
        system_prompt,
        user_prompt,
        max_new_tokens=256,
        temperature=0.3
    ):

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # inputs = self.tokenizer(
        #     prompt,
        #     return_tensors="pt"
        # ).to(self.model.device)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        inputs = {
            k: v.to("cpu")
            for k, v in inputs.items()
        }

        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=max_new_tokens,
        #     temperature=temperature,
        #     do_sample=True
        # )
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0)
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        # response = self.tokenizer.decode(
        #     outputs[0],
        #     skip_special_tokens=True
        # )

        response = response.split("<|assistant|>")[-1].strip()

        return response