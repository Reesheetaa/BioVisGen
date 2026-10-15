import torch
import open_clip

from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq
)

from config import (
    DEVICE,
    VLM_MODEL
)


class Critic:

    def __init__(self):

        print("Loading CLIP critic...")

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai"
            )
        )

        self.model = self.model.to(DEVICE)

        self.tokenizer = open_clip.get_tokenizer(
            "ViT-B-32"
        )

        # =====================================
        # VLM CRITIC
        # Lazy loaded only when needed
        # =====================================

        self.vlm_model = None
        self.vlm_processor = None

        print("CLIP critic loaded successfully.")

    # =====================================================
    # CLIP SCORE
    # =====================================================

    def score(
        self,
        image,
        prompt
    ):

        image_input = (
            self.preprocess(image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        text_input = (
            self.tokenizer([prompt])
            .to(DEVICE)
        )

        with torch.no_grad():

            image_features = (
                self.model.encode_image(
                    image_input
                )
            )

            text_features = (
                self.model.encode_text(
                    text_input
                )
            )

        image_features /= image_features.norm(
            dim=-1,
            keepdim=True
        )

        text_features /= text_features.norm(
            dim=-1,
            keepdim=True
        )

        similarity = cosine_similarity(
            image_features.cpu().numpy(),
            text_features.cpu().numpy()
        )[0][0]

        normalized_score = (
            similarity + 1
        ) / 2

        return float(normalized_score)

    # =====================================================
    # LOAD VLM ONLY IF REQUIRED
    # =====================================================

    def load_vlm(self):

        if self.vlm_model is not None:
            return

        print(
            "\nLoading Qwen2.5-VL critic..."
        )

        self.vlm_processor = (
            AutoProcessor.from_pretrained(
                VLM_MODEL
            )
        )

        # self.vlm_model = (
        #     AutoModelForVision2Seq
        #     .from_pretrained(
        #         VLM_MODEL,
        #         torch_dtype=torch.bfloat16,
        #         device_map="auto"
        #     )
        # )
        
        self.vlm_model = (
            AutoModelForVision2Seq
            .from_pretrained(
                VLM_MODEL,
                torch_dtype=torch.float32
            )
        )

        self.vlm_model = self.vlm_model.to("cpu")

        self.vlm_model.eval()

        print(
            "Qwen2.5-VL critic loaded."
        )

    # =====================================================
    # GENERATE REFINEMENT INSTRUCTIONS
    # =====================================================

    def generate_refinement(
        self,
        image,
        prompt
    ):

        self.load_vlm()

        system_prompt = """
You are an expert evaluator of educational biology illustrations.

Your task is to evaluate ONLY the biological content and structure of the image.

IMPORTANT:

Ignore all text in the image.

Do not evaluate:
- labels
- spelling
- typography
- font quality
- text placement
- annotation appearance
- text rendering artifacts
- incorrect words
- missing words

Treat all labels as correct regardless of their appearance.

Assume that any text-related issue has already been accepted and should never be corrected.

Evaluate ONLY:

1. Presence of biological structures required by the prompt
2. Anatomical correctness of structures
3. Spatial relationships between structures
4. Educational organization and clarity
5. Separation and visibility of important biological components

Generate image editing instructions that improve biological structure only.

Rules:

- Every instruction must refer to a biological structure or anatomical relationship.
- Never mention labels.
- Never mention text.
- Never mention spelling.
- Never mention words.
- Never mention annotations.
- Never mention typography.
- Never mention image quality.
- Never mention aesthetics.
- Never mention backgrounds unless they hide a biological structure.

- Prefer minimal modifications.
- Preserve as much of the image as possible.
- Modify only structures that are missing, incorrect, unclear, or poorly separated.

Maximum 50 words.

Output only editing instructions.

End with exactly this sentence:

Preserve all existing correct biological structures, composition, colors, labels, and educational layout. Modify only the specified biological structures.

Examples:

Add visible underground mycelial network beneath the stalk.
Separate the gills more clearly from the underside of the cap.
Increase visibility of the annulus ring around the stalk.
Preserve all existing correct biological structures, composition, colors, labels, and educational layout. Modify only the specified biological structures.
"""

        # messages = [
        #     {
        #         "role": "system",
        #         "content": system_prompt
        #     },
        #     {
        #         "role": "user",
        #         "content":
        #         f"Prompt:\n{prompt}"
        #     }
        # ]
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": f"""
        Required biological content:

{prompt}

Evaluate only biological structures and anatomical relationships.

Ignore all text in the image.

Generate structural editing instructions only.
        """
                    }
                ]
            }
        ]
        
        text = (
            self.vlm_processor
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        inputs = self.vlm_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        # text = (
        #     self.vlm_processor
        #     .apply_chat_template(
        #         messages,
        #         tokenize=False,
        #         add_generation_prompt=True
        #     )
        # )

        # inputs = self.vlm_processor(
        #     text=[text],
        #     images=[image],
        #     return_tensors="pt"
        # )

        # inputs = {
        #     k: v.to(self.vlm_model.device)
        #     for k, v in inputs.items()
        # }
        inputs = {
            k: v.to("cpu")
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.vlm_model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False
            )

        # response = self.vlm_processor.batch_decode(
        #     outputs,
        #     skip_special_tokens=True
        # )[0]

        # response = response.strip()

        # print("\nRefinement Instructions:")
        # print(response)

        # return response
        
        generated_ids = outputs[
            0,
            inputs["input_ids"].shape[1]:
        ]

        response = self.vlm_processor.decode(
            generated_ids,
            skip_special_tokens=True
        )

        response = response.strip()

        print("\nRefinement Instructions:")
        print(response)

        return response