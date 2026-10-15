import os
import sys
import torch

from peft import PeftModel

from config import OMNIGEN_REPO_PATH
if OMNIGEN_REPO_PATH and os.path.isdir(OMNIGEN_REPO_PATH):
    sys.path.insert(0, OMNIGEN_REPO_PATH)

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

from config import (
    BASE_MODEL,
    DEVICE,
    HEIGHT,
    WIDTH,
    NUM_STEPS,
    GUIDANCE_SCALE,
    LORA_PATH,
    IMAGE_GUIDANCE_SCALE,
    SEED
)


class ImageGenerator:

    def __init__(self):

        print("Loading OmniGen2...")

        self.pipe = OmniGen2Pipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(DEVICE)

        self.pipe.transformer.eval()

        self.base_transformer = self.pipe.transformer

        self.pipe.transformer = PeftModel.from_pretrained(
            self.pipe.transformer,
            LORA_PATH,
        )

        self.pipe.transformer.enable_teacache = False
        self.pipe.transformer.enable_taylorseer = False
        self.pipe.transformer.eval()

        print("OmniGen2 loaded successfully.")

    def generate(
        self,
        prompt,
        guidance_scale=GUIDANCE_SCALE
    ):

        with torch.no_grad():

            # output = self.pipe(
            #     prompt=prompt,
            #     height=HEIGHT,
            #     width=WIDTH,
            #     text_guidance_scale=guidance_scale,
            #     num_inference_steps=NUM_STEPS
            # )
            generator = torch.Generator(
                device=DEVICE
            ).manual_seed(SEED)

            output = self.pipe(
                prompt=prompt,
                height=HEIGHT,
                width=WIDTH,
                text_guidance_scale=guidance_scale,
                num_inference_steps=NUM_STEPS,
                generator=generator
            )

        image = output.images[0]

        return image

    def edit(
        self,
        image,
        prompt
    ):

        with torch.no_grad():

            generator = torch.Generator(
                device=DEVICE
            ).manual_seed(SEED)
            
            output = self.pipe(
                prompt=[prompt],
                input_images=[image],
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_STEPS,
                max_sequence_length=1024,
                text_guidance_scale=GUIDANCE_SCALE,
                image_guidance_scale=IMAGE_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                output_type="pil",
                generator=generator,
            )
        #     output = self.pipe(
        #         prompt=[prompt],
        #         input_images=[[image]],
        #         size=[(WIDTH, HEIGHT)],
        #         num_inference_steps=NUM_STEPS,
        #         max_sequence_length=1024,
        #         text_guidance_scale=GUIDANCE_SCALE,
        #         # image_guidance_scale=2.0,
        #         image_guidance_scale=IMAGE_GUIDANCE_SCALE,
        #         num_images_per_prompt=1,
        #         output_type="pil",
        #     )

        edited_image = output.images[0]

        return edited_image