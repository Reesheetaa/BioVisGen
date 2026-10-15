import os
import re

from config import (
    GUIDANCE_SCALE,
    MAX_REFINEMENT_ROUNDS,
    CRITIC_THRESHOLD,
    INTERMEDIATE_DIR,
    FINAL_DIR
)

from utils.helpers import create_timestamp

from agents.llm_agent import LLMAgent
from agents.prompt_refiner import PromptRefiner
from agents.image_generator import ImageGenerator
from agents.critic import Critic
from agents.mask_generator import TextMaskGenerator
from agents.sdxl_inpainter import SDXLInpainter


class Orchestrator:

    def __init__(self):

        print("Initializing agents...")

        self.llm_agent = LLMAgent()

        self.prompt_refiner = PromptRefiner(
            self.llm_agent
        )

        self.image_generator = ImageGenerator()

        self.critic = Critic()
        self.text_mask_generator = None
        self.sdxl_inpainter = None

        print("All agents initialized successfully.")

    # =====================================================
    # ORCHESTRATOR DECISION:
    # SHOULD PROMPT BE REFINED?
    # =====================================================

    def should_refine_prompt(self, prompt):

        system_prompt = """
You are the Orchestrator of an educational biology image generation framework.

Your task is to evaluate whether a prompt contains sufficient information
for direct educational image generation.

Evaluate the prompt across five dimensions.

Dimension 1: Scientific Structures
Score 1 if specific biological structures, components, organs,
cells, tissues, processes, or anatomical elements are mentioned.
Otherwise score 0.

Dimension 2: Spatial Relationships
Score 1 if the prompt describes positions, organization,
arrangement, orientation, hierarchy, connections, interactions,
or relationships between structures.
Otherwise score 0.

Dimension 3: Educational Presentation
Score 1 if the prompt specifies educational characteristics such as:
diagram, illustration, textbook style, educational figure,
scientific visualization, learning-oriented presentation,
annotations, explanations, or instructional formatting.
Otherwise score 0.

Dimension 4: Visual Organization
Score 1 if the prompt specifies visual structure such as:
component separation, visual hierarchy, layout, grouping,
cross-section, exploded view, clarity requirements,
or organizational guidance.
Otherwise score 0.

Dimension 5: Labeling Requirements
Score 1 if labels, annotations, callouts, naming requirements,
or identification of structures are explicitly requested.
Otherwise score 0.

TOTAL_SCORE =
Scientific Structures +
Spatial Relationships +
Educational Presentation +
Visual Organization +
Labeling Requirements

Respond ONLY with a number from 0 to 5.

Do not explain.
Do not repeat the prompt.
Do not output anything else.
"""

        response = self.llm_agent.generate(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_new_tokens=5,
            temperature=0.01
        )

        response = response.strip()

        print("\nPrompt Quality Response:")
        print(response)

        match = re.search(r"[0-5]", response)

        if match:
            score = int(match.group())
        else:
            score = 0

        print(f"Prompt Quality Score = {score}")

        needs_refinement = score < 4

        print(
            f"Refinement Required = {needs_refinement}"
        )

        return needs_refinement

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def run_pipeline(
        self,
        user_prompt,
        progress=print
    ):

        intermediate_images = []

        progress(
            "Step 1: Orchestrator evaluating whether prompt refinement is required..."
        )

        needs_refinement = self.should_refine_prompt(
            user_prompt
        )

        # =====================================================
        # PROMPT REFINEMENT
        # =====================================================

        if needs_refinement:

            progress(
                "Prompt refinement required."
            )

            final_prompt = (
                self.prompt_refiner.refine_prompt(
                    user_prompt
                )
            )

            progress(
                "Prompt successfully refined."
            )

        else:

            progress(
                "Prompt refinement not required."
            )

            final_prompt = user_prompt

        progress("\nFinal Prompt Used:\n")
        progress(final_prompt)

        timestamp = create_timestamp()

        best_score = -1.0
        best_image = None

        progress(
            "\nStarting image generation pipeline...\n"
        )

        # =====================================================
        # INITIAL GENERATION
        # =====================================================

        progress(
            "Generating initial image..."
        )

        image = self.image_generator.generate(
            prompt=final_prompt,
            guidance_scale=GUIDANCE_SCALE
        )

        image_path = os.path.join(
            INTERMEDIATE_DIR,
            f"{timestamp}_generation.png"
        )

        image.save(image_path)

        intermediate_images.append(
            image_path
        )

        progress(
            f"Saved:\n{image_path}"
        )

        # =====================================================
        # CRITIC + REFINEMENT LOOP
        # =====================================================

        for iteration in range(
            MAX_REFINEMENT_ROUNDS + 1
        ):

            progress(
                "\nEvaluating image..."
            )

            current_score = (
                self.critic.score(
                    image=image,
                    prompt=final_prompt
                )
            )

            progress(
                f"Critic Score: {current_score:.4f}"
            )

            if current_score > best_score:

                best_score = current_score
                best_image = image

            if current_score >= CRITIC_THRESHOLD:

                progress(
                    "\nCritic threshold satisfied."
                )

                progress(
                    "Stopping refinement loop."
                )

                break

            if iteration >= MAX_REFINEMENT_ROUNDS:

                progress(
                    "\nMaximum refinement rounds reached."
                )

                break

            progress(
                "\nCritic score below threshold."
            )

            progress(
                "Generating refinement instructions..."
            )

            refinement_prompt = (
                self.critic.generate_refinement(
                    image=image,
                    prompt=final_prompt
                )
            )

            progress(
                "\nRefinement Instructions:\n"
            )

            progress(
                refinement_prompt
            )

            progress(
                "\nApplying OmniGen2 edit..."
            )

            image = self.image_generator.edit(
                image=image,
                prompt=refinement_prompt
            )

            image_path = os.path.join(
                INTERMEDIATE_DIR,
                f"{timestamp}_refinement_{iteration+1}.png"
            )

            image.save(image_path)

            intermediate_images.append(
                image_path
            )

            progress(
                f"Saved:\n{image_path}"
            )

        # =====================================================
        # SAVE FINAL OUTPUT
        # =====================================================

        final_output_path = os.path.join(
            FINAL_DIR,
            f"{timestamp}_final.png"
        )

        best_image.save(
            final_output_path
        )

        progress(
            "\nFinal image saved successfully."
        )

        progress(
            f"Final Path:\n{final_output_path}"
        )

        progress(
            f"\nBest Critic Score: {best_score:.4f}"
        )

        return {
            "final_prompt": final_prompt,
            "best_score": best_score,
            "intermediate_images": intermediate_images,
            "final_image": best_image,
            "final_image_path": final_output_path
        }