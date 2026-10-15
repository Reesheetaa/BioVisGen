import os
import csv

from config import (
    GUIDANCE_SCALE,
    MAX_REFINEMENT_ROUNDS,
    CRITIC_THRESHOLD
)

from agents.orchestrator import Orchestrator

from prompts import prompts

# =====================================================
# OUTPUT DIRECTORY
# =====================================================

OUTPUT_DIR = "batch_outputs_with_vlm_critic_3"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =====================================================
# LOAD AGENTS ONCE
# =====================================================

orchestrator = Orchestrator()

# =====================================================
# CSV STORAGE
# =====================================================

csv_rows = []

# =====================================================
# BASIC PROMPTS ONLY
# =====================================================

for prompt_name, prompt_text in prompts.items():

    if not prompt_name.endswith("b"):
        continue

    prompt_number = prompt_name[:-1]

    print("\n" + "=" * 60)
    print(f"Processing Prompt {prompt_name}")
    print("=" * 60)

    # =====================================================
    # PROMPT REFINEMENT
    # =====================================================

    needs_refinement = (
        orchestrator.should_refine_prompt(
            prompt_text
        )
    )

    if needs_refinement:

        final_prompt = (
            orchestrator.prompt_refiner.refine_prompt(
                prompt_text
            )
        )

    else:

        final_prompt = prompt_text

    print("\nFinal Prompt:")
    print(final_prompt)

    # =====================================================
    # INITIAL GENERATION
    # =====================================================

    image = (
        orchestrator.image_generator.generate(
            prompt=final_prompt,
            guidance_scale=GUIDANCE_SCALE
        )
    )

    iteration = 1

    score = (
        orchestrator.critic.score(
            image=image,
            prompt=final_prompt
        )
    )

    best_score = score
    best_iteration = 1

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{prompt_number}_1_original.png"
    )

    image.save(output_path)

    print(
        f"\nIteration 1 Score = {score:.4f}"
    )

    # =====================================================
    # CRITIC REFINEMENT LOOP
    # =====================================================

    while (
        score < CRITIC_THRESHOLD
        and iteration <= MAX_REFINEMENT_ROUNDS
    ):

        print(
            "\nGenerating refinement instructions..."
        )

        refinement_prompt = (
            orchestrator.critic.generate_refinement(
                image=image,
                prompt=final_prompt
            )
        )

        print(
            "\nRefinement Prompt:"
        )

        print(
            refinement_prompt
        )

        image = (
            orchestrator.image_generator.edit(
                image=image,
                prompt=refinement_prompt
            )
        )

        iteration += 1

        score = (
            orchestrator.critic.score(
                image=image,
                prompt=final_prompt
            )
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{prompt_number}_{iteration}_edited.png"
        )

        image.save(output_path)

        print(
            f"\nIteration {iteration} Score = {score:.4f}"
        )

        if score > best_score:

            best_score = score
            best_iteration = iteration

    # =====================================================
    # CSV RECORD
    # =====================================================

    csv_rows.append(
        {
            "prompt_number": prompt_number,
            "iterations": iteration,
            "best_iteration": best_iteration,
            "best_score": round(
                best_score,
                4
            )
        }
    )

# =====================================================
# WRITE CSV
# =====================================================

csv_file = os.path.join(
    OUTPUT_DIR,
    "generation_summary.csv"
)

with open(
    csv_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "prompt_number",
            "iterations",
            "best_iteration",
            "best_score"
        ]
    )

    writer.writeheader()

    writer.writerows(
        csv_rows
    )

print("\nFinished.")
print(
    f"CSV saved to: {csv_file}"
)