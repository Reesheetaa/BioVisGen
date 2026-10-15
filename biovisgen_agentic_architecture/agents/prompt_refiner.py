class PromptRefiner:

    def __init__(self, llm_agent):

        self.llm = llm_agent

        self.system_prompt = """
You are an expert educational biology prompt refinement agent.

Your task is to rewrite short user prompts into highly detailed,
scientifically accurate educational image-generation prompts.

The rewritten prompt must:
- describe biological structures clearly
- include spatial relationships
- emphasize component separation
- encourage educational textbook-style diagrams
- maintain scientific correctness
- improve image generation quality
- avoid decorative or artistic wording
- use concise but information-rich scientific language

Generate ONLY the rewritten prompt.
"""

    def refine_prompt(self, prompt):

        user_prompt = f"""
Rewrite this biology image-generation prompt into a more detailed,
educational, scientifically structured prompt.

Original Prompt:
{prompt}

"""

        print("\nREFINED PROMPT:")
        refined_prompt = self.llm.generate(
            self.system_prompt,
            user_prompt,
            max_new_tokens=200
        )
        print(refined_prompt)
        print("=" * 80)

        return refined_prompt