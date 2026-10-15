from agents.llm_agent import LLMAgent
from agents.prompt_refiner import PromptRefiner

llm = LLMAgent()

refiner = PromptRefiner(llm)

prompt = "human heart diagram"

refined = refiner.refine_prompt(prompt)

print("\n====================")
print("REFINED PROMPT")
print("====================\n")

print(refined)
