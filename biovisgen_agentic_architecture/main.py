"""Gradio UI for BioVisGen. Run from repository root with `python run.py`."""
import gradio as gr
from agents.orchestrator import Orchestrator

orchestrator = None

def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        orchestrator = Orchestrator()
    return orchestrator

def run_action(mode, prompt, source_image, editor_data):
    logs = []
    def progress(message):
        logs.append(str(message))
        print(message)

    if mode == "Generate a biology illustration":
        if not prompt or not prompt.strip():
            raise gr.Error("Enter a biology prompt.")
        result = get_orchestrator().run_pipeline(prompt, progress=progress)
        return result["final_prompt"], result["final_image"], None, result["best_score"], "\\n".join(logs)

    if source_image is None and not editor_data:
        raise gr.Error("Upload an image first.")

    if mode == "Remove text (EasyOCR + SDXL)":
        image = source_image
        if image is None and editor_data:
            image = editor_data.get("background")
        result, mask = get_orchestrator().remove_text(image, progress=progress)
        return "Text removal: EasyOCR-generated mask + SDXL inpainting", result, mask, None, "\\n".join(logs)

    if editor_data is None or editor_data.get("background") is None:
        raise gr.Error("Upload an image in the image editor and draw a mask.")
    layers = editor_data.get("layers") or []
    if not layers:
        raise gr.Error("Draw a mask over the region to modify.")
    import numpy as np
    from PIL import Image
    layer = layers[0].convert("RGBA")
    mask = layer.getchannel("A")
    # Some Gradio versions expose the drawing as RGB with opaque alpha.
    arr = np.asarray(layer)
    if np.asarray(mask).max() == 0 and arr.shape[-1] == 4:
        mask = Image.fromarray(np.asarray(layer)[:, :, 0])
    mask = mask.point(lambda px: 255 if px > 0 else 0)
    result = get_orchestrator().localized_refinement(
        editor_data["background"], mask, prompt, progress=progress
    )
    return prompt, result, mask, None, "\\n".join(logs)

with gr.Blocks(title="BioVisGen") as demo:
    gr.Markdown("# BioVisGen\nAgentic generation, text removal, and localized refinement for biology illustrations.")
    mode = gr.Radio(
        ["Generate a biology illustration", "Remove text (EasyOCR + SDXL)", "Localized refinement (SDXL)"],
        value="Generate a biology illustration", label="Orchestrator action"
    )
    prompt = gr.Textbox(label="Biology prompt / localized modification prompt", lines=3)
    with gr.Row():
        source_image = gr.Image(type="pil", label="Input image (text removal)")
        editor = gr.ImageEditor(type="pil", label="Input image + draw a mask (localized refinement)")
    run = gr.Button("Run", variant="primary")
    final_prompt = gr.Textbox(label="Prompt used")
    result_image = gr.Image(type="pil", label="Result")
    mask_output = gr.Image(type="pil", label="Generated / used mask")
    score = gr.Number(label="Best CLIP critic score (generation only)")
    logs = gr.Textbox(label="Orchestrator log", lines=12)
    run.click(
        run_action,
        inputs=[mode, prompt, source_image, editor],
        outputs=[final_prompt, result_image, mask_output, score, logs]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
