import os
import gradio as gr
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch

MODEL_DIR = "results/quantized_model"
print("[INFO] - Loading quantized ONNX model and tokenizer...")

# Load model and tokenizer
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] - Model successfully loaded on {device}")

# Sample article text for demo
SAMPLE_ARTICLE = """AI Startups Redefine Healthcare with Predictive Technology

Over the past few years, artificial intelligence has begun transforming the healthcare industry in ways that were once considered science fiction. Startups across the globe are developing AI-powered systems capable of predicting diseases before they manifest, optimizing hospital operations, and improving the accuracy of diagnoses.

One of the most notable breakthroughs is the use of deep learning models to analyze patient data and identify early signs of chronic illnesses such as diabetes, heart disease, and cancer. By detecting patterns invisible to human eyes, these systems allow doctors to intervene earlier, increasing the chances of successful treatment.

Moreover, AI-driven tools are helping hospitals manage resources more efficiently. Predictive models forecast patient inflow, enabling better allocation of staff, equipment, and hospital beds. This optimization not only reduces operational costs but also improves patient satisfaction by minimizing wait times.

However, experts caution that the integration of AI in healthcare raises important ethical and privacy concerns. Patient data must be handled responsibly, and AI models need to be transparent enough for medical professionals to understand their decision-making process.

Despite these challenges, the momentum behind AI innovation in healthcare continues to accelerate. Investors are pouring billions into startups working on predictive diagnostics, robotic surgery, and personalized medicine. Many believe that within the next decade, AI will become as essential to healthcare as the stethoscope — revolutionizing patient care on a global scale.
"""

# Summarization function
def summarize_text(article, max_length=120, min_length=30):
    if not article.strip():
        return " Please enter or load some text to summarize."

    inputs = tokenizer(
        article, return_tensors="pt",
        truncation=True, padding=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=4
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# Gradio UI
with gr.Blocks(title="SmartSumm - FLAN-T5 Summarizer") as demo:
    gr.Markdown(
        """
        #  SmartSumm: FLAN-T5 Summarizer  
        ---
        A fine-tuned **FLAN-T5-small** model trained on the **CNN/DailyMail** dataset,  
        then **quantized to 8-bit using ONNX Runtime** for efficient inference.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Enter Article or Paragraph",
                placeholder="Paste or type your text here...",
                lines=12
            )

            with gr.Row():
                load_sample_btn = gr.Button("Load Sample Article")
                summarize_btn = gr.Button("Generate Summary")

            max_len = gr.Slider(50, 250, value=100, step=10, label="Max Summary Length")
            min_len = gr.Slider(10, 100, value=30, step=5, label="Min Summary Length")

        with gr.Column(scale=1):
            output_summary = gr.Textbox(
                label=" Generated Summary",
                placeholder="Summary will appear here...",
                lines=10
            )

    # Button logic
    load_sample_btn.click(
        fn=lambda: SAMPLE_ARTICLE,
        inputs=[],
        outputs=[input_text],
    )

    summarize_btn.click(
        fn=summarize_text,
        inputs=[input_text, max_len, min_len],
        outputs=output_summary,
    )

    gr.Markdown("---")
    gr.Markdown(
        """
        **Note:**  
        This model is a fine-tuned and quantized version of *FLAN-T5-small* trained on *CNN/DailyMail*.  
        Quantized using **ONNX Runtime** for faster and memory-efficient summarization.
        """
    )

demo.launch(share=True)
