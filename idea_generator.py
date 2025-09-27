from transformers import pipeline
import gradio as gr

generator = pipeline(
    "text2text-generation",
    model="MBZUAI/LaMini-flan-T5-783M",
    device_map="auto"
)

def generate_ideas(domain):
    if not domain or not domain.strip():
        return "Please provide a domain (e.g., Agritech, Fintech, EdTech)."
    
    prompt = f"""
    Generate a startup pitch for the domain: {domain}.
    Include the following sections:

    1) Business ideas (2-3 sentences)
    2) Problem Statement (explain the challenges clearly)
    3) Solution (explain how the startup solves it)
    """

    result = generator(
        prompt,
        max_new_tokens=400,
        temperature=0.8,
        do_sample=True,
    )

    return result[0]["generated_text"]

demo = gr.Interface(
    fn=generate_ideas,
    inputs=gr.Textbox(label="Domain of Interest", placeholder="e.g., Agritech"),
    outputs=gr.Textbox(label="Startup Pitch", lines=12),
    title="Startup Idea + Pitch Generator",
    description="Enter a domain (Agritech, Fintech, EdTech, etc.) and get a startup pitch with Problem & Solution."
)

if __name__ == "__main__":
    demo.launch(share=True)
