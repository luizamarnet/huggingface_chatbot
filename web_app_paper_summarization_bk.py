import gradio as gr
from paper_summarization_backend import retriever_qa
from gradio_cutstom_theme import earth_yellow

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    #allow_flagging=.....,
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath")  # Drag and drop file upload
    ],
    outputs=[gr.Textbox(label="Paper Title"),
             gr.Textbox(label="Paper Authors"),
             gr.Textbox(label="Paper Topic"),
             gr.Textbox(label="Paper Conclusions")],
    title="My Own Personal Bot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
    theme=earth_yellow.theme,  # Applied here
    css=earth_yellow.css              # Applied here
)

# Launch the app
rag_application.launch(server_name="127.0.0.1", server_port= 7860, share=True)
