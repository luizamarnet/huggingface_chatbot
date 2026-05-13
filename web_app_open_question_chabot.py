import gradio as gr
from open_question_chabot_chatbot_backend import retriever_qa
from gradio_cutstom_theme import earth_yellow

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    #allow_flagging=.....,
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Bot Answer"),
    title="My Own Personal Bot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
    theme=earth_yellow.theme,  # Applied here
    css=earth_yellow.css              # Applied here
)

# Launch the app
rag_application.launch(server_name="127.0.0.1", server_port= 7860, share=True)

# gradio deploy --app-file web_app_paper_summarization.py

