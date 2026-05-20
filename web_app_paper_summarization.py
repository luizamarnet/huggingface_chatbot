import gradio as gr
from paper_summarization_backend import retriever_qa
from gradio_cutstom_theme import earth_yellow

with gr.Blocks(theme=earth_yellow.theme, css=earth_yellow.css) as rag_application:
    
    # Cabeçalho do App
    gr.Markdown("# <center>My Own Personal Bot</center>")
    gr.Markdown("<center>Upload a PDF document. The chatbot will extract key information using the provided document.</center>")
    
    '''# Add bloco HTML vazio com espaçamento vertical
    gr.HTML("<div style='height: 20px;'></div>")'''
    
    # --- LINHA 1: UPLOAD E BOTÕES CENTRALIZADOS ---
    with gr.Row():
        gr.Column(scale=1)  # Margem esquerda para empurrar pro centro
        
        with gr.Column(scale=2):  # Bloco central do Upload
            pdf_input = gr.File(
                label="Upload PDF File", 
                file_count="single", 
                file_types=['.pdf'], 
                type="filepath"
            )
            
            # Botões lado a lado logo abaixo do arquivo
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Submit", variant="primary")
                
        gr.Column(scale=1)  # Margem direita
        
    '''# Add bloco HTML vazio com espaçamento vertical
    gr.HTML("<div style='height: 20px;'></div>")'''
    
    # --- LINHA 2: OUTPUTS EM DUAS COLUNAS PARALELAS ---
    with gr.Row():
        # Coluna da Esquerda: Título e Tópico (Sumário)
        with gr.Column(scale=1):
            out_title = gr.Textbox(label="Paper Title", interactive=False, lines=2)
            out_authors = gr.Textbox(label="Paper Authors", interactive=False, lines=2)
            out_affiliation = gr.Textbox(label="Authors Affiliation", interactive=False, lines=2)
            
        # Coluna da Direita: Autores e Conclusões
        with gr.Column(scale=1):
            out_topic = gr.Textbox(label="Paper Topic", interactive=False, lines=8)
            out_conclusions = gr.Textbox(label="Paper Conclusions", interactive=False, lines=8)

    # --- MAPEAMENTO DAS AÇÕES DOS BOTÕES ---
    submit_btn.click(
        fn=retriever_qa,
        inputs=[pdf_input],
        outputs=[out_title, out_authors, out_affiliation, out_topic, out_conclusions]
    )
    
    # Ação do botão limpar (esvazia todos os campos)
    clear_btn.click(
        fn=lambda: [None, "", "", "", "", ""],
        inputs=[],
        outputs=[pdf_input, out_title, out_authors, out_affiliation, out_topic, out_conclusions]
    )

if __name__ == "__main__":
    rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)
