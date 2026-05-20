import gradio as gr

class EarthYellow:
    
    theme = gr.themes.Citrus(
        neutral_hue="yellow",  
    )
    css = """
    button.primary {
        color: #854D0E !important /* var(--block-title-text-color) !important; - color: #FFFFFF !important;   Submit button text color (e.g., White) */
    }
    button.secondary {
        color: #FEF9C3 !important;  /* color: orange !important;   Clear button text color (e.g., Orange) */
    }
    .gradio-container .upload-html,
    .gradio-container .uploading,
    .gradio-container .file-name,
    .gradio-container span.file-size {
        color: #954D0E !important;
    }
    .gradio-container .eta,
    .gradio-container .progress-text,
    .gradio-container .meta-text,
    .gradio-container [class*="eta"],
    .gradio-container [class*="progress"] {
        color: #954D0E !important;
    }
    """



earth_yellow = EarthYellow()