# 🤖 Gradio-Apps Academic RAG Assistants [🚧 Project Under Construction 🚧]


This workspace hosts multiple **Retrieval-Augmented Generation (RAG)** applications tailored for scientific papers analysis, using LangChain, Hugging Face, and Gradio.

---


## 📱 Available Applications

For now, this repository features two applications:

---

### 1. 🔍 Automated Paper Information Extractor (`web_app_paper_summarization.py`)
Optimized for extracting the paper main information of an academic paper. 
The user only needs to upload and submit the paper and the application provides the paper's:
* Title;
* Authors; 
* Authors' affiliations;
* Core topic and objectives;
* Main conclusions.

**Design Feature:** The app is engineered to provide the pages and sections utilized to extract the core objectives and conclusions. Each fact should includes a short, 4-to-5-word 'search phrase' that users can copy and paste into the original PDF to easily find and verify the app's answers. *(Note: This reference functionality is currently under optimization, by now it does not provide precise references, and will be fully refined in future updates).*


---


### 2. 💬 Open Question RAG Chatbot (`web_app_open_question_chabot.py`)
Designed for interactive, deep-dive document exploration.
* **Free-Form Queries**: Provides an interface which allows users to ask any custom or open-ended question about the uploaded PDF content.


## 🛠️ Technical Stack & Architecture

* **LLM Engine**: Hugging Face Endpoint (`meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Meta-Llama-3-8B-Instruct`)
* **Local Embedding Layer**: `sentence-transformers/all-MiniLM-L6-v2` running natively inside the host CPU memory allocation.
* **Data Processing**: Recursive character text splitting ($Chunk Size: 1000$, $Overlap: 200$) with integrated **NFKC Unicode normalization** to repair broken encoding artifacts.
* **Vector Database**: Transient `chromadb.EphemeralClient` configured with automated collection self-clearing mechanisms for multi-session stability.


## 📦 Local Installation & Setup

### 1. Clone & Navigate
```bash
git clone [https://github.com](https://github.com/luizamarnet/huggingface_chatbot.git)
cd huggingface_chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root folder of your project and add your personal hugging face tokes, which you should create in your hugging face account under 'Access Tokens' (in the menu under your profile photo):
```env
OPENAI_API_KEY=<your_hugging_face_token_here>
```

### 5. Launch the Apps on your local PC
* **To run the Automated Paper Information Extractor:**
  ```bash
  python web_app_paper_summarization.py
  ```
* **To run the Open Question RAG Chatbot:**
  ```bash
  python web_app_open_question_chabot.py
  ```

---

## 🔮 Future Work

1. Clean and document the codes.
2. Optimize the application's colors for mobile versions.
3. Benchmark and integrate alternative Hugging Face LLM Endpoints.
4. Implement a user-facing dropdown to switch dynamically between LLMs, such as:
   * `meta-llama/Llama-3.1-8B-Instruct`
   * `deepseek-ai/DeepSeek-R1`
   * `Qwen/Qwen2.5-7B-Instruct` 
   * `meta-llama/Meta-Llama-3-8B-Instruct`
5. Refine the verbatim citation and reference logic in the **Automated Paper Information Extractor** application.
6. Improve prompt engineering for extracting the core topic, objective, and conclusion of the pepers uploaded into the **Automated Paper Information Extractor** application.

---


## 📊 Empirical Results & Limitations

Below are observations and evaluation notes for the **Automated Paper Information Extractor** application, with screenshots showcasing tests run on 3 different papers, in which our application achieved varying degrees of accuracy:

### 1. Paper Evaluation: [Bridging the Sim-to-Real GAP for Underwater Image Segmentation](https://stellagrasshof.com/assets/pdf/2024_marnet_oceans.pdf)
* The paper was submitted twice to this application. The extracted text in each result differed slightly but summarized the main topic, objectives, and conclusions quite well. However, in both trials, the app failed to identify that this paper also released a new public dataset.
* Character decoding limitations were observed: the application failed to accurately reconstruct the special Unicode character `ą` in the author's name *Wąsowski*.

### 2. Paper Evaluation: [MIMIR-UW: A Multipurpose Synthetic Dataset for Underwater Navigation and Inspection](https://www.researchgate.net/profile/Olaya-Alvarez-Tunon/publication/376503081_MIMIR-UW_A_Multipurpose_Synthetic_Dataset_for_Underwater_Navigation_and_Inspection/links/65c09fb71bed776ae32f3367/MIMIR-UW-A-Multipurpose-Synthetic-Dataset-for-Underwater-Navigation-and-Inspection.pdf)
* In this test case, the system failed to recognize the dataset release (*MIMIR*) as the core contribution of the paper. The app stated that the primary objective was testing different SLAM algorithms on the MIMIR dataset. The true core objective was to publish the MIMIR dataset itself, which included testing visual SLAM and semantic segmentation algorithms.

### 3. Paper Evaluation: [SubPipe: A Submarine Pipeline Inspection Dataset for Segmentation and Visual-inertial Localization](https://arxiv.org/pdf/2401.17907)
* The application correctly identified that the primary objective was introducing a new dataset called *SubPipe*. However, it omitted the fact that it is a multipurpose dataset, profiling it strictly as a semantic segmentation dataset. 
* While the extracted conclusions were not factually incorrect, the context synthesis lacked detail. The app discussed a 'model' and its performance results without properly explaining which model was used or how it correlated to the dataset released by the paper, leaving the user without understanding the correlation between the conclusions and the topic/objective of the paper.