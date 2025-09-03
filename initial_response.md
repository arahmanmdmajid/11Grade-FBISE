### **Action Plan: Building "Ask KSA" in 2 Weeks**  
Here's your battle-tested roadmap to execute the project efficiently. We'll focus on **core RAG functionality first**, then add voice/multilingual features.

---

#### **Phase 1: Setup & Data Pipeline (Days 1-2)**  
**Goal**: Process official KSA PDFs into queryable chunks.  

1. **Environment Setup**  
   ```bash
   conda create -n ask_ksa python=3.10
   conda activate ask_ksa
   pip install pymupdf langchain sentence-transformers faiss-cpu
   ```

2. **PDF Processing Script** (`data_ingest.py`)  
   ```python
   import fitz  # PyMuPDF
   from langchain.text_splitter import RecursiveCharacterTextSplitter

   def extract_pdf_text(file_path):
       text = ""
       with fitz.open(file_path) as doc:
           for page in doc:
               text += page.get_text() + f"\n[Page {page.number}]"
       return text

   def chunk_text(text):
       splitter = RecursiveCharacterTextSplitter(
           chunk_size=500,
           chunk_overlap=50,
           separators=["\n\n", "\n", ". ", "! ", "? "]
       )
       return splitter.split_text(text)

   # Example usage
   pdf_text = extract_pdf_text("absher_visa.pdf")
   chunks = chunk_text(pdf_text)
   ```

---

#### **Phase 2: Vector Database (Days 3-4)**  
**Goal**: Create searchable knowledge base.  

1. **Embedding & FAISS Setup** (`vector_db.py`)  
   ```python
   from sentence_transformers import SentenceTransformer
   import faiss
   import numpy as np

   # Load embedding model
   model = SentenceTransformer('BAAI/bge-small-en-v1.5')

   # Generate embeddings
   embeddings = model.encode(chunks)

   # Create FAISS index
   dimension = embeddings.shape[1]
   index = faiss.IndexFlatL2(dimension)
   index.add(embeddings)

   # Save index
   faiss.write_index(index, "ksa_laws.index")
   ```

---

#### **Phase 3: Query Engine (Days 5-6)**  
**Goal**: Build core Q&A system.  

1. **Retrieval & Generation** (`query_engine.py`)  
   ```python
   from langchain.chains import RetrievalQA
   from langchain.llms import OpenAI  # or HuggingFaceHub for local LLM

   # Initialize components
   llm = OpenAI(api_key="YOUR_API_KEY", temperature=0)
   qa = RetrievalQA.from_chain_type(
       llm=llm,
       chain_type="stuff",
       retriever=index.as_retriever(search_kwargs={"k": 3})
   )

   def get_answer(query):
       result = qa({"query": query})
       return result["result"]
   ```

---

#### **Phase 4: UI & Citations (Days 7-8)**  
**Goal**: Create interactive interface.  

1. **Streamlit App** (`app.py`)  
   ```python
   import streamlit as st

   st.title("🇸🇦 Ask KSA")
   query = st.text_input("Ask about visas, iqama, or labor laws:")

   if query:
       answer = get_answer(query)
       st.subheader("Answer:")
       st.write(answer)

       # Display sources
       st.divider()
       st.subheader("Sources:")
       for source in retrieve_sources(query):  # Implement using FAISS metadata
           st.caption(f"📄 {source['title']} (Page {source['page']})")
   ```

---

#### **Phase 5: Multilingual & Voice (Days 9-10)**  
**Goal**: Add stretch features.  

1. **Arabic Support** (`translation.py`)  
   ```python
   from transformers import pipeline

   translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")

   def translate_arabic(query):
       return translator(query)[0]['translation_text']
   ```

2. **Voice Interface** (`voice.py`)  
   ```python
   import whisper
   from TTS.api import TTS

   # Speech-to-text
   stt_model = whisper.load_model("base")
   audio = st.audio_recorder("Speak:")
   if audio:
       query = stt_model.transcribe(audio.bytes)["text"]

   # Text-to-speech
   tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
   tts.tts_to_file(text=answer, file_path="output.wav", speaker="female")
   st.audio("output.wav")
   ```

---

#### **Phase 6: Deployment (Days 11-12)**  
**Goal**: Ship a working prototype.  

1. **Dockerize** (`Dockerfile`)  
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["streamlit", "run", "app.py"]
   ```

2. **Deploy to**  
   - Streamlit Cloud (free)  
   - Hugging Face Spaces (free)  
   - AWS EC2 (requires credit)  

---

### **Critical Pro Tips**  
1. **Prioritize Ruthlessly**  
   - Day 1: Get 1 PDF ingesting → chunking → querying  
   - Use sample PDF: [Absher Visa Guide](https://absher.sa/documents/Absher_User_Guide_En.pdf)  

2. **LLM Fallback Strategy**  
   ```python
   # config.py
   LLM_BACKUP = "openai"  # Alternatives: "huggingface" (free), "ollama" (local)
   ```

3. **Testing Workflow**  
   ```python
   TEST_CASES = [
       ("How to renew iqama?", "Visit Absher..."),
       ("ما هي مدة تجديد الإقامة؟", "عادة 3 أشهر...")
   ]
   ```

4. **Documentation**  
   - GitHub README with:  
     - Setup instructions  
     - Demo GIF  
     - Roadmap for future features  

---

### **Sample Daily Targets**  
| **Day** | **Key Milestone**                     | **Acceptance Criteria**                          |
|---------|--------------------------------------|-------------------------------------------------|
| 1       | PDF → Text extraction                | See parsed visa rules in console                |
| 3       | First FAISS query                    | "iqama renewal" returns relevant chunks         |
| 5       | Basic Streamlit UI                   | Type question → See GPT-generated answer        |
| 8       | Arabic query support                 | "كم رسوم تجديد الإقامة؟" → Arabic answer        |
| 10      | Voice input/output                   | Record question → Hear spoken answer            |
| 12      | Deployed prototype                   | Public URL works                               |

### **When You Get Stuck**  
1. **Steal Code**:  
   - LangChain RAG template: [GitHub Gist](https://gist.github.com/yourgist)  
   - Arabic Streamlit app: [Hugging Face Space](https://huggingface.co/spaces/yourspace)  
   
2. **Ask for Help**:  
   ```python
   # Emergency channels
   UMARI_HELP = "Slack @Umair, Office Hours Tue/Thu"
   COHORT_BUDDY = "Pair with GIS-savvy classmate"
   ```

You've got all the pieces – now execute like a pro! 🚀
