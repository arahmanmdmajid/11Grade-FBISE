### **Jupyter Notebook Workflow for "Ask KSA"**  
I'll structure this as a series of notebooks for step-by-step development. Create these files in your project directory:

```
ask-ksa/
├── 1_PDF_Processing.ipynb
├── 2_Vector_Database.ipynb
├── 3_Query_Engine.ipynb
├── 4_Multilingual_Voice.ipynb
├── 5_Streamlit_App.ipynb
└── data/
    └── absher_visa.pdf  # Sample PDF
```

---

### **1. PDF Processing** (`1_PDF_Processing.ipynb`)  
*Extract text from KSA government PDFs with metadata*

```python
# !pip install pymupdf langchain
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def extract_pdf(file_path):
    """Extract text with page numbers from PDF"""
    doc = fitz.open(file_path)
    text = ""
    metadata = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        text += f"{page_text}\n[PAGE:{page_num+1}]"
        metadata.append({
            "source": file_path,
            "page": page_num+1,
            "text": page_text
        })
    
    return text, metadata

# Test with sample PDF
text, metadata = extract_pdf("data/absher_visa.pdf")
print(f"Extracted {len(text.split())} words from PDF")

# Save metadata for later
with open("data/metadata.json", "w") as f:
    json.dump(metadata, f)

# Chunk text for processing
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", "! ", "? "]
)
chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks")
```

---

### **2. Vector Database** (`2_Vector_Database.ipynb`)  
*Create searchable knowledge base with FAISS*

```python
# !pip install sentence-transformers faiss-cpu
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Load chunks from previous step
with open("data/metadata.json") as f:
    metadata = json.load(f)
chunks = [item['text'] for item in metadata]

# Initialize embedding model
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# Save index
faiss.write_index(index, "data/ksa_laws.index")

# Test retrieval
test_query = "iqama renewal process"
query_embedding = model.encode([test_query])
distances, indices = index.search(query_embedding, k=3)

print("Top 3 results for:", test_query)
for idx in indices[0]:
    print(f"- Page {metadata[idx]['page']}: {chunks[idx][:100]}...")
```

---

### **3. Query Engine** (`3_Query_Engine.ipynb`)  
*Implement RAG with LLM integration*

```python
# !pip install openai langchain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import faiss
from sentence_transformers import SentenceTransformer
import os

# Initialize components
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
index = faiss.read_index("data/ksa_laws.index")
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

class KSARetriever:
    def __init__(self, index, model, metadata):
        self.index = index
        self.model = model
        self.metadata = metadata
        
    def get_relevant_documents(self, query, k=3):
        emb = self.model.encode([query])
        distances, indices = self.index.search(emb, k)
        return [self.metadata[idx] for idx in indices[0]]

retriever = KSARetriever(index, model, metadata)

def generate_answer(query):
    # Retrieve relevant chunks
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([f"Source (Page {doc['page']}): {doc['text']}" for doc in docs])
    
    # Generate answer
    prompt = f"""
    You're an expert on Saudi government procedures. Answer the user's question
    using ONLY information from the provided context. Always cite your sources.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    return llm(prompt)

# Test
print(generate_answer("What documents are needed for iqama renewal?"))
```

---

### **4. Multilingual & Voice** (`4_Multilingual_Voice.ipynb`)  
*Add Arabic support and voice interface*

```python
# !pip install transformers soundfile speechbrain
from transformers import pipeline
from speechbrain.pretrained import EncoderDecoderASR

# Arabic-to-English Translation
ar2en = pipeline("translation_ar_to_en", model="Helsinki-NLP/opus-mt-ar-en")

# English-to-Arabic Translation
en2ar = pipeline("translation_en_to_ar", model="Helsinki-NLP/opus-mt-en-ar")

# Speech-to-Text (Arabic/English)
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-ar", savedir="tmpdir_ar")

def process_arabic(query):
    # Arabic speech → text
    if isinstance(query, bytes):  # Audio input
        with open("tmp_audio.wav", "wb") as f:
            f.write(query)
        query = asr_model.transcribe_file("tmp_audio.wav")
    
    # Arabic text → English → Process → Arabic response
    en_query = ar2en(query)[0]['translation_text']
    en_answer = generate_answer(en_query)
    ar_answer = en2ar(en_answer)[0]['translation_text']
    return ar_answer

# Test Arabic text
print(process_arabic("ما هي مدة تجديد الإقامة؟"))
```

---

### **5. Streamlit App** (`5_Streamlit_App.ipynb`)  
*Build the final interface*

```python
# %%writefile app.py
import streamlit as st
from query_engine import generate_answer, process_arabic
import soundfile as sf
from speechbrain.pretrained import Tacotron2, HIFIGAN

# Initialize TTS
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")

st.title("🇸🇦 Ask KSA")
input_mode = st.radio("Input Mode:", ["Text", "Voice"])

if input_mode == "Text":
    query = st.text_input("Ask about visas, iqama, or labor laws:")
    lang = st.radio("Language:", ["English", "Arabic"])
else:
    audio = st.audio_recorder("Speak your question:")
    lang = "Arabic"  # Default for voice

if query or audio:
    if lang == "English":
        answer = generate_answer(query)
    else:
        answer = process_arabic(query if input_mode=="Text" else audio)
    
    st.subheader("Answer:")
    st.write(answer)
    
    # Text-to-speech
    if st.button("Hear Response"):
        mel_output, mel_length = tacotron2.encode_text(answer)
        waveforms = hifi_gan.decode_batch(mel_output)
        sf.write("response.wav", waveforms.squeeze(1).cpu().numpy(), 22050)
        st.audio("response.wav")
```

---

### **Execution Workflow**  
1. **Sequential Run**:  
   ```bash
   jupyter notebook 1_PDF_Processing.ipynb  # Run all cells
   jupyter notebook 2_Vector_Database.ipynb   # Run all cells
   ```
   
2. **API Setup**:  
   Get OpenAI API key from [platform.openai.com](https://platform.openai.com/)  
   ```bash
   export OPENAI_API_KEY='sk-...'  # Add to .env file
   ```

3. **Test Components**:  
   - In Notebook 3: Test with sample English queries  
   - In Notebook 4: Test Arabic translation  

4. **Launch App**:  
   ```bash
   pip install streamlit soundfile
   streamlit run app.py
   ```

---

### **Debugging Tips**  
1. **Common Issues**:  
   - PDF extraction fails → Try different PDFs (some are image-based)  
   - Arabic translation poor → Try `mbart-large-50` model  
   - FAISS loading error → Rebuild index with same dimension  

2. **Fallback Strategies**:  
   ```python
   # In query_engine.py
   try:
       answer = generate_answer(query)
   except:
       answer = "Sorry, I couldn't process your request. Please try again."
   ```

3. **Performance Monitoring**:  
   ```python
   %%timeit
   generate_answer("iqama renewal")
   ```

You now have a complete, debuggable workflow! Start with Notebook 1 and progress sequentially. Let me know if you hit any snags 🛠️
