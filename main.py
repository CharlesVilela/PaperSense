import streamlit as st
import time
import uuid
import re
from datetime import datetime
from collections import deque
from PyPDF2 import PdfReader
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer

from execute_model import *

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"
CHUNK_SIZE = 500

# ================= FUNÇÕES =================
def extract_text_from_pdf1(pdf_file):
    """Extrai texto de todas as páginas do PDF."""
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, size=500, overlap=50):
    """Divide o texto em blocos (chunks) com leve sobreposição."""
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i+size]
        chunks.append(chunk)
    return chunks

# def chunk_text(text, size=500, overlap=50):
#     sections = re.split(r'\n(?=[A-ZÁÉÍÓÚÇ][^\n]{0,50}\n)', text)
#     return [sec.strip() for sec in sections if len(sec.strip()) > 100]

def create_faiss_index(chunks, model):
    """Cria índice FAISS com embeddings dos chunks."""
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_relevant_chunks(query, model, index, chunks, k=3):
    """Busca os chunks mais relevantes."""
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0]]


def historic():
    # Exibe todo o histórico
    for message in st.session_state.chatbot_responses:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])
        else:
            with st.chat_message("user"):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])


def display_incremental_response(text):
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        for chunk in text.split():
            full_text += chunk + " "
            placeholder.markdown(full_text + "▌")
            time.sleep(0.05)
        placeholder.markdown(full_text)
    return full_text


def main():
    st.title("PaperSense")

    audio_bk = None

    # Inicializações
    if "chatbot_responses" not in st.session_state:
        st.session_state.chatbot_responses = deque()
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = str(uuid.uuid4())
        st.session_state['start_time'] = time.time()
    # isQuestionAudio = False
    # isResponseAudio = False
    if 'isQuestionAudio' not in st.session_state:
        st.session_state.isQuestionAudio = False
    if 'isResponseAudio' not in st.session_state:
        st.session_state.isResponseAudio = False
    # # Inicialização do gravador
    # if 'gravador' not in st.session_state:
    #     st.session_state.gravador = GravadorAudio()

    # Inicialização de session_state
    if 'ultima_transcricao' not in st.session_state:
        st.session_state.ultima_transcricao = ""
    if 'tempo_audio' not in st.session_state:
        st.session_state.tempo_audio = 0
    if 'tempo_transcricao' not in st.session_state:
        st.session_state.tempo_transcricao = 0
    if 'timestamp' not in st.session_state:
        st.session_state.timestamp = None
    if 'pergunta_isaudio' not in st.session_state:
        st.session_state.pergunta_isaudio = False
    if 'audio_gravado' not in st.session_state:
        st.session_state.pergunta_isaudio = None
    if 'ultimo_audio' not in st.session_state:
        st.session_state.ultimo_audio = None

    if 'embed_model' not in st.session_state:
        st.session_state.embed_model = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'process_pdf' not in st.session_state:
        st.session_state.process_pdf = False

    with st.sidebar:

        uploaded_file = st.file_uploader("Choose a file")
        print('Uploaded file:', uploaded_file)

        if uploaded_file is not None:
            with st.spinner("Extraindo texto do PDF..."):
                text = extract_text_from_pdf(uploaded_file)

            st.success("✅ Texto extraído com sucesso!")
            st.write(f"📄 O PDF possui **{len(text)} caracteres**")

            if st.session_state.process_pdf != True:
                with st.spinner("🔹 Gerando embeddings e criando índice FAISS..."):
                    embed_model = SentenceTransformer(EMBED_MODEL)
                    chunks = chunk_text(text, size=CHUNK_SIZE)
                    index, _ = create_faiss_index(chunks, embed_model)

                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.embed_model = embed_model
                    st.session_state.process_pdf = True

                    st.success(f"✅ Índice criado com {len(chunks)} chunks!")

            st.divider()
            st.subheader("💬 Faça perguntas sobre o PDF")


    # Mostra histórico antes de gerar nova resposta
    historic()

    # Entrada do usuário
    user_input = st.chat_input("Me pergunte algo.")

    # Se houve entrada manual, prioriza ela; senão, usa transcrição se existir
    if user_input:
        prompt = user_input
        is_audio = False
        st.session_state.isQuestionAudio = False
    elif st.session_state.ultima_transcricao:
        prompt = st.session_state.ultima_transcricao
        # Limpa a transcrição após usar
        st.session_state.ultima_transcricao = ""
        is_audio = True
        st.session_state.isQuestionAudio = True
    else:
        prompt = None
        is_audio = False
        st.session_state.isQuestionAudio = False

    # Se houve nova pergunta
    if prompt:
        if is_audio:
            # Adiciona SOMENTE o áudio ao histórico
            with st.chat_message("user"):
                st.audio(st.session_state.ultimo_audio)

            st.session_state.chatbot_responses.append({
                "role": "user",
                "content": [{
                    "type": "audio_file",
                    "audio_file": st.session_state.ultimo_audio,
                }]
            })
            st.session_state.ultimo_audio = None
        else:
            # Mostrar pergunta imediatamente
            with st.chat_message("user"):
                st.write(prompt)
            # Adicionar pergunta no histórico
            st.session_state.chatbot_responses.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })

        # Gerar resposta
        with st.spinner("Gerando..."):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            timestamp_start = datetime.now()

            # response = model_ollama(prompt)
            relevant_chunks = retrieve_relevant_chunks(prompt, st.session_state.embed_model, st.session_state.index, st.session_state.chunks, k=10)
            context = "\n\n".join(relevant_chunks)
            response = ask_ollama(context, prompt, OLLAMA_MODEL)


            # Exibir resposta incrementalmente
        full_response = display_incremental_response(response)

        # Adicionar resposta no histórico
        st.session_state.chatbot_responses.append({
            "role": "assistant",
            "content": [{"type": "text", "text": full_response}]
        })

        isQuestionAudio = st.session_state.isQuestionAudio
        isResponseAudio = st.session_state.isResponseAudio
        timestamp_end = datetime.now()
        delta = timestamp_end - timestamp_start
        time_in_seconds = delta.total_seconds()
        # interaction.log_interaction(prompt, response, isQuestionAudio, isResponseAudio, time_in_seconds)

        st.rerun()


if __name__ == "__main__":
    main()