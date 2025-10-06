import ollama
from transformers import AutoTokenizer
import torch
import textwrap

# ========= CONFIGURAÇÃO =========
MODEL_OLLAMA = "llama3"  # Pode ser 'mistral', 'phi3', etc.
MODEL_HF = "bert-base-uncased"  # usado apenas para tokenização/análise

# ========= PREPARAÇÃO =========
print("🔹 Carregando tokenizer do Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)

custom_prompt = "Responda como se fosse um professor explicando para iniciantes."

def model_ollama(user_input):

    # Pré-processamento simples com tokenizer
    tokens = tokenizer.tokenize(user_input)
    print(f"🧩 Tokens: {tokens[:10]} ...")

    # Chamada ao modelo Ollama
    response = ollama.chat(
        model=MODEL_OLLAMA,
        messages=[
            {"role": "system", "content": custom_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    print(f"<UNK> Response: {response["message"]["content"]}")

    return response["message"]["content"]

def ask_ollama(context, query, model_name):
    # ======== 🔹 Montagem do prompt estruturado ========
    full_prompt = textwrap.dedent(f"""
        Você é um assistente inteligente especializado em responder perguntas com base em documentos.
        Use APENAS as informações contidas no CONTEXTO abaixo. 
        Se a resposta não estiver presente no contexto, diga claramente que ela não foi encontrada.

        ### CONTEXTO:
        {context}

        ### PERGUNTA:
        {query}

        ### INSTRUÇÕES:
        - Baseie-se apenas no conteúdo do CONTEXTO.
        - Não invente informações.
        - Se possível, mencione explicitamente a parte do texto que justifica sua resposta.
        """)

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": "Você é um assistente útil e preciso."},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response["message"]["content"]