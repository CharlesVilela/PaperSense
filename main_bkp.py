import ollama
from transformers import AutoTokenizer
import torch

# ========= CONFIGURAÇÃO =========
MODEL_OLLAMA = "llama3"  # Pode ser 'mistral', 'phi3', etc.
MODEL_HF = "bert-base-uncased"  # usado apenas para tokenização/análise

# ========= PREPARAÇÃO =========
print("🔹 Carregando tokenizer do Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)

print("🔹 Chatbot iniciado com modelo:", MODEL_OLLAMA)
print("Digite 'exit' para sair.\n")

# ========= LOOP DE CONVERSA =========
while True:
    user_input = input("Você: ")
    if user_input.lower() in ["exit", "sair", "quit"]:
        print("👋 Encerrando chatbot.")
        break

    # Pré-processamento simples com tokenizer
    tokens = tokenizer.tokenize(user_input)
    print(f"🧩 Tokens: {tokens[:10]} ...")

    # Chamada ao modelo Ollama
    response = ollama.chat(
        model=MODEL_OLLAMA,
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    print("🤖 Bot:", response["message"]["content"])
