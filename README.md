# PaperSense
---

## 🧠 **PaperSense — Entendendo Conhecimento, um Artigo de Cada Vez**

**PaperSense** é um aplicativo interativo que permite fazer perguntas inteligentes sobre o conteúdo de **artigos científicos e documentos em PDF**, combinando **Processamento de Linguagem Natural (PLN)**, **embeddings semânticos** e **modelos locais de linguagem (Ollama)** — tudo isso **sem depender de APIs externas** ou bibliotecas complexas como LangChain.

---

### 🚀 **Como funciona**

1. **Upload do PDF**
   O usuário envia um arquivo PDF diretamente pela interface do Streamlit.
   O sistema extrai automaticamente o texto usando `pdfplumber`.

2. **Pré-processamento e indexação**
   O texto é dividido em **blocos semânticos (chunks)** e convertido em **vetores de embeddings** usando o modelo `sentence-transformers/all-MiniLM-L6-v2`.
   Esses vetores são armazenados em um índice **FAISS** para buscas rápidas e eficientes.

3. **Busca de contexto relevante**
   Quando o usuário faz uma pergunta, o PaperSense localiza os trechos mais relacionados no texto, com base na similaridade vetorial entre a pergunta e os chunks do documento.

4. **Geração de resposta com o Ollama (LLaMA 3)**
   O contexto recuperado é enviado ao modelo **LLaMA3 (via Ollama)**, que elabora uma resposta **coerente, contextualizada e fiel ao conteúdo do PDF**.
   Nenhuma informação externa é usada — o modelo se baseia apenas no documento fornecido.

---

### 🧩 **Principais Tecnologias**

| Componente                                | Descrição                                         |
| ----------------------------------------- | ------------------------------------------------- |
| **Streamlit**                             | Interface interativa e simples para o usuário     |
| **pdfplumber / PyPDF2**                   | Extração precisa de texto de PDFs                 |
| **SentenceTransformer (MiniLM-L6-v2)**    | Geração de embeddings semânticos                  |
| **FAISS (Facebook AI Similarity Search)** | Indexação vetorial eficiente                      |
| **Ollama (LLaMA 3)**                      | Modelo local de linguagem para respostas naturais |
| **Python 3.10+**                          | Linguagem base do projeto                         |

---

### ⚙️ **Destaques do projeto**

* 🧾 Leitura e compreensão de artigos científicos diretamente em PDF
* ⚡ Busca semântica rápida e contextual com FAISS
* 🔒 Totalmente offline — sem dependência de APIs externas
* 🧠 Respostas geradas com base **exclusivamente** no conteúdo do documento
* 💬 Interface intuitiva no estilo *ChatGPT*, feita com Streamlit
* 🧩 Código modular e extensível (facilmente adaptável para outros domínios)

---

### 💡 **Exemplo de uso**

1. Faça upload de um artigo em PDF.
2. Espere o processamento e a criação dos embeddings.
3. Pergunte, por exemplo:

   * “Qual é o objetivo principal deste estudo?”
   * “Quais métodos foram utilizados?”
   * “Quais são as conclusões apresentadas?”
4. Receba respostas detalhadas, com base apenas no conteúdo do artigo.

---

### 🧭 **Próximos passos (roadmap)**

* [ ] Suporte a múltiplos PDFs
* [ ] Resumo automático do artigo
* [ ] Geração de citações automáticas
* [ ] Opção de áudio (leitura da resposta)
* [ ] Interface com histórico de contexto persistente




