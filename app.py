# Initialize Groq API
groq_api_key = ""

from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import faiss
import numpy as np
import os
import pickle
from groq import Groq

app = Flask(__name__)

# Initialize Groq API
client = Groq(api_key=groq_api_key)

# Initialize models
embedding_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Constants
MAX_TOKENS = 30000
CONTEXT_TOKEN_LIMIT = 4000
EMBEDDING_FILE = "chapter_embeddings.pkl"

# Data
chunk_data = []  # Stores metadata for each chunk
chunk_faiss_index = None  # FAISS index for chunk search
threads = {}  # Stores chat history for each thread

# Load embeddings or preprocess documents into chunks
def load_embeddings():
    global chunk_data, chunk_faiss_index
    doc_dir = "doc"
    chunk_embeddings = []

    if os.path.exists(EMBEDDING_FILE):
        # Load precomputed embeddings
        with open(EMBEDDING_FILE, "rb") as f:
            chunk_data, chunk_embeddings = pickle.load(f)
        print(f"Loaded {len(chunk_data)} chunks.")
    else:
        # Generate embeddings if not available
        if not os.path.exists(doc_dir):
            raise FileNotFoundError(f"Directory '{doc_dir}' not found.")

        print("Processing documents...")
        for file_name in os.listdir(doc_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(doc_dir, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read().strip()
                    for i, chunk_text in enumerate(content.split("\n\n")):
                        chunk_data.append({"id": f"{file_name}_{i}", "chunk": chunk_text})
                        chunk_embeddings.append(embedding_model.encode(chunk_text))

        with open(EMBEDDING_FILE, "wb") as f:
            pickle.dump((chunk_data, chunk_embeddings), f)
        print(f"Processed and saved {len(chunk_data)} chunks.")

    # Build FAISS index
    print("Building FAISS index...")
    dimension = len(chunk_embeddings[0])
    chunk_faiss_index = faiss.IndexFlatL2(dimension)
    chunk_embeddings = np.vstack(chunk_embeddings)
    chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)  # Normalize
    chunk_faiss_index.add(chunk_embeddings)
    print("FAISS index built.")

# Retrieve relevant chunks
def retrieve_relevant_chunks(query, k=10):
    if chunk_faiss_index is None or chunk_faiss_index.ntotal == 0:
        print("FAISS index is empty.")
        return []

    query_embedding = embedding_model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
    k = min(k, chunk_faiss_index.ntotal)
    _, indices = chunk_faiss_index.search(np.array(query_embedding), k)
    return [chunk_data[i] for i in indices[0]]

# Generate response using Groq API
def generate_response(query, context):
    if not context.strip():
        return "No relevant information found to answer the query."

    prompt_template = f"""
    Below is the context of several topics and a question. Use the context to provide an answer to the question.

    Context:
    {{}}

    Question: {query}
    Answer:
    """
    truncated_context = context[:CONTEXT_TOKEN_LIMIT]
    prompt = prompt_template.format(truncated_context)
    print("prompt : ", prompt)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )
    return response.choices[0].message.content.strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask_question", methods=["POST"])
def ask_question():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query is required."}), 400

    relevant_chunks = retrieve_relevant_chunks(query)
    context = "\n".join([chunk["chunk"] for chunk in relevant_chunks])
    response = generate_response(query, context)

    return jsonify({"response": response}), 200

@app.route("/create_thread", methods=["POST"])
def create_thread():
    thread_id = request.json.get("thread_id")
    if thread_id in threads:
        return jsonify({"error": "Thread already exists"}), 400
    threads[thread_id] = []
    return jsonify({"message": f"Thread {thread_id} created successfully"}), 201

@app.route("/show_all_threads", methods=["GET"])
def show_all_threads():
    return jsonify({"threads": list(threads.keys())}), 200

@app.route("/show_conversations/<thread_id>", methods=["GET"])
def show_conversations(thread_id):
    if thread_id not in threads:
        return jsonify({"error": "Thread does not exist"}), 404
    return jsonify({"thread_id": thread_id, "chat_history": threads[thread_id]}), 200

@app.route("/add_message", methods=["POST"])
def add_message():
    thread_id = request.json.get("thread_id")
    message = request.json.get("message")
    if thread_id not in threads:
        return jsonify({"error": "Thread does not exist"}), 404
    threads[thread_id].append(message)
    return jsonify({"message": "Message added successfully."}), 200

if __name__ == "__main__":
    load_embeddings()  # Automatically preprocess if needed
    app.run(debug=True, host="0.0.0.0", port=5003)
