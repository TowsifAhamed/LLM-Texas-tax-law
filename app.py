from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import faiss
import numpy as np
import os
from groq import Groq

app = Flask(__name__)

# Initialize Groq API
groq_api_key = ""  # Replace with your actual Groq API key
client = Groq(api_key=groq_api_key)

# Initialize models
embedding_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Constants
MAX_TOKENS = 30000
CONTEXT_TOKEN_LIMIT = 4000

# Data
threads = {}  # Stores chat history for each thread
chapters = []  # Stores chapter metadata
faiss_index = None  # FAISS index for similarity search

# Load chapters from the 'doc' directory
def load_chapters_from_doc():
    global chapters, faiss_index
    doc_dir = "doc"
    if not os.path.exists(doc_dir):
        print(f"Error: Directory '{doc_dir}' not found.")
        exit(1)

    chapters = []
    for file_name in os.listdir(doc_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(doc_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                chapters.append({"id": file_name, "content": content})

    # Create FAISS index
    texts = [chapter["content"] for chapter in chapters]
    embeddings = embedding_model.encode(texts)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings))
    print(f"FAISS index built with {len(chapters)} documents.")

# Retrieve relevant documents
def retrieve_relevant_documents(query, k=2):
    """Retrieve the top-k relevant chapters based on the query."""
    query_embedding = embedding_model.encode([query])
    _, indices = faiss_index.search(np.array(query_embedding), k)
    relevant_docs = [chapters[idx]["content"] for idx in indices[0]]
    print(f"Top {k} relevant documents retrieved.")
    return relevant_docs

# Filter relevant portions from documents
def filter_relevant_portions(query, documents, max_context_tokens=CONTEXT_TOKEN_LIMIT - 500):
    """Filter the most relevant portions of the documents."""
    chunks = []
    for doc in documents:
        chunks.extend(doc.split("\n"))  # Split each document into sentences or lines

    # Encode query and chunks for similarity scoring
    query_embedding = embedding_model.encode([query])
    chunk_embeddings = embedding_model.encode(chunks)

    # Compute similarity scores
    scores = np.dot(chunk_embeddings, query_embedding.T).squeeze()

    # Sort chunks by relevance
    ranked_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    # Select the most relevant chunks while staying within the token limit
    filtered_context = []
    current_token_count = 0
    for chunk, score in ranked_chunks:
        token_count = len(tokenizer(chunk)["input_ids"])
        if current_token_count + token_count > max_context_tokens:
            break
        filtered_context.append(chunk)
        current_token_count += token_count

    print(f"Filtered context token count: {current_token_count}")
    return "\n".join(filtered_context)

# Generate response using Groq
def generate_response_with_groq(query, context, chat_history, max_history=5):
    # Use only the last 'max_history' turns for context
    recent_history = chat_history[-(max_history * 2):]  # Each turn has both query and response
    compact_history = "\n".join(recent_history)

    # Combine context and recent history
    full_context = f"{compact_history}\n{context}"
    prompt_template = f"""
    Below is the context of several topics and a question. Use the context to provide an answer to the question.

    Context:
    {{}}

    Question: {query}
    Answer:
    """
    base_prompt = prompt_template.format("")
    base_token_length = len(tokenizer(base_prompt)["input_ids"])
    print(f"Base prompt token count: {base_token_length}")

    # Truncate context if it exceeds token limit
    context_lines = full_context.split("\n")
    while True:
        truncated_context = "\n".join(context_lines)
        total_token_length = len(tokenizer(truncated_context)["input_ids"]) + base_token_length
        if total_token_length <= CONTEXT_TOKEN_LIMIT or not context_lines:
            break
        context_lines.pop()

    # Create the final prompt
    prompt = prompt_template.format(truncated_context)
    print(f"Final prompt token count: {len(tokenizer(prompt)['input_ids'])}")
    print(f"Final prompt for debug:\n{prompt}")

    # Call Groq API
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )

    # Extract response content
    return response.choices[0].message.content.strip()

@app.route("/")
def index():
    return render_template("index.html")

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

@app.route("/ask_question", methods=["POST"])
def ask_question():
    thread_id = request.json.get("thread_id")
    query = request.json.get("query")
    if thread_id not in threads:
        return jsonify({"error": "Thread does not exist"}), 404

    # Retrieve chat history
    chat_history = threads[thread_id]

    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_documents(query)

    # Filter relevant portions from the documents
    context = filter_relevant_portions(query, relevant_docs)

    # Generate response using Groq
    response = generate_response_with_groq(query, context, chat_history)

    # Update thread history
    threads[thread_id].append(f"Query: {query}")
    threads[thread_id].append(f"Response: {response}")

    return jsonify({"response": response}), 200

if __name__ == "__main__":
    # Load chapters
    load_chapters_from_doc()
    app.run(debug=True)
