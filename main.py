import os
import uuid
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import boto3
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Configuration
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize ChromaDB
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("qa_collection", embedding_function=embedding_fn)

# Initialize Bedrock
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# PDF processing
def process_pdf(path, filename):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text
    split_chunks_and_store(full_text, filename)

# Image processing
def process_image(path, filename):
    image = Image.open(path)
    text = pytesseract.image_to_string(image)
    split_chunks_and_store(text, filename)

# Text splitting and storage
def split_chunks_and_store(text, filename):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [doc.page_content for doc in splitter.create_documents([text])]
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(documents=chunks, ids=ids, metadatas=[{"filename": filename}] * len(chunks))

# Home
@app.route("/")
def index():
    files = os.listdir(UPLOAD_FOLDER)
    return render_template("index.html", pdf_files=files)

# Upload
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    ext = filename.lower().split(".")[-1]
    try:
        if ext == "pdf":
            process_pdf(path, filename)
        elif ext in ["png", "jpg", "jpeg"]:
            process_image(path, filename)
        else:
            return jsonify({"error": "Unsupported file type"}), 400
        return jsonify({"message": "File uploaded & processing started."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Delete file
@app.route("/delete", methods=["POST"])
def delete():
    data = request.get_json()
    filename = data.get("filename")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    try:
        docs = collection.get()
        ids_to_delete = [
            id for id, meta in zip(docs["ids"], docs["metadatas"])
            if meta.get("filename") == filename
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return jsonify({"message": f"{filename} deleted."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ask question
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        results = collection.query(query_texts=[question], n_results=3)
        context = "\n".join(doc for doc_list in results["documents"] for doc in doc_list)
        prompt = f"""Use only the below context to answer:
        
Context:
{context}

Question: {question}"""

        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.7
            })
        )
        body = json.loads(response['body'].read())
        message = body.get("content", "No answer returned.")
        return jsonify({"answer": message})

    except Exception as e:
        print(f"❌ Bedrock error: {e}")
        return jsonify({"error": "Failed to get answer from Claude"}), 500

if __name__ == "__main__":
    app.run(debug=True)
