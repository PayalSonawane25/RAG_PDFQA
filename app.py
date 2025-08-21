import os
import uuid
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from redis import Redis
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Redis rate limiter
redis_client = Redis(host="localhost", port=6379, db=0)
limiter = Limiter(get_remote_address, app=app, storage_uri="redis://localhost:6379")

# Uploads folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ChromaDB setup
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_collection", embedding_function=embedding_function)

# AWS Bedrock - Claude 3 Sonnet
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# Thread pool
executor = ThreadPoolExecutor()


# ---------------- File Processing ----------------

def process_pdf(filepath, filename):
    try:
        reader = PdfReader(filepath)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        if not text.strip():
            raise ValueError("No extractable text in PDF.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = [doc.page_content for doc in splitter.create_documents([text])]
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadata = [{"filename": filename} for _ in chunks]

        collection.add(documents=chunks, ids=ids, metadatas=metadata)
        print(f"✅ PDF processed: {filename} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")


def process_excel(filepath, filename):
    try:
        df = pd.read_excel(filepath)
        if df.empty:
            raise ValueError("Excel is empty.")

        rows = []
        for _, row in df.iterrows():
            row_text = "\n".join(f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]))
            rows.append(row_text)

        full_text = "\n\n".join(rows)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = [doc.page_content for doc in splitter.create_documents([full_text])]
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadata = [{"filename": filename} for _ in chunks]

        collection.add(documents=chunks, ids=ids, metadatas=metadata)
        print(f"✅ Excel processed: {filename} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"❌ Error processing Excel: {e}")


# ---------------- Routes ----------------

@app.route('/')
def home():
    files = os.listdir(UPLOAD_FOLDER)
    return render_template('index.html', pdf_files=files)


@app.route('/upload', methods=['POST'])
@limiter.limit("20 per minute")
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No filename specified"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if file.filename.lower().endswith(".pdf"):
        executor.submit(process_pdf, filepath, file.filename)
    elif file.filename.lower().endswith(".xlsx"):
        executor.submit(process_excel, filepath, file.filename)
    else:
        return jsonify({"error": "Only .pdf or .xlsx files are supported."}), 400

    return jsonify({"message": "File uploaded and processing started."})


@app.route('/delete', methods=['POST'])
def delete_file():
    data = request.json
    filename = data.get("filename", "")

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"🗑 Deleted file: {filename}")
    else:
        return jsonify({"error": "File not found"}), 404

    try:
        data = collection.get()
        ids_to_delete = [
            doc_id for doc_id, meta in zip(data["ids"], data["metadatas"])
            if meta.get("filename") == filename
        ]
        collection.delete(ids=ids_to_delete)
        print(f"🗑 Deleted {len(ids_to_delete)} ChromaDB entries.")
    except Exception as e:
        print(f"❌ Failed to delete from ChromaDB: {e}")
        return jsonify({"error": "Failed to delete data"}), 500

    return jsonify({"message": f"{filename} and associated data deleted."})


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        print("🧠 Question:", question)

        results = collection.query(query_texts=[question], n_results=3)
        documents = results.get("documents", [])
        relevant_texts = [doc for doc_list in documents for doc in doc_list]

        print("📦 Top ChromaDB matches:")
        for i, doc in enumerate(relevant_texts):
            print(f"Chunk {i+1}:", doc[:150], "...")

        if not relevant_texts:
            return jsonify({
                "summary": "No matching data found.",
                "detailed_answer": "No relevant information found in the uploaded documents."
            })

        context = "\n\n".join(relevant_texts[:3])

        # Build prompt for detailed answer
        detailed_prompt = f"""
Use the context below to answer the user's question in detail.
If the answer is not found in the context, say: 'Answer not available in the provided documents.'

Context:
{context}

Question: {question}
"""
        body_detailed = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": detailed_prompt}],
            "max_tokens": 700,
            "temperature": 0.7
        }

        # Claude detailed answer
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body_detailed)
        )

        response_body = json.loads(response['body'].read())
        content_data = response_body.get("content", "")

        if isinstance(content_data, list):
            detailed_answer = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content_data
            )
        else:
            detailed_answer = str(content_data)

        # Summary prompt
        summary_prompt = f"Summarize the following answer in 2-3 sentences:\n\n{detailed_answer}"
        body_summary = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": summary_prompt}],
            "max_tokens": 300,
            "temperature": 0.5
        }

        summary_response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body_summary)
        )

        summary_body = json.loads(summary_response['body'].read())
        summary_data = summary_body.get("content", "")

        if isinstance(summary_data, list):
            summary = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in summary_data
            )
        else:
            summary = str(summary_data)

        return jsonify({
            "summary": summary.strip(),
            "detailed_answer": detailed_answer.strip()
        })

    except Exception as e:
        print(f"❌ Bedrock error: {e}")
        return jsonify({"error": "Failed to process your question."}), 500


# Debug route to preview chunks
@app.route('/debug_chunks', methods=['GET'])
def debug_chunks():
    try:
        results = collection.get()
        all_docs = results.get("documents", [])
        return jsonify({
            "total_chunks": len(all_docs),
            "chunk_preview": [doc[0][:200] for doc in all_docs if doc]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run server
if __name__ == '__main__':
    app.run(debug=True)
