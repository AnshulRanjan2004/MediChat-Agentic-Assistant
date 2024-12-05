import os
import json
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Flush ChromaDB
def flush_chroma_db(persist_directory):
    print("[INFO] Flushing ChromaDB vector store...")
    if os.path.exists(persist_directory):
        for root, dirs, files in os.walk(persist_directory, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        print("[INFO] ChromaDB vector store flushed successfully.")
    else:
        print("[INFO] No existing ChromaDB store found. Starting fresh.")


# Preprocess JSON Files
def preprocess_json_files(json_dir):
    print("[INFO] Starting JSON preprocessing...")
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for idx, file_name in enumerate(os.listdir(json_dir)):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                # Flatten JSON sections
                for section, content in data.items():
                    if isinstance(content, str) and content.strip():
                        # Split into chunks
                        for chunk in splitter.split_text(content):
                            documents.append(
                                Document(page_content=chunk, metadata={
                                    "section": section,
                                    "file": file_name.split(".json")[0]  # Store medicine name as metadata
                                })
                            )
            print(f"[INFO] Processed {file_name} ({idx + 1}/{len(os.listdir(json_dir))})")
    print(f"[INFO] JSON preprocessing completed. Total chunks: {len(documents)}")
    return documents


# Create Vector Store
def create_vector_store(documents, persist_directory):
    print("[INFO] Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    print("[INFO] Vector store created and persisted successfully.")
    return vector_store


if __name__ == "__main__":
    # Configuration
    json_dir = "/Users/ashwin/Desktop/LLM_Hackathon/datasets/microlabs_usa"
    persist_directory = "./chroma_db"

    # Step 1: Flush Vector Store
    flush_chroma_db(persist_directory)

    # Step 2: Preprocess JSON Files
    print("[INFO] Preprocessing JSON files...")
    documents = preprocess_json_files(json_dir)
    print(f"[INFO] Processed {len(documents)} document chunks.")

    # Step 3: Create Vector Store
    print("[INFO] Creating vector store...")
    create_vector_store(documents, persist_directory)
    print("[INFO] Vector store setup complete.")
