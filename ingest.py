import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_documents():
    print("Loading documents...")
    all_documents = []
    books_dir = "books/"

    for filename in os.listdir(books_dir):
        if filename.endswith(".epub"):
            filepath = os.path.join(books_dir, filename)
            print(f"Loading {filename}...")

            book = epub.read_epub(filepath)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    text = soup.get_text()
                    text = text.strip()
                    if len(text) > 100:
                        doc = Document(
                            page_content=text,
                            metadata={"source": filename}
                        )
                        all_documents.append(doc)

    print(f"Loaded {len(all_documents)} sections")
    return all_documents


def split_documents(documents):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_embeddings_and_save(chunks):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Creating embeddings and saving to FAISS...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    print("Done! FAISS index saved.")


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    create_embeddings_and_save(chunks)
