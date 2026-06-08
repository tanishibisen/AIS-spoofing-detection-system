from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

CHROMA_PATH = "chroma_db"
KNOWLEDGE_BASE_PATH = Path("knowledge_base")

def load_pdfs():
    docs = []
    pdf_files = list(KNOWLEDGE_BASE_PATH.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files:")
    
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            docs.extend(pages)
            print(f"  → {len(pages)} pages loaded")
        except Exception as e:
            print(f"  → Error loading {pdf_path.name}: {e}")
    
    print(f"\nTotal pages loaded: {len(docs)}")
    return docs

def build_knowledge_base():
    print("=" * 50)
    print("BUILDING AIS KNOWLEDGE BASE FROM PDFs")
    print("=" * 50)

    docs = load_pdfs()

    if not docs:
        print("No documents loaded!")
        return None

    print("\nSplitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Building vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"\nKnowledge base built with {vectorstore._collection.count()} vectors")
    print(f"Saved to {CHROMA_PATH}/")
    return vectorstore

if __name__ == "__main__":
    build_knowledge_base()