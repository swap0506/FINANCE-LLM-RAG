import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

# Load env variables
load_dotenv()

# Use relative path (IMPORTANT for deployment)
DATA_DIR = os.path.join(os.getcwd(), "Data")

def ingest_docs():
    """Load and process financial documents into Qdrant Cloud"""
    try:
        print(f"Loading documents from: {DATA_DIR}")

        # Load PDFs
        loader = DirectoryLoader(
            DATA_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()

        if not documents:
            print("No documents found!")
            return False

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # ✅ CLOUD QDRANT
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60
        )

        collection_name = "financial_docs"

        # Check if collection exists
        collections = client.get_collections().collections
        existing = [c.name for c in collections]

        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance="Cosine")
            )
            print("Created collection")
        else:
            print("Collection already exists (no deletion ✔)")

        # Vector DB
        db = Qdrant(
            client=client,
            embeddings=embeddings,
            collection_name=collection_name
        )

        # Batch insert
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            db.add_documents(batch)
            print(f"Batch {i//batch_size + 1} done")

        print(f"✅ Successfully ingested {len(documents)} documents")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    ingest_docs()