import os
import time
from dotenv import load_dotenv
from pinecone import ServerlessSpec, Pinecone as pc
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

# Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "smartheal-docs"
PDF_DIR = "docs"

# Chunking Parameters
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 150

def load_pdf_documents(directory: str) -> list[Document]:
    """Load all PDF documents from a specified directory"""
    documents = []
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist. Creating it...")
        os.makedirs(directory)
        return documents
        
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print(f"Loading {filepath}...")
            try:
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    return documents

def create_and_store_embeddings(documents: list[Document], pinecone_index_name: str, embeddings_model: CohereEmbeddings):
    """Create chunks and store embeddings in Pinecone with rate limiting"""
    print("Setting up chunking and embedding storage...")

    # Define splitters
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )

    pine_client = pc(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    if pinecone_index_name not in pine_client.list_indexes().names():
        print("Creating Pinecone index...")
        pine_client.create_index(
            name=pinecone_index_name,
            metric="cosine",
            dimension=1024,  # Cohere embed-english-v3.0 dimension
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
        print(f"Index '{pinecone_index_name}' created successfully!")

    # Initialize Pinecone vector store
    vectorstore = PineconeVectorStore(
        index_name=pinecone_index_name,
        embedding=embeddings_model
    )

    # Split documents into parent chunks first
    parent_docs = parent_splitter.split_documents(documents)
    print(f"Created {len(parent_docs)} parent chunks")

    # Process in small batches with delays
    batch_size = 3
    total_processed = 0

    for i in range(0, len(parent_docs), batch_size):
        batch = parent_docs[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(parent_docs) + batch_size - 1)//batch_size} ({len(batch)} documents)...")
        
        try:
            # Split each parent document into child chunks
            child_docs = []
            for parent_doc in batch:
                child_chunks = child_splitter.split_documents([parent_doc])
                child_docs.extend(child_chunks)
            
            # Add to vectorstore with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore.add_documents(child_docs)
                    total_processed += len(child_docs)
                    print(f"Successfully added {len(child_docs)} child chunks. Total processed: {total_processed}")
                    break
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        wait_time = (attempt + 1) * 45
                        print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error processing batch: {e}")
                        break
            
            # Wait between batches to respect rate limits
            print("Waiting 15 seconds before next batch...")
            time.sleep(15)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue

    print(f"Finished! Total child chunks processed: {total_processed}")
    return vectorstore

def test_retrieval(vectorstore):
    """Test the retrieval system"""
    query = "What should I do if there is muscle pain on my wrists?"
    print(f"\nTesting retrieval with query: '{query}'")
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"--- Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}) ---")
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            print("-" * 50)
    except Exception as e:
        print(f"Error during retrieval test: {e}")

if __name__ == "__main__":
    print("Starting document processing...")
    
    # Initialize embeddings
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    
    # Load documents
    raw_documents = load_pdf_documents(PDF_DIR)

    if not raw_documents:
        print(f"No PDF documents found in '{PDF_DIR}'. Please place your PDF files there.")
    else:
        print(f"Found {len(raw_documents)} documents to process.")
        
        # Process with rate limiting
        vectorstore = create_and_store_embeddings(
            raw_documents, 
            PINECONE_INDEX_NAME, 
            embeddings
        )
        
        # Test retrieval
        test_retrieval(vectorstore)

    print("Process completed!")
