import os
import logging
from flask import Flask, render_template, request, jsonify, session
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pinecone import Pinecone as pc

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "smartheal-secret-key-2024")

# Configuration
PINECONE_INDEX_NAME = "smartheal-docs"

def init_vectorstore():
    """Initialize Pinecone vector store"""
    try:
        # Get API keys
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        
        if not cohere_api_key or not pinecone_api_key:
            raise ValueError("API keys not found in environment variables")
        
        # Initialize embeddings
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=cohere_api_key
        )
        
        # Initialize Pinecone client and check if index exists
        pine_client = pc(api_key=pinecone_api_key)
        if PINECONE_INDEX_NAME not in pine_client.list_indexes().names():
            raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Please run store_embed.py first.")
        
        # Create vector store
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        
        return vectorstore
    
    except Exception as e:
        logging.error(f"Error initializing vector store: {str(e)}")
        raise

# Initialize vector store at startup
try:
    app.vectorstore = init_vectorstore()
    logging.info("Vector store initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize vector store: {str(e)}")
    app.vectorstore = None

@app.route('/')
def index():
    """Main page"""
    # Don't reset session on every load - only initialize if needed
    if 'messages' not in session:
        session['messages'] = []
    
    # Check if API keys are available and vector store is ready
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    return render_template('index.html', 
                         has_api_keys=bool(cohere_api_key and pinecone_api_key),
                         vectorstore_ready=app.vectorstore is not None,
                         messages=session.get('messages', []))

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question asking"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        if not app.vectorstore:
            return jsonify({'error': 'Vector store not available. Please ensure embeddings are loaded in Pinecone.'}), 400
        
        # Get API key
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        if not cohere_api_key:
            return jsonify({'error': 'COHERE API key not configured'}), 500
        
        # Create QA chain
        retriever = app.vectorstore.as_retriever(search_kwargs={"k": 4})
        llm = ChatCohere(
            model="command-r-plus",
            cohere_api_key=cohere_api_key,
            temperature=0.1
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )
        
        # Get answer
        response = qa_chain.invoke({"query": question})["result"]
        
        # Update session messages
        if 'messages' not in session:
            session['messages'] = []
        
        session['messages'].append({"role": "user", "content": question})
        session['messages'].append({"role": "assistant", "content": response})
        
        # Keep only last 20 messages to prevent session from growing too large
        if len(session['messages']) > 20:
            session['messages'] = session['messages'][-20:]
        
        session.modified = True
        
        return jsonify({
            'success': True,
            'answer': response,
            'question': question
        })
    
    except Exception as e:
        logging.error(f"Question answering error: {str(e)}")
        return jsonify({'error': f'Error getting answer: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    session['messages'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        
        # Test vector store
        vectorstore_status = "ready" if app.vectorstore else "not_ready"
        
        # Get document count if available
        doc_count = 0
        if app.vectorstore:
            try:
                # Try a simple search to verify the index has documents
                test_results = app.vectorstore.similarity_search("test", k=1)
                doc_count = len(test_results) if test_results else 0
            except:
                doc_count = "unknown"
        
        return jsonify({
            'cohere_api_key': bool(cohere_api_key),
            'pinecone_api_key': bool(pinecone_api_key),
            'vectorstore_status': vectorstore_status,
            'document_count': doc_count,
            'index_name': PINECONE_INDEX_NAME
        })
    
    except Exception as e:
        logging.error(f"Status check error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)