# SmartHeal - Document Q&A Chatbot 🏥💬

A sophisticated document-based question-answering chatbot that allows users to interact with their document knowledge base using Retrieval-Augmented Generation (RAG). Built with Flask, Pinecone, Cohere, and powered by advanced embeddings.

## 🌟 Features

- **Document Embedding & Storage**: Automatically processes PDF documents and stores embeddings in Pinecone
- **Intelligent Chunking**: Uses hierarchical document splitting for optimal retrieval
- **RAG-Powered Responses**: Combines document retrieval with Cohere's language model for accurate answers
- **Real-time Chat Interface**: Clean, responsive web interface for seamless conversations
- **Session Management**: Persistent chat history across page refreshes
- **Audio Inpuy**: Audio input avaiable for English, Hindi and Bengali.

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Vector Database**: Pinecone
- **Embeddings**: Cohere
- **Language Model**: Cohere
- **Document Processing**: LangChain + PyPDF
- **Frontend**: Bootstrap 5 + Vanilla JavaScript
- **Speech-to-Text**: OpenAI Whisper
- **Speaker Diarization**: Pyannote Audio
- **Audio Processing**: FFmpeg

## 📋 Prerequisites

- Python 3.9+
- Pinecone account and API key
- Cohere account and API key
- HuggingFace account and token
- FFmpeg (for audio conversion)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/cr7ritesh/smartheal-chatbot.git
cd Runverve
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

**Windows:**

🎬 [Install FFmpeg on Windows](https://www.youtube.com/watch?v=SG1Fc5QB8RE) by TechwithMonir

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```


### 3. Environment Setup

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

### 4. Prepare Your Documents

- Create a `docs` folder in the project root
- Place your PDF files in the `docs` folder

### 5. Process Documents & Create Embeddings

```bash
python store_embed.py
```

This will:

- Load all PDFs from the folder
- Create intelligent chunks using hierarchical splitting
- Generate embeddings using Cohere
- Store everything in Pinecone

### 6. Launch the Web Application

```bash
python app.py
```

Visit `http://localhost:5000` to start chatting with your documents!

## 🔧 Usage

### Processing New Documents

1. Add PDF files to the `documents` folder
2. Run `python store_embed.py` to update embeddings
3. The web app will automatically use the updated knowledge base

### Chatting with Documents

1. Open the web interface at `http://localhost:5000`
2. Ensure the system shows "Documents loaded and ready"
3. Type your questions in the chat interface
4. Get AI-powered responses based on your document content
