"""
RAG-Based Chatbot for Senior Citizens
A Flask web application using LangChain for Retrieval-Augmented Generation
"""

from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import logging

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
import torch


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here-change-in-production')

# Global variables
chat_history = []
chatbot = None
FEEDBACK_FILE = 'feedback.jsonl'


class RAGChatbot:
    """RAG Chatbot with conversation history and context compression"""
    
    def __init__(self, dataset_path, max_history=5):
        self.max_history = max_history
        self.conversation_history = []
        self.dataset_path = dataset_path
        self.setup_model()
    
    def setup_model(self):
        """Initialize the RAG pipeline with retrievers and LLM"""
        try:
            # Get API tokens from environment variables
            cohere_token = os.getenv('COHERE_API_KEY')
            if not cohere_token:
                raise ValueError("COHERE_API_KEY not found in environment variables")
            
            logger.info("Loading documents...")
            # Load and process documents
            documents = []
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
            
            for filename in os.listdir(self.dataset_path):
                if filename.endswith('.txt'):
                    filepath = os.path.join(self.dataset_path, filename)
                    try:
                        loader = TextLoader(filepath, encoding='utf-8')
                        documents.extend(loader.load())
                    except Exception as e:
                        logger.warning(f"Could not load {filename}: {e}")
            
            if not documents:
                raise ValueError(f"No documents loaded from {self.dataset_path}")
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            self.text_splits = text_splitter.split_documents(documents)
            logger.info(f"Created {len(self.text_splits)} text chunks")
            
            # Setup embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5"
            )
            
            # Create vector store
            logger.info("Creating vector store...")
            vectorstore = FAISS.from_documents(self.text_splits, embeddings)
            retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Create BM25 retriever
            logger.info("Creating BM25 retriever...")
            keyword_retriever = BM25Retriever.from_documents(self.text_splits)
            keyword_retriever.k = 5
            
            # Combine retrievers
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[retriever_vectordb, keyword_retriever],
                weights=[0.5, 0.5]
            )
            
            # Setup Cohere reranker
            logger.info("Setting up reranker...")
            compressor = CohereRerank(
                cohere_api_key=cohere_token,
                model="rerank-english-v3.0"
            )
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.ensemble_retriever
            )
            
            # Initialize LLM
            logger.info("Initializing LLM...")
            
            MODEL_PATH = "./models/mistral-7b-instruct-v0.2"
            pipe = pipeline(
                "text-generation",
                model=MODEL_PATH,
                dtype=torch.float16,
                device_map="auto"
            )
            self.model = HuggingFacePipeline(
                pipeline=pipe,
                model_kwargs={
                    "temperature": 0.5,
                    "max_new_tokens": 512
                }
            )
            
            # Create prompt template
            self.template = """
                You are a helpful AI assistant for senior citizens in old age homes.
                Provide compassionate, clear, and accurate information about their physical and mental wellbeing.

                Previous conversation:
                {chat_history}

                Context from knowledge base:
                {context}

                User question:
                {query}

                Assistant:
            """
            
            self.prompt = ChatPromptTemplate.from_template(self.template)
            self.output_parser = StrOutputParser()
            
            logger.info("RAG chatbot setup complete!")
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
    
    def format_chat_history(self):
        """Format recent conversation history"""
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted_history = []
        # Get last N conversations
        recent_history = self.conversation_history[-self.max_history:]
        
        for entry in recent_history:
            formatted_history.append(f"User: {entry['query']}")
            formatted_history.append(f"Assistant: {entry['response']}")
        
        return "\n".join(formatted_history)
    
    def clean_response(self, response):
        """Extract clean response from model output"""
        # Split by common delimiters and get the last meaningful part
        lines = response.strip().split('\n')
        
        # Filter out system messages and empty lines
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('<|') and not line.startswith('</'):
                cleaned_lines.append(line)
        
        # Return the last paragraph or full response
        if cleaned_lines:
            return cleaned_lines[-1] if len(cleaned_lines) > 1 else ' '.join(cleaned_lines)
        
        return response.strip()
    
    def generate_response(self, query):
        """Generate response using RAG pipeline"""
        try:
            # Retrieve relevant documents
            compressed_docs = self.compression_retriever.invoke(query)
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in compressed_docs])
            
            # Build the chain
            chain = (
                {
                    "context": lambda x: context,
                    "chat_history": lambda x: self.format_chat_history(),
                    "query": RunnablePassthrough()
                }
                | self.prompt
                | self.model
                | self.output_parser
            )
            
            # Generate response
            response = chain.invoke(query)
            
            # Clean the response
            cleaned_response = self.clean_response(response)
            
            # Update conversation history
            self.conversation_history.append({
                "query": query,
                "response": cleaned_response
            })
            
            # Limit history size
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
            
            return cleaned_response, compressed_docs
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


def initialize_chatbot():
    """Initialize the chatbot with dataset path from environment"""
    global chatbot
    if chatbot is None:
        dataset_path = os.getenv('DATASET_PATH', 'patientdata/')
        chatbot = RAGChatbot(dataset_path=dataset_path, max_history=5)
    return chatbot


@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html', chat_history=chat_history)


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    global chat_history
    
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Empty query'
            }), 400
        
        # Initialize chatbot if needed
        bot = initialize_chatbot()
        
        # Generate response
        response, docs = bot.generate_response(user_query)
        
        # Create message object
        message_id = len(chat_history)
        new_message = {
            'id': message_id,
            'query': user_query,
            'response': response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        chat_history.append(new_message)
        
        return jsonify({
            'success': True,
            'id': new_message['id'],
            'query': new_message['query'],
            'response': new_message['response'],
            'timestamp': new_message['timestamp']
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/clear', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    global chat_history
    chat_history = []
    
    # Reset chatbot conversation history
    if chatbot is not None:
        chatbot.conversation_history = []
    
    return jsonify({'success': True})


@app.route('/feedback', methods=['POST'])
def save_feedback():
    """Save user feedback"""
    try:
        feedback_data = request.get_json()
        message_id = feedback_data.get('messageId')
        rating = feedback_data.get('rating')
        comment = feedback_data.get('comment', '')
        
        # Find the message
        message = next((msg for msg in chat_history if msg['id'] == message_id), None)
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message not found'
            }), 404
        
        # Create feedback entry
        feedback_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message_id': message_id,
            'query': message['query'],
            'response': message['response'],
            'rating': rating,
            'comment': comment
        }
        
        # Save to file (JSONL format)
        with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        logger.info(f"Feedback saved for message {message_id}")
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'chatbot_initialized': chatbot is not None,
        'messages_count': len(chat_history)
    })


if __name__ == '__main__':
    # Initialize chatbot on startup
    try:
        initialize_chatbot()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        logger.warning("Server starting anyway - chatbot will initialize on first request")
    
    # Run the app
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    app.run(
        debug=debug,
        host='0.0.0.0',
        port=port
    )