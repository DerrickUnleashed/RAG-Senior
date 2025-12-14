from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Store chat history in memory
chat_history = []

# Load environment variables
load_dotenv()

# Initialize RAG Chatbot
chatbot = None

def initialize_chatbot():
    global chatbot
    if chatbot is None:
        chatbot = RAGChatbot(max_history=5)
    return chatbot

def get_last_paragraph(text):
    lines = text.splitlines()
    # Filter out any empty lines
    non_empty_lines = [line for line in lines if line.strip() != '']
    if non_empty_lines:
        return non_empty_lines[-1]
    return text  # Return original text if no paragraphs found

# Ensure feedback directory exists
FEEDBACK_FILE = 'feedback.txt'

@app.route('/', methods=['GET', 'POST'])
def index():
    global chat_history
    
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            user_query = data.get('query', '')
        else:
            user_query = request.form.get('query', '')
        
        # Initialize chatbot if not already done
        bot = initialize_chatbot()
        
        # Get response from RAG chatbot
        try:
            response, docs = bot.generate_response(user_query)
            response = get_last_paragraph(response)  # Get last paragraph of response
        except Exception as e:
            print(f"Error generating response: {e}")
            response = "I apologize, but I encountered an error processing your request."
        
        # Add the new message pair to chat history
        message_id = len(chat_history)
        new_message = {
            'id': message_id,
            'query': user_query,
            'response': response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        chat_history.append(new_message)
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'id': new_message['id'],
                'query': new_message['query'],
                'response': new_message['response']
            })
            
        return render_template('index.html', chat_history=chat_history)
    
    return render_template('index.html', chat_history=chat_history)

@app.route('/clear', methods=['POST'])
def clear_chat():
    global chat_history
    chat_history = []
    # Reset chatbot conversation history
    if chatbot is not None:
        chatbot.conversation_history = []
    return jsonify(success=True)

@app.route('/feedback', methods=['POST'])
def save_feedback():
    feedback_data = request.json
    message_id = feedback_data.get('messageId')
    rating = feedback_data.get('rating')
    comment = feedback_data.get('comment')
    
    message = next((msg for msg in chat_history if msg['id'] == message_id), None)
    if message:
        feedback_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message_id': message_id,
            'query': message['query'],
            'response': message['response'],
            'rating': rating,
            'comment': comment
        }
        
        with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        return jsonify(success=True)
    
    return jsonify(success=False, error="Message not found"), 404

class RAGChatbot:
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.conversation_history = []
        self.setup_model()
        
    def setup_model(self):
        # Your existing setup code
        HF_token = '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'
        dataset_folder_path=r'C:/Users/arvi0/Downloads/patientdata/'
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token
        
        # Load and process documents
        documents = []
        for file in os.listdir(dataset_folder_path):
            loader = TextLoader(dataset_folder_path+file)
            documents.extend(loader.load())
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self.text_splits = text_splitter.split_documents(documents)
        
        # Setup embeddings and retrievers
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_token,
            model_name='BAAI/bge-base-en-v1.5'
        )
        
        vectorstore = FAISS.from_documents(self.text_splits, embeddings)
        retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        keyword_retriever = BM25Retriever.from_documents(self.text_splits)
        keyword_retriever.k = 5
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_vectordb, keyword_retriever],
            weights=[0.5, 0.5]
        )
        
        # Setup Cohere reranker
        os.environ["COHERE_API_KEY"] = 'VdFqnqPEjEsMzGI9R0AD2uquSjaZu58MpaMFvnfR'
        
        from langchain.llms import HuggingFaceHub
        self.model = HuggingFaceHub(
            repo_id='HuggingFaceH4/zephyr-7b-alpha',
            model_kwargs={"temperature":0.5, "max_new_tokens":512, "max_length":64}
        )
        
        compressor = CohereRerank()
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=self.ensemble_retriever
        )
        
        self.template = """
        <|system|>
        You are an AI Assistant that follows instructions extremely well.
        Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT
        
        Previous conversation:
        {chat_history}
        
        Current context: {context}
        </s>
        <|user|>
        {query}
        </s>
        <|assistant|>
        """
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.output_parser = StrOutputParser()
    
    def format_chat_history(self):
        formatted_history = []
        for i in range(min(len(self.conversation_history), self.max_history)):
            entry = self.conversation_history[-(i+1)]
            formatted_history.insert(0, f"User: {entry['query']}\nAssistant: {entry['response']}")
        return "\n\n".join(formatted_history)
    
    def generate_response(self, query):
        augmented_query = query
        if self.conversation_history:
            recent_context = self.format_chat_history()
            augmented_query = f"{recent_context}\n\nCurrent query: {query}"
        
        compressed_docs = self.compression_retriever.get_relevant_documents(augmented_query)
        
        chain = (
            {
                "context": lambda x: "\n".join(doc.page_content for doc in compressed_docs),
                "chat_history": lambda x: self.format_chat_history(),
                "query": RunnablePassthrough()
            }
            | self.prompt
            | self.model
            | self.output_parser
        )
        
        response = chain.invoke(query)
        
        self.conversation_history.append({
            "query": query,
            "response": response
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        return response, compressed_docs

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')