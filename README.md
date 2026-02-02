# RAG-Based Assistant for Senior Citizens

A Flask-based web application that uses Retrieval-Augmented Generation (RAG) to provide health and wellbeing support for senior citizens in old age homes.

## Features

- 🤖 **AI-Powered Conversations**: Uses LangChain with HuggingFace models for intelligent responses
- 📚 **Knowledge Base**: Retrieves relevant information from patient data using RAG
- 💬 **Conversation History**: Maintains context across multiple exchanges
- 🎯 **Hybrid Retrieval**: Combines vector similarity (FAISS) and keyword search (BM25)
- 🔄 **Re-ranking**: Uses Cohere for improved answer relevance
- 📊 **Feedback System**: Collects user feedback to improve the system
- 🎨 **Modern UI**: Beautiful, responsive interface with gradient design

## Prerequisites

- Python 3.8 or higher
- HuggingFace API token
- Cohere API key
- Patient data in text format

## Installation

### 1. Clone or Download the Project

```bash
# If you have the files, navigate to the project directory
cd rag-assistant
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys
# Use any text editor to edit the file
```

Required environment variables:
- `HUGGINGFACE_API_TOKEN`: Get from https://huggingface.co/settings/tokens
- `COHERE_API_KEY`: Get from https://dashboard.cohere.com/api-keys
- `DATASET_PATH`: Path to your patient data folder (default: `patientdata/`)

### 5. Prepare Dataset

Create a folder named `patientdata/` in the project root and add your text files:

```
rag-assistant/
├── app.py
├── templates/
│   └── index.html
├── patientdata/
│   ├── patient1.txt
│   ├── patient2.txt
│   └── ...
├── requirements.txt
└── .env
```

## Running the Application

### Development Mode

```bash
# Make sure virtual environment is activated
python app.py
```

The application will start at `http://localhost:8000`

### Production Mode

For production deployment:

1. Set `DEBUG=False` in `.env`
2. Use a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Type your health-related question in the input box
3. Click "Send" or press Enter
4. The AI will respond based on the knowledge base
5. Provide feedback using the 👍/👎 buttons
6. Use "Clear Chat" to start a new conversation

## Project Structure

```
rag-assistant/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Frontend template
├── patientdata/          # Dataset folder (create this)
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create from .env.example)
├── .env.example         # Example environment file
├── feedback.jsonl       # Feedback storage (auto-created)
└── README.md            # This file
```

## How It Works

### RAG Pipeline

1. **Document Loading**: Loads text files from the dataset folder
2. **Text Splitting**: Chunks documents into manageable pieces (512 tokens)
3. **Embeddings**: Creates vector representations using HuggingFace embeddings
4. **Vector Store**: Stores embeddings in FAISS for fast retrieval
5. **Hybrid Retrieval**: 
   - Vector similarity search (semantic understanding)
   - BM25 keyword search (exact matches)
6. **Re-ranking**: Cohere re-ranks results for relevance
7. **Generation**: Zephyr-7B model generates contextual responses
8. **History**: Maintains last 5 conversation turns for context

### API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Send message and get response
- `POST /clear` - Clear chat history
- `POST /feedback` - Submit feedback
- `GET /health` - Health check endpoint

## Configuration

### Adjustable Parameters

In `app.py`, you can modify:

```python
# RAGChatbot.__init__
max_history=5  # Number of conversation turns to remember

# Text splitting
chunk_size=512  # Size of text chunks
chunk_overlap=50  # Overlap between chunks

# Retrieval
search_kwargs={"k": 5}  # Number of documents to retrieve
weights=[0.5, 0.5]  # Vector vs BM25 weights

# Model parameters
temperature=0.5  # Response randomness (0-1)
max_new_tokens=512  # Maximum response length
```

## Troubleshooting

### Common Issues

**1. Module not found errors**
```bash
# Make sure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt
```

**2. API key errors**
```bash
# Check .env file has correct keys
# Verify keys are valid on respective platforms
```

**3. No documents loaded**
```bash
# Ensure patientdata/ folder exists
# Check that folder contains .txt files
# Verify DATASET_PATH in .env is correct
```

**4. Memory errors**
```bash
# Reduce chunk_size in app.py
# Limit number of documents
# Use smaller model or reduce max_new_tokens
```

### Debug Mode

Enable debug logging by setting in `.env`:
```
DEBUG=True
```

This will show detailed logs in the console.

## Security Notes

- Never commit `.env` file to version control
- Change `SECRET_KEY` in production
- Use HTTPS in production
- Keep API keys secure
- Validate and sanitize user inputs
- Implement rate limiting for production

## Performance Tips

1. **Dataset Optimization**
   - Use clean, well-formatted text files
   - Remove unnecessary whitespace
   - Limit file sizes to improve loading time

2. **Caching**
   - FAISS index is created on startup
   - Consider saving/loading pre-built index for faster startup

3. **Scaling**
   - Use Gunicorn with multiple workers
   - Deploy on cloud platforms (AWS, GCP, Azure)
   - Consider using Redis for session management

## Contributing

To improve this project:

1. Add more sophisticated prompts
2. Implement user authentication
3. Add conversation export feature
4. Improve error handling
5. Add unit tests
6. Implement analytics dashboard

## License

This project is for educational purposes. Modify as needed for your use case.

## Acknowledgments

- LangChain for the RAG framework
- HuggingFace for models and embeddings
- Cohere for re-ranking capabilities
- Flask for the web framework

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review LangChain documentation
3. Check HuggingFace model cards
4. Review Cohere API documentation

## Future Enhancements

- [ ] Add voice input/output
- [ ] Multi-language support
- [ ] Integration with health monitoring devices
- [ ] Appointment scheduling
- [ ] Medication reminders
- [ ] Emergency contact system
- [ ] Analytics and reporting
- [ ] Mobile app version