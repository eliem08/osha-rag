from flask import Flask, request, jsonify, render_template
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os
import uuid
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

app = Flask(__name__)

# Ensure the OpenAI API key is set
if 'OPENAI_API_KEY' not in os.environ:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

# Load the FAISS index and embeddings once at startup
embeddings = OpenAIEmbeddings()
faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = faiss_index.as_retriever()

# Initialize the LLM
llm = OpenAI()

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# In-memory data store for conversations
conversations = {}

@app.route('/')
def index():
    # Render the index page
    return render_template('index.html')

@app.route('/chat')
def chat():
    # Render the chat page
    return render_template('chat.html')

@app.route('/conversations', methods=['GET'])
def get_conversations():
    # Return the list of conversation IDs
    convo_list = [{'id': cid, 'name': f'Conversation {i+1}'} for i, cid in enumerate(conversations.keys())]
    return jsonify(convo_list)

@app.route('/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    # Return the messages in the conversation
    if conversation_id in conversations:
        return jsonify(conversations[conversation_id])
    else:
        return jsonify({'error': 'Conversation not found.'}), 404

@app.route('/conversation', methods=['POST'])
def create_conversation():
    # Create a new conversation
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = []
    return jsonify({'conversation_id': conversation_id})

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    conversation_id = data.get('conversation_id', '')

    if not user_query:
        return jsonify({'error': 'No query provided.'}), 400

    if not conversation_id:
        return jsonify({'error': 'No conversation ID provided.'}), 400

    try:
        # Append user's message to the conversation
        conversations.setdefault(conversation_id, []).append({'sender': 'user', 'text': user_query})

        # Generate assistant's response
        result = qa_chain({"query": user_query})
        answer = result["result"]
        source_documents = result["source_documents"]

        # Prepare sources
        sources = []
        for doc in source_documents:
            url = doc.metadata.get('url', 'N/A')
            paragraph = doc.metadata.get('paragraph', 'N/A')
            sources.append({
                "url": url,
                "paragraph": paragraph,
                "snippet": doc.page_content[:200]
            })

        # Append assistant's message to the conversation
        conversations[conversation_id].append({'sender': 'assistant', 'text': answer})

        response = {
            'answer': answer,
            'sources': sources
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
