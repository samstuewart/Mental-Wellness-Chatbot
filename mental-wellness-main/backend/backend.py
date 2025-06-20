    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from textblob import TextBlob
    from dotenv import load_dotenv
    from langchain_ollama import OllamaLLM
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings  # Change this for real embeddings if needed

    # Load environment variables
    load_dotenv()

    # Initialize Flask App
    app = Flask(__name__)
    CORS(app)

    # Load Llama 2 model from Ollama
    llm = OllamaLLM(model="llama2")

    # Initialize FAISS vector store
    vector_store = None

    # ===================== KNOWLEDGE BASE (Psychologist-Style) ===================== #

    def create_knowledge_base():
        """Creates a FAISS vector store with predefined psychology-based responses"""
        global vector_store
        knowledge_text = """
        - Feeling sad is completely normal. It's okay to take time for yourself.
        - If you're overwhelmed, take a deep breath and allow yourself to relax.
        - You're stronger than you think, and your feelings are valid.
        - Sometimes, expressing emotions can be the best way to process them.
        - If you're feeling anxious, try grounding techniques such as focusing on your breath or listing things you see.
        - You're not alone. Even when things feel dark, there is light ahead.
        - Self-care is important. It's okay to rest when you need to.
        - If you're happy, embrace the moment and share your joy with others.
        """
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_text(knowledge_text)

        # Generate embeddings (using fake embeddings for simplicity)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

        # Store embeddings in FAISS
        vector_store = FAISS.from_texts(chunks, embeddings)

    # Create knowledge base on startup
    create_knowledge_base()

    # ===================== SENTIMENT ANALYSIS ===================== #

    def analyze_sentiment(text):
        """Returns the sentiment type based on TextBlob's polarity score"""
        sentiment_score = TextBlob(text).sentiment.polarity
        if sentiment_score > 0.2:
            return "positive"
        elif sentiment_score < -0.2:
            return "negative"
        else:
            return "neutral"

    # ===================== EMPATHETIC AI CHATBOT ===================== #

    @app.route("/chat", methods=["POST"])
    def chat():
        """Processes user input, detects sentiment, and generates an empathetic response"""
        global vector_store

        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Perform sentiment analysis
        sentiment = analyze_sentiment(user_message)

        # Retrieve relevant psychology-based context using FAISS
        retrieved_docs = vector_store.similarity_search(user_message, k=2)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Adjust response tone based on sentiment
        if sentiment == "positive":
            response_tone = "I'm so happy to hear that! Here's something to celebrate your joy:"
        elif sentiment == "negative":
            response_tone = "I'm really sorry you're feeling this way. You're not alone, and I'm here to listen:"
        else:
            response_tone = "I hear you. Let's explore this together with an open heart:"

        # Create prompt for AI (Psychologist-style chat)
        prompt_text = f"""
        {response_tone}

        Empathetic Advice:
        {context}

        User: {user_message}
        AI:
        """

        try:
            # Generate response using Llama 2 (via Ollama)
            answer = llm.invoke(prompt_text)
            return jsonify({"sentiment": sentiment, "response": answer})
        except Exception as e:
            return jsonify({"error": f"Failed to generate response: {str(e)}"})

    if __name__ == "__main__":
        app.run(debug=True)