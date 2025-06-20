from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import random
import os
import re

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BERT-based GoEmotions Model from Hugging Face
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", top_k=5)

# Load Llama 2 model from Ollama
llm = OllamaLLM(model="llama2")

# Initialize FAISS vector store
vector_store = None

# ===================== KNOWLEDGE BASE ===================== #
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(knowledge_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

# Create knowledge base on startup
create_knowledge_base()
def analyze_emotion(text):
    """Detects emotions using a fine-tuned BERT model on GoEmotions"""
    results = emotion_pipeline(text)
    detected_emotions = [result["label"] for result in results[0]]

    # Depression-related emotions
    depression_indicators = {"sadness", "grief", "loneliness", "despair", "hopelessness"}

    is_depressed = any(emotion in depression_indicators for emotion in detected_emotions)
    return detected_emotions, is_depressed

# ===================== NEGATIVE TEST CASE HANDLING ===================== #
def handle_negative_cases(user_message):
    """Handles negative test cases such as empty input, gibberish, offensive language, etc."""
    
    # 1️⃣ **Empty Input**
    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty. Please share your thoughts.")

    # 2️⃣ **Gibberish Text**
    if len(set(user_message)) < 5:  # Checks if input has too few unique characters
        return "I want to understand you better. Can you rephrase that?"

    # 3️⃣ **Profanity / Offensive Language**
    offensive_words = ["stupid", "dumb", "idiot", "useless"]
    if any(word in user_message.lower() for word in offensive_words):
        return "I believe in respectful conversations. Let’s talk about something that helps you feel better."

    # 4️⃣ **Self-Harm & Emergency Support**
    crisis_words = ["suicide", "want to die", "kill myself", "end my life"]
    if any(phrase in user_message.lower() for phrase in crisis_words):
        return "💙 I'm really sorry you're feeling this way. You are not alone. Please reach out to a close friend, family member, or professional. If you're in immediate danger, please call a crisis helpline. 💙"

    # 5️⃣ **Repetitive Messages**
    if user_message.count(user_message.split()[0]) > 3:  # Checks if first word is repeated multiple times
        return "I understand you're trying to express something important. Can you tell me more about how you're feeling?"

    # 6️⃣ **Extremely Long Text**
    if len(user_message) > 500:
        return "That’s a lot to process! Can you summarize your thoughts in a few sentences?"

    # 7️⃣ **Unrelated Questions (Football scores, Weather, etc.)**
    unrelated_topics = ["football", "weather", "lottery", "stocks"]
    if any(topic in user_message.lower() for topic in unrelated_topics):
        return "I'm here to support you emotionally. Want to talk about how you're feeling today?"

    # 8️⃣ **Asking About the Chatbot**
    if "are you human" in user_message.lower() or "who are you" in user_message.lower():
        return "I'm an AI designed to help you process emotions and support your mental well-being."

    return None  # If no negative case is detected, return None

# ===================== FOLLOW-UP QUESTIONS FOR DEPRESSION ===================== #
depression_followup_questions = [
    "Can you share what’s been on your mind lately?",
    "What is something small that made you feel a little better recently?",
    "Have you talked to someone about how you’re feeling?",
    "Would you like me to suggest some self-care activities?",
    "How has your sleep and appetite been lately?",
    "Do you have a safe space where you feel comfortable?",
    "I’m here for you. Would you like some professional support resources?",
    "What’s one thing you enjoy doing, even if just a little?"
]

def get_followup_question():
    """Returns a random follow-up question for users experiencing depression"""
    return random.choice(depression_followup_questions)
class UserMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat(data: UserMessage):
    """Processes user input, detects emotions, and provides responses"""
    global vector_store

    user_message = data.message.strip()

    # Check for negative test cases
    negative_case_response = handle_negative_cases(user_message)
    if negative_case_response:
        return JSONResponse(content={"response": negative_case_response})

    # Perform emotion and depression analysis
    detected_emotions, is_depressed = analyze_emotion(user_message)

    # Retrieve relevant psychology-based context using FAISS
    retrieved_docs = vector_store.similarity_search(user_message, k=2)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    response_tone = "I hear you. Let's explore this together with an open heart."

    # If depression is detected, modify response to provide support & ask follow-up question
    followup_question = None
    if is_depressed:
        response_tone += "\n\n💙 I sense that you might be feeling really low. Remember, you are not alone. It's always okay to seek support from close ones or mental health professionals. 💙"
        followup_question = get_followup_question()

    try:
        answer = llm.invoke(response_tone)

        response_data = {
            "emotions": detected_emotions,
            "is_depressed": is_depressed,
            "response": answer + '<br/><br/> <p style="color:red">Your emotions are : '+', '.join(detected_emotions)+'</p>',
            "followup_question":followup_question,
            "response_tone":response_tone
        }

        if followup_question:
            response_data["followup_question"] = followup_question

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
