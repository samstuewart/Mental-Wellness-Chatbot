import React from "react";
import DOMPurify from "dompurify"; // Import DOMPurify for sanitization

const Message = ({ text, sender, sentiment }) => {
    const happyEmotions = ['joy', 'love', 'excitement', 'happy'];

    // Sentiment analysis function
    const checkhappyRes = (sentiment = [], happyEmotions = []) => {
        if (!Array.isArray(sentiment) || !Array.isArray(happyEmotions)) {
            return '';
        }
        
        if (sentiment.length === 0 || happyEmotions.length === 0) {
            return 'negative'; // or return some other message or handle differently
        }
    
        if (sentiment.some(item => happyEmotions.includes(item))) {
            return 'positive';
        } else {
            return 'negative';
        }
    };

    // Get sentiment color
    const getSentimentColor = (checkhappyRes) => {
        switch (checkhappyRes) {
            case "positive":
                return "green";
            case "negative":
                return "red";
            case "neutral":
                return "gray";
            default:
                return "black";
        }
    };

    // Sanitize the text to avoid XSS attacks (if necessary)
    const sanitizedText = DOMPurify.sanitize(text);

    return (
        <div className={`message ${sender}`}>
            <p style={{ color: getSentimentColor(checkhappyRes(sentiment, happyEmotions)) }} 
               dangerouslySetInnerHTML={{ __html: sanitizedText }} />
        </div>
    );
};

export default Message;
