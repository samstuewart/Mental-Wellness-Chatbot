import React, { useState } from "react";
import axios from "axios";
import Message from "./Message"; // ✅ Correct Import Path
import "../Chat.css"; // Import custom styling

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);

    const sendMessage = async () => {
        if (!input.trim()) return;

        const newMessage = { text: input, sender: "user" };
        setMessages([...messages, newMessage]);
        setInput("");
        setLoading(true);

        try {
            const response = await axios.post("http://127.0.0.1:8000/chat", { message: input });
            const { response: botResponse, emotions } = response.data;
            
            const botMessage = { text: botResponse, sender: "bot", emotions };
            setMessages([...messages, newMessage, botMessage]);
        } catch (error) {
            console.error("Error sending message:", error);
            const errorMessage = { text: "Failed to get response. Try again.", sender: "bot", sentiment: "error" };
            setMessages([...messages, newMessage, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="chat-container">
            <div className="chat-box">
                {messages.map((msg, index) => (
                    <Message key={index} text={msg.text} sender={msg.sender} sentiment={msg} />
                ))}
                {loading && <p className="loading">Processing your request...</p>}
            </div>
            <div className="input-box">
                <input 
                    type="text" 
                    placeholder="Type a message..." 
                    value={input} 
                    onChange={(e) => setInput(e.target.value)} 
                    onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                />
                <button onClick={sendMessage}>Send</button>
            </div>
        </div>
    );
};

export default Chat;