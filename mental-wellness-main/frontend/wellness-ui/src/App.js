import React from "react";
import Chat from "./components/Chat"; // ✅ Correct Import Path
import "./App.css"; // Import global styles

const App = () => {
    return (
        <div className="app">
            <h1>AI-Powered Mental Wellness Companion By <br/>AI-Bots</h1>
            <Chat />
        </div>
    );
};

export default App;