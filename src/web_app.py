"""
FastAPI Web Application for PubMed RAG

Modern web interface replacing the terminal UI.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.rag_engine import PubMedRAG
from core.vector_db import ChromaDBManager
from chat.session_manager import SessionManager


# Pydantic models for API
class TopicRequest(BaseModel):
    topic: str
    email: str

class QuestionRequest(BaseModel):
    question: str
    topic: str

class SessionResponse(BaseModel):
    id: str
    name: str
    topic: str
    created_at: str
    question_count: int


# FastAPI app
app = FastAPI(title="PubMed RAG", description="Medical Research Assistant")

# Initialize core components
db_manager = ChromaDBManager()
rag_engine = PubMedRAG(db_manager)
session_manager = SessionManager(openai_client=rag_engine.openai_client)

# Global state
current_sessions: Dict[str, Any] = {}


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, websocket: WebSocket, message: dict):
        await websocket.send_text(json.dumps(message))


manager = ConnectionManager()


# API Routes
@app.get("/")
async def get_index():
    """Serve the main web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PubMed Medical AI Assistant</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <div id="app">
            <!-- Sidebar -->
            <div id="sidebar" class="sidebar collapsed">
                <div class="sidebar-header">
                    <h3>Sessions</h3>
                    <button id="toggleSidebar" class="toggle-btn">←</button>
                </div>
                <div id="sessionList" class="session-list">
                    <!-- Sessions will be loaded here -->
                </div>
            </div>

            <!-- Main Content -->
            <div id="mainContent" class="main-content sidebar-collapsed">
                <!-- Header -->
                <div class="header">
                    <button id="hamburgerBtn" class="hamburger-btn">
                        <span></span>
                        <span></span>
                        <span></span>
                    </button>
                    <h1>PubMed Medical AI Assistant</h1>
                    <div id="topicIndicator" class="topic-indicator" style="display: none;">
                        No topic selected
                    </div>
                </div>

                <!-- Welcome/Setup Area -->
                <div id="welcomeContainer" class="welcome-container">
                    <div class="welcome-content">
                        <h2>Welcome to PubMed Research Assistant</h2>
                        <p>Get instant insights from the latest medical research using AI-powered literature review.</p>
                        
                        <div id="emailStep" class="setup-step">
                            <h3>Step 1: Enter Your Email</h3>
                            <p>Required for accessing PubMed abstracts via NCBI Entrez API</p>
                            <div class="input-group">
                                <input type="email" id="welcomeEmailInput" placeholder="your.email@example.com">
                                <button id="emailNextBtn">Next</button>
                            </div>
                        </div>

                        <div id="topicStep" class="setup-step" style="display: none;">
                            <h3>Step 2: Choose Your Research Topic</h3>
                            <p>What medical topic would you like to explore?</p>
                            <div class="input-group">
                                <input type="text" id="welcomeTopicInput" placeholder="e.g., diabetes treatment, cancer immunotherapy">
                                <button id="topicStartBtn">Start Research</button>
                            </div>
                            <div class="topic-examples">
                                <p>Popular topics:</p>
                                <span class="topic-tag" data-topic="diabetes treatment">Diabetes Treatment</span>
                                <span class="topic-tag" data-topic="cancer immunotherapy">Cancer Immunotherapy</span>
                                <span class="topic-tag" data-topic="alzheimer disease">Alzheimer's Disease</span>
                                <span class="topic-tag" data-topic="covid-19">COVID-19</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Chat Container (initially hidden) -->
                <div id="chatContainer" class="chat-container" style="display: none;">
                    <div id="messages" class="messages">
                        <!-- Messages will appear here -->
                    </div>
                </div>

                <!-- Input Area (initially hidden) -->
                <div id="inputArea" class="input-area" style="display: none;">
                    <div class="input-container">
                        <input type="text" id="messageInput" placeholder="Ask about medical research...">
                        <button id="sendButton">Send</button>
                    </div>
                </div>

                <!-- Floating Action Buttons -->
                <div class="fab-container">
                    <button id="chunksBtn" class="fab" title="View Chunks" disabled>📄</button>
                    <button id="metricsBtn" class="fab" title="Data Metrics" disabled>📊</button>
                    <button id="clearBtn" class="fab" title="Clear Data" disabled>🗑️</button>
                    <button id="newSessionBtn" class="fab" title="New Session">➕</button>
                </div>
            </div>
        </div>

        <!-- Modals -->

        <div id="chunksModal" class="modal">
            <div class="modal-content large">
                <div class="modal-header">
                    <h2>Retrieved Chunks</h2>
                    <button id="closeChunks" class="close-btn">×</button>
                </div>
                <div id="chunksContent" class="chunks-content">
                    <!-- Chunks will be displayed here -->
                </div>
            </div>
        </div>

        <div id="metricsModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Data Loading Metrics</h2>
                    <button id="closeMetrics" class="close-btn">×</button>
                </div>
                <div id="metricsContent" class="metrics-content">
                    <!-- Metrics will be displayed here -->
                </div>
            </div>
        </div>

        <script src="/static/script.js"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/sessions")
async def get_sessions():
    """Get all available sessions"""
    sessions = session_manager.list_sessions()
    return [SessionResponse(**session) for session in sessions]


@app.post("/api/sessions")
async def create_session(request: TopicRequest):
    """Create a new session and load PubMed data"""
    try:
        # Save email
        session_manager.save_email(request.email)
        
        # Create session
        session = session_manager.create_session(request.topic)
        
        # Process PubMed data
        result = rag_engine.process_pubmed_data(request.email, request.topic)
        
        return {
            "session_id": session.id,
            "session_name": session.name,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def load_session(session_id: str):
    """Load a specific session"""
    session = session_manager.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get database stats
    stats = db_manager.get_stats()
    topic_collection = session.topic.lower().replace(" ", "_")[:50]
    existing_chunks = stats.get(topic_collection, 0)
    
    return {
        "session": session.to_dict(),
        "existing_chunks": existing_chunks
    }


@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer"""
    try:
        answer, retrieval_info = rag_engine.answer_question(
            request.topic,
            request.question,
            k=10
        )
        
        # Save to session
        session_manager.add_conversation_to_current(
            request.question, answer, retrieval_info
        )
        
        return {
            "answer": answer,
            "retrieval_info": retrieval_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chunks/{topic}")
async def get_chunks(topic: str):
    """Get the last retrieved chunks for a topic"""
    # This would need to be stored somewhere accessible
    # For now, return empty - we'll handle this via WebSocket
    return {"chunks": [], "distances": []}


@app.delete("/api/data/{topic}")
async def clear_topic_data(topic: str):
    """Clear data for a specific topic"""
    success = db_manager.clear_topic(topic)
    if success:
        return {"message": f"Cleared data for topic: {topic}"}
    else:
        raise HTTPException(status_code=404, detail="No data found for topic")


@app.delete("/api/data")
async def clear_all_data():
    """Clear all data"""
    db_manager.clear_all()
    return {"message": "All data cleared"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "question":
                # Process question and stream response
                topic = message["topic"]
                question = message["question"]
                
                # Get answer
                answer, retrieval_info = rag_engine.answer_question(topic, question, k=10)
                
                # Stream response in chunks
                words = answer.split()
                for i in range(0, len(words), 3):  # Send 3 words at a time
                    chunk = " ".join(words[i:i+3])
                    await manager.send_message(websocket, {
                        "type": "answer_chunk",
                        "chunk": chunk + " "
                    })
                    await asyncio.sleep(0.05)  # Small delay for streaming effect
                
                # Send completion message
                await manager.send_message(websocket, {
                    "type": "answer_complete",
                    "retrieval_info": retrieval_info
                })
                
                # Save to session
                session_manager.add_conversation_to_current(question, answer, retrieval_info)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 