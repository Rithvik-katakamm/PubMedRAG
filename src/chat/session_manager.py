"""
Chat Session Manager

Handles session creation, loading, saving, and auto-naming.
Sessions are stored as JSON files with conversation history.
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import threading
from openai import OpenAI


class ChatSession:
    """Represents a single chat session with conversation history."""
    
    def __init__(self, topic: str, session_id: Optional[str] = None):
        self.id = session_id or str(uuid.uuid4())
        self.topic = topic
        self.name = f"{topic} Session - {datetime.now().strftime('%b %d, %I:%M %p')}"
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.conversations: List[Dict[str, Any]] = []
        self.auto_named = False
    
    def add_conversation(self, question: str, answer: str, retrieval_info: Dict[str, Any]):
        """Add a Q&A pair to the conversation history."""
        self.conversations.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "chunks_retrieved": retrieval_info.get("metrics", {}).get("chunks_retrieved", 0),
            "retrieval_time_ms": retrieval_info.get("metrics", {}).get("total_time_ms", 0),
            "tokens_used": retrieval_info.get("token_usage", {}).get("total_tokens", 0)
        })
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "topic": self.topic,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "auto_named": self.auto_named,
            "question_count": len(self.conversations),
            "conversations": self.conversations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create session from dictionary."""
        session = cls(topic=data["topic"], session_id=data["id"])
        session.name = data["name"]
        session.created_at = data["created_at"]
        session.updated_at = data["updated_at"]
        session.auto_named = data.get("auto_named", False)
        session.conversations = data["conversations"]
        return session


class SessionManager:
    """Manages all chat sessions with auto-save and auto-naming."""
    
    def __init__(self, sessions_dir: str = "./data/sessions", openai_client: Optional[OpenAI] = None):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[ChatSession] = None
        self.openai_client = openai_client or OpenAI()
    
    def create_session(self, topic: str) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(topic)
        self.current_session = session
        self.save_session(session)
        return session
    
    def save_session(self, session: Optional[ChatSession] = None):
        """Save session to JSON file."""
        if session is None:
            session = self.current_session
        
        if session is None:
            return
        
        filename = f"session_{session.id}.json"
        filepath = self.sessions_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load a session by ID."""
        filename = f"session_{session_id}.json"
        filepath = self.sessions_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        session = ChatSession.from_dict(data)
        self.current_session = session
        return session
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions with summary info."""
        sessions = []
        
        for filepath in sorted(self.sessions_dir.glob("session_*.json"), reverse=True):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract summary info
            sessions.append({
                "id": data["id"],
                "name": data["name"],
                "topic": data["topic"],
                "created_at": data["created_at"],
                "question_count": data.get("question_count", len(data.get("conversations", [])))
            })
        
        return sessions
    
    def auto_generate_name(self, session: ChatSession, first_question: str):
        """Generate a session name based on the first question (runs in background)."""
        def generate_name():
            try:
                prompt = f"""Generate a concise 3-5 word session name for this medical research question.
The name should capture the main topic being researched.

Question: {first_question}

Examples:
- "What are the latest insulin pump technologies?" → "Insulin Pump Technologies"
- "How effective is immunotherapy for lung cancer?" → "Lung Cancer Immunotherapy"
- "Side effects of ACE inhibitors in elderly?" → "ACE Inhibitors Elderly Effects"

Session name (3-5 words only):"""

                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=20
                )
                
                new_name = response.choices[0].message.content.strip()
                # Clean up the name
                new_name = new_name.replace('"', '').replace("'", "").strip()
                
                # Update session name
                session.name = new_name
                session.auto_named = True
                self.save_session(session)
                
            except Exception as e:
                # If naming fails, just keep the default name
                pass
        
        # Run in background thread to not block main flow
        import threading
        thread = threading.Thread(target=generate_name, daemon=True)
        thread.start()
    
    def get_current_session(self) -> Optional[ChatSession]:
        """Get the current active session."""
        return self.current_session
    
    def add_conversation_to_current(self, question: str, answer: str, retrieval_info: Dict[str, Any]):
        """Add a conversation to the current session and auto-save."""
        if self.current_session is None:
            return
        
        self.current_session.add_conversation(question, answer, retrieval_info)
        
        # Auto-generate name on first question
        if len(self.current_session.conversations) == 1 and not self.current_session.auto_named:
            self.auto_generate_name(self.current_session, question)
        
        # Auto-save after each conversation
        self.save_session()
    
    def get_config_path(self) -> Path:
        """Get path to config file."""
        return self.sessions_dir.parent / "config.json"
    
    def save_email(self, email: str):
        """Save user email to config."""
        config_path = self.get_config_path()
        config = {}
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        config["email"] = email
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_email(self) -> Optional[str]:
        """Load saved email from config."""
        config_path = self.get_config_path()
        
        if not config_path.exists():
            return None
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config.get("email")