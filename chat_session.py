# chat_session.py
import uuid                               # <-- add this line
from typing import List, Dict, Any
from db.db_utils import log_conversation

class ChatSession:
    def __init__(self, rag, email: str, topic: str):
        self.id     = str(uuid.uuid4())
        self.email  = email
        self.topic  = topic
        self.rag    = rag
        self.turns: List[Dict[str, Any]] = [
            {"role": "system",
             "content": "You are a helpful medical research assistant."}
        ]

    # ---------- public API ----------
    def ask(self, user_q: str, k: int = 10, temperature: float = 0.2) -> str:
        """Append user question, call RAG, append assistant answer, return answer."""
        self.turns.append({"role": "user", "content": user_q})

        answer, retrieval = self.rag.answer_question(
            user_q, k=k, temperature=temperature
        )

        self.turns.append({
            "role": "assistant",
            "content": answer,
            "pubmed_ids": retrieval["ids"][0],
            "chunk_ids":  retrieval["ids"][0],
            "distances":  retrieval["distances"][0],
        })

        return answer

    def save(self):
        log_conversation(self.email, self.topic,
                         {"id": self.id, "turns": self.turns})