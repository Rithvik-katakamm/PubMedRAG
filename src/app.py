"""
PubMed RAG Application

Main entry point with simplified flow and clean UI.
"""

import sys
import signal
import time
from typing import Optional, Dict, Any
from rich.prompt import Prompt

from core.rag_engine import PubMedRAG
from core.vector_db import ChromaDBManager
from chat.session_manager import SessionManager
from ui.terminal_chat import TerminalChat


class PubMedRAGApp:
    """Main application class with simplified flow."""
    
    def __init__(self):
        self.ui = TerminalChat()
        self.db_manager = ChromaDBManager()
        self.rag_engine = PubMedRAG(self.db_manager)
        self.session_manager = SessionManager(openai_client=self.rag_engine.openai_client)
        self.current_topic = None
        self.email = None
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        self.ui.show_goodbye()
        sys.exit(0)
    
    def run(self):
        """Main application loop with simplified flow."""
        try:
            # Show welcome
            self.ui.show_welcome()
            
            # Get email (saved for future)
            self.email = self._get_email()
            
            # Show sessions and get choice
            self._handle_session_selection()
            
            # Main chat loop
            self._chat_loop()
            
        except KeyboardInterrupt:
            self.ui.show_goodbye()
        except Exception as e:
            self.ui.show_error("An unexpected error occurred", str(e))
    
    def _get_email(self) -> str:
        """Get email from saved config or user input."""
        saved_email = self.session_manager.load_email()
        
        if saved_email:
            self.ui.console.print(f"[dim white]Using saved email: {saved_email}[/dim white]\n")
            return saved_email
        else:
            email = self.ui.get_email_input()
            self.session_manager.save_email(email)
            return email
    
    def _handle_session_selection(self):
        """Handle session selection with simplified flow."""
        sessions = self.session_manager.list_sessions()
        
        # Show sessions if any exist
        if sessions:
            self.ui.show_session_list(sessions)
            choice = self.ui.get_session_choice()
            
            # Try to load session by number
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(sessions):
                    session = self.session_manager.load_session(sessions[index]["id"])
                    if session:
                        self.current_topic = session.topic
                        self.ui.clear_screen()
                        self.ui.show_header(session.name, session.topic)
                        
                        # Don't reload data - just show what we have
                        stats = self.db_manager.get_stats()
                        topic_collection = session.topic.lower().replace(" ", "_")[:50]
                        existing_chunks = stats.get(topic_collection, 0)
                        
                        if existing_chunks > 0:
                            self.ui.console.print(f"\n[green]✓[/green] Using existing data: {existing_chunks} chunks\n")
                        
                        # Show last few conversations
                        if session.conversations:
                            self.ui.console.print("[dim white]Recent conversations:[/dim white]\n")
                            for conv in session.conversations[-2:]:
                                q = conv["question"]
                                a = conv["answer"]
                                if len(a) > 200:
                                    a = a[:200] + "..."
                                self.ui.console.print(f"[green]>[/green] {q}")
                                self.ui.console.print(f"{a}\n")
                        return
        
        # No session selected or no sessions - start new
        self._start_new_session()
    
    def _start_new_session(self):
        """Start a new research session."""
        topic = self.ui.get_topic_input()
        self.current_topic = topic
        
        # Create session
        session = self.session_manager.create_session(topic)
        
        # Show header
        self.ui.clear_screen()
        self.ui.show_header(session.name, topic)
        
        # Check if we already have data for this topic
        stats = self.db_manager.get_stats()
        topic_collection = topic.lower().replace(" ", "_")[:50]
        
        existing_chunks = 0
        for collection_name, count in stats.items():
            if collection_name == topic_collection:
                existing_chunks = count
                break
        
        if existing_chunks > 0:
            self.ui.console.print(f"\n[green]✓[/green] Found existing data: {existing_chunks} chunks\n")
        else:
            # Load PubMed data
            self._load_pubmed_data(topic)
    
    def _load_pubmed_data(self, topic: str):
        """Load PubMed data with progress display."""
        self.ui.console.print(f"\n[white]Searching PubMed for '{topic}'...[/white]\n")
        
        # Track each step
        start_total = time.time()
        
        # Process data
        result = self.rag_engine.process_pubmed_data(self.email, topic)
        
        if "error" in result:
            self.ui.show_error(result["error"])
            return
        
        # Show individual step timings
        metrics = result.get("metrics", {})
        if metrics:
            self.ui.show_data_loading_progress("Fetching from PubMed", metrics.get("fetch_time_ms"))
            self.ui.show_data_loading_progress("Chunking text", metrics.get("chunking_time_ms"))
            self.ui.show_data_loading_progress("Generating embeddings", metrics.get("embedding_time_ms"))
            self.ui.show_data_loading_progress("Storing in database", metrics.get("storage_time_ms"))
        
        # Show summary
        total_time = int((time.time() - start_total) * 1000)
        self.ui.console.print(
            f"\n[green]✓[/green] Loaded {result['abstracts_count']} abstracts "
            f"({result['chunk_count']} chunks) | Total: {self.ui.format_time(total_time)}\n"
        )
    
    def _chat_loop(self):
        """Main chat interaction loop."""
        self.ui.console.print("[white]Ready for your questions![/white]\n")
        
        while True:
            try:
                # Get user input
                question = self.ui.get_user_question()
                
                # Check for exit
                if question.lower() in ["exit", "quit", "bye", "/exit", "/quit"]:
                    break
                
                # Handle commands
                if question.startswith("/"):
                    if self._handle_command(question):
                        continue
                
                # Handle shortcuts
                if question.lower() == "c":
                    self.ui.show_chunks()
                    continue
                
                if not question.strip():
                    continue
                
                # Process the question
                self._process_question(question)
                
                self.ui.console.print()  # Spacing between Q&As
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.ui.show_error("Error processing question", str(e))
        
        self.ui.show_goodbye()
    
    def _handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if command was handled."""
        parts = command.strip().split()
        cmd = parts[0].lower()
        
        if cmd == "/chunks":
            self.ui.show_chunks()
            return True
        
        elif cmd == "/clear":
            if len(parts) > 1 and parts[1] == "all":
                self.db_manager.clear_all()
                self.ui.show_success("All database collections cleared")
            elif self.current_topic:
                if self.db_manager.clear_topic(self.current_topic):
                    self.ui.show_success(f"Cleared database for topic: {self.current_topic}")
                    # Reload data
                    self._load_pubmed_data(self.current_topic)
                else:
                    self.ui.show_error(f"No data found for topic: {self.current_topic}")
            return True
        
        elif cmd == "/new":
            self._start_new_session()
            return True
        
        elif cmd == "/sessions":
            sessions = self.session_manager.list_sessions()
            if sessions:
                self.ui.show_session_list(sessions)
                self.ui.console.print("[dim]Press Enter to continue[/dim]")
                input()
            else:
                self.ui.console.print("[yellow]No sessions found[/yellow]")
            return True
        
        elif cmd == "/help":
            self.ui.show_help()
            return True
        
        elif cmd in ["/exit", "/quit"]:
            return False
        
        else:
            self.ui.show_error(f"Unknown command: {cmd}. Type /help for available commands.")
            return True
    
    def _process_question(self, question: str):
        """Process a user question through the RAG pipeline."""
        # Show inline progress
        start_time = time.time()
        
        # Get answer
        answer, retrieval_info = self.rag_engine.answer_question(
            self.current_topic,
            question,
            k=10
        )
        
        # Show timing feedback inline
        if "metrics" in retrieval_info:
            metrics = retrieval_info["metrics"]
            if "retrieval_time_ms" in metrics:
                self.ui.console.print(f"[yellow]Retrieving chunks... {self.ui.format_time(metrics['retrieval_time_ms'])}[/yellow]")
            if "generation_time_ms" in metrics:
                self.ui.console.print(f"[yellow]Generating response... {self.ui.format_time(metrics['generation_time_ms'])}[/yellow]")
        
        # Store chunks for viewing
        if retrieval_info.get("documents") and retrieval_info.get("distances"):
            self.ui.store_chunks(
                retrieval_info["documents"][0],
                retrieval_info["distances"][0]
            )
        
        # Show answer with metrics
        total_time = retrieval_info.get("metrics", {}).get("total_rag_time_ms", 0)
        chunks_count = retrieval_info.get("metrics", {}).get("chunks_retrieved", 0)
        
        self.ui.show_answer(answer, chunks_count, total_time)
        
        # Save to session
        self.session_manager.add_conversation_to_current(
            question, answer, retrieval_info
        )


def main():
    """Entry point."""
    app = PubMedRAGApp()
    app.run()


if __name__ == "__main__":
    main()