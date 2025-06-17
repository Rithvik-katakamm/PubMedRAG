import sys
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich import box
from rich.text import Text


class TerminalChat:
    """Clean, minimal terminal UI for the PubMed RAG system."""
    
    def __init__(self):
        self.console = Console()
        self.last_chunks = None
        self.last_distances = None
        self._current_session_name = ""
    
    def clear_screen(self):
        """Clear the terminal screen."""
        self.console.clear()
    
    def show_header(self, session_name: str = "New Session", topic: str = ""):
        """Display the minimal header."""
        # Main title
        title = Panel(
            "[bold white]PubMed Medical Research Assistant[/bold white]",
            box=box.DOUBLE,
            style="white",
            expand=False
        )
        self.console.print(title)
        
        # Session info - more subtle
        if topic:
            session_text = f"[white]{session_name}[/white] • Topic: [green]{topic}[/green]"
        else:
            session_text = f"[white]{session_name}[/white]"
        
        self.console.print(f"\n{session_text}\n")
        self._current_session_name = session_name
    
    def update_session_name_inline(self, new_name: str):
        """Update session name in place (when auto-naming completes)."""
        # Store for next header refresh
        self._current_session_name = new_name
    
    def show_welcome(self):
        """Show simple welcome screen."""
        self.clear_screen()
        
        welcome = Panel(
            "[bold white]PubMed Medical Research Assistant[/bold white]\n\n"
            "Search and analyze medical literature with AI",
            box=box.DOUBLE,
            style="white",
            padding=(1, 2),
            expand=False
        )
        
        self.console.print(welcome)
        self.console.print()
    
    def get_email_input(self) -> str:
        """Get email input with validation."""
        while True:
            email = Prompt.ask(
                "[green]Email address for NCBI[/green]",
                console=self.console
            )
            
            if "@" in email and "." in email:
                return email
            
            self.console.print("[red]Please enter a valid email address[/red]")
    
    def get_topic_input(self) -> str:
        """Get research topic input."""
        while True:
            topic = Prompt.ask(
                "[green]Research topic[/green]",
                console=self.console
            )
            
            if topic.strip():
                return topic.strip()
            
            self.console.print("[red]Please enter a valid topic[/red]")
    
    def show_session_list(self, sessions: List[Dict[str, Any]]) -> Optional[str]:
        """Display numbered list of sessions for easy selection."""
        if not sessions:
            return None
        
        # Create simple numbered table
        table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
        table.add_column("", style="bright_white", width=3)
        table.add_column("Session", style="white")
        table.add_column("Date", style="white")
        table.add_column("", style="dim white", justify="right")
        
        for i, session in enumerate(sessions[:9], 1):  # Limit to 9 for single digit
            created = datetime.fromisoformat(session["created_at"])
            date_str = created.strftime("%b %d")
            
            table.add_row(
                f"{i}.",
                session["name"],
                date_str,
                f"({session['question_count']} questions)"
            )
        
        self.console.print("\n")
        self.console.print(Panel(table, title="[bold white]Previous Sessions[/bold white]", box=box.ROUNDED))
        
        return None  # Let caller handle the prompt
    
    def get_session_choice(self) -> str:
        """Get session number or Enter for new."""
        choice = Prompt.ask(
            "\n[green]Session number (or Enter for new)[/green]",
            default="",
            console=self.console
        )
        return choice
    
    def show_processing(self, message: str) -> Tuple[Any, float]:
        """Show processing with timer."""
        start_time = time.time()
        self.console.print(f"[yellow]{message}...[/yellow] ", end="")
        return None, start_time
    
    def end_processing(self, start_time: float):
        """End processing and show time."""
        elapsed = time.time() - start_time
        if elapsed < 1.0:
            self.console.print(f"[yellow]{int(elapsed * 1000)}ms[/yellow]")
        else:
            self.console.print(f"[yellow]{elapsed:.1f}s[/yellow]")
    
    def format_time(self, milliseconds: int) -> str:
        """Format time for display."""
        if milliseconds < 1000:
            return f"{milliseconds}ms"
        else:
            return f"{milliseconds / 1000:.1f}s"
    
    def show_metrics_inline(self, metrics: Dict[str, Any]):
        """Show minimal inline metrics during processing."""
        if "retrieval_time_ms" in metrics:
            _, start = self.show_processing("Retrieving chunks")
            time.sleep(0.01)  # Visual feedback
            self.end_processing(start - metrics["retrieval_time_ms"]/1000)
        
        if "generation_time_ms" in metrics:
            _, start = self.show_processing("Generating response")
            time.sleep(0.01)  # Visual feedback
            self.end_processing(start - metrics["generation_time_ms"]/1000)
    
    def get_user_question(self) -> str:
        """Get user question with clean prompt."""
        question = Prompt.ask(
            "[green bold]>[/green bold]",
            console=self.console
        )
        return question
    
    def show_answer(self, answer: str, chunks_count: int, total_time_ms: int):
        """Display answer with metrics."""
        # Show answer
        self.console.print(f"\n{answer}\n")
        
        # Show summary metrics
        time_str = self.format_time(total_time_ms)
        self.console.print(f"[dim white]Retrieved {chunks_count} chunks | Total: {time_str}[/dim white]")
    
    def show_chunks(self):
        """Display all retrieved chunks in full."""
        if not self.last_chunks:
            self.console.print("[yellow]No chunks to display[/yellow]")
            return
        
        self.console.print("\n[bold white]Retrieved Chunks (Latest Query)[/bold white]")
        self.console.print("─" * 80)
        
        # Show all chunks with full text
        for i, (chunk, dist) in enumerate(zip(self.last_chunks, self.last_distances), 1):
            similarity = 1.0 - dist
            
            # Chunk header with similarity score
            self.console.print(f"\n[bold]Chunk {i}[/bold] | [green]Similarity: {similarity:.3f}[/green]")
            self.console.print("─" * 40)
            
            # Full chunk text - no truncation
            self.console.print(chunk)
            
            # Add separator between chunks
            if i < len(self.last_chunks):
                self.console.print("\n" + "─" * 80)
        
        self.console.print("\n" + "─" * 80)
        self.console.print("\n[dim]Press Enter to continue[/dim]")
        input()
    
    def store_chunks(self, chunks: List[str], distances: List[float]):
        """Store chunks for later viewing."""
        self.last_chunks = chunks
        self.last_distances = distances
    
    def show_help(self):
        """Display all available commands and shortcuts."""
        help_content = """[bold white]Commands:[/bold white]
  /chunks     - View chunks from your last question
  /clear      - Clear database for current topic  
  /clear all  - Clear entire database
  /new        - Start a new session
  /sessions   - View all sessions
  /help       - Show this help
  /exit       - Exit the application

[bold white]Shortcuts:[/bold white]
  c          - Quick view chunks (same as /chunks)
  Enter      - Send message
  Ctrl+C     - Exit anytime"""
        
        panel = Panel(
            help_content,
            title="[bold white]Available Commands[/bold white]",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print("\n")
        self.console.print(panel)
        self.console.print("\n")
    
    def show_error(self, message: str, details: Optional[str] = None):
        """Display an error message."""
        self.console.print(f"[red]Error: {message}[/red]")
        if details:
            self.console.print(f"[dim]{details}[/dim]")
    
    def show_success(self, message: str):
        """Display a success message."""
        self.console.print(f"[green]✓[/green] {message}")
    
    def show_data_loading_progress(self, step: str, time_ms: Optional[int] = None):
        """Show progress for data loading steps."""
        if time_ms is not None:
            time_str = self.format_time(time_ms)
            self.console.print(f"[green]✓[/green] {step}... {time_str}")
        else:
            self.console.print(f"[yellow]•[/yellow] {step}...")
    
    def show_goodbye(self):
        """Display goodbye message."""
        self.console.print("\n[white]Session saved. Goodbye![/white]\n")