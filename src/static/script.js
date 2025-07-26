// PubMed RAG Web Interface JavaScript
class PubMedRAGApp {
    constructor() {
        this.currentSession = null;
        this.currentTopic = null;
        this.lastRetrievalInfo = null;
        this.websocket = null;
        this.isStreaming = false;
        this.sidebarCollapsed = true; // Start with sidebar collapsed
        
        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.connectWebSocket();
        
        // Check if we have existing sessions
        this.loadSessions().then(() => {
            // Show welcome flow for new users or when starting fresh
            this.showWelcomeFlow();
        });
    }

    setupEventListeners() {
        // Hamburger button toggle
        document.getElementById('hamburgerBtn').addEventListener('click', () => {
            this.toggleSidebar();
        });

        // Old sidebar toggle (keep for compatibility)
        document.getElementById('toggleSidebar').addEventListener('click', () => {
            this.toggleSidebar();
        });

        // Welcome flow - email step
        document.getElementById('emailNextBtn').addEventListener('click', () => {
            this.handleEmailNext();
        });

        document.getElementById('welcomeEmailInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleEmailNext();
            }
        });

        // Welcome flow - topic step
        document.getElementById('topicStartBtn').addEventListener('click', () => {
            this.handleTopicStart();
        });

        document.getElementById('welcomeTopicInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleTopicStart();
            }
        });

        // Topic tag clicks
        document.querySelectorAll('.topic-tag').forEach(tag => {
            tag.addEventListener('click', () => {
                document.getElementById('welcomeTopicInput').value = tag.dataset.topic;
                this.handleTopicStart();
            });
        });

        // New session button
        document.getElementById('newSessionBtn').addEventListener('click', () => {
            this.showWelcomeFlow();
        });

        // Message input
        const messageInput = document.getElementById('messageInput');
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.isStreaming) {
                this.sendMessage();
            }
        });

        document.getElementById('sendButton').addEventListener('click', () => {
            if (!this.isStreaming) {
                this.sendMessage();
            }
        });

        // Floating action buttons
        document.getElementById('chunksBtn').addEventListener('click', () => {
            this.showChunksModal();
        });

        document.getElementById('metricsBtn').addEventListener('click', () => {
            this.showMetricsModal();
        });

        document.getElementById('clearBtn').addEventListener('click', () => {
            this.clearData();
        });

        // Modal close buttons
        document.getElementById('closeChunks').addEventListener('click', () => {
            this.hideChunksModal();
        });

        document.getElementById('closeMetrics').addEventListener('click', () => {
            this.hideMetricsModal();
        });

        // Click outside modal to close
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                e.target.classList.remove('active');
            }
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
        };

        this.websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };

        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }

    handleWebSocketMessage(message) {
        if (message.type === 'answer_chunk') {
            this.appendToCurrentResponse(message.chunk);
        } else if (message.type === 'answer_complete') {
            this.completeResponse(message.retrieval_info);
        }
    }

    async loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const sessions = await response.json();
            this.renderSessions(sessions);
        } catch (error) {
            console.error('Error loading sessions:', error);
        }
    }

    renderSessions(sessions) {
        const sessionList = document.getElementById('sessionList');
        sessionList.innerHTML = '';

        if (sessions.length === 0) {
            sessionList.innerHTML = '<div style="text-align: center; color: #7d8590; padding: 20px;">No sessions yet</div>';
            return;
        }

        sessions.forEach(session => {
            const sessionElement = document.createElement('div');
            sessionElement.className = 'session-item';
            sessionElement.innerHTML = `
                <div class="session-name">${session.name}</div>
                <div class="session-topic">${session.topic} • ${session.question_count} questions</div>
            `;
            sessionElement.addEventListener('click', () => this.loadSession(session.id));
            sessionList.appendChild(sessionElement);
        });
    }

    async loadSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}`);
            const data = await response.json();
            
            this.currentSession = data.session;
            this.currentTopic = data.session.topic;
            
            // Hide welcome and show chat interface
            this.hideWelcomeFlow();
            this.showChatInterface();
            this.showSidebar();
            
            // Update UI
            this.updateTopicIndicator(this.currentTopic);
            this.enableInputs();
            this.renderChatHistory(data.session.conversations);
            
            // Update active session in sidebar
            document.querySelectorAll('.session-item').forEach(item => {
                item.classList.remove('active');
            });
            event.target.closest('.session-item').classList.add('active');
            
            // Show existing chunks info
            if (data.existing_chunks > 0) {
                this.addSystemMessage(`Using existing data: ${data.existing_chunks} chunks`);
            }
            
        } catch (error) {
            console.error('Error loading session:', error);
            this.addSystemMessage('Error loading session', 'error');
        }
    }

    renderChatHistory(conversations) {
        const messagesContainer = document.getElementById('messages');
        messagesContainer.innerHTML = '';
        
        conversations.forEach(conv => {
            this.addMessage(conv.question, 'user', false);
            this.addMessage(conv.answer, 'assistant', false);
        });
        
        this.scrollToBottom();
    }

    handleEmailNext() {
        const email = document.getElementById('welcomeEmailInput').value.trim();
        
        if (!email || !this.isValidEmail(email)) {
            alert('Please enter a valid email address');
            return;
        }

        // Save email
        this.saveEmail(email);
        
        // Show topic step
        document.getElementById('emailStep').style.display = 'none';
        document.getElementById('topicStep').style.display = 'block';
        document.getElementById('welcomeTopicInput').focus();
    }

    async handleTopicStart() {
        const email = document.getElementById('welcomeEmailInput').value.trim();
        const topic = document.getElementById('welcomeTopicInput').value.trim();

        if (!topic) {
            alert('Please enter a research topic');
            return;
        }

        // Hide welcome, show loading
        this.hideWelcomeFlow();
        this.showChatInterface();
        this.addSystemMessage('Creating new session and loading PubMed data...', 'loading');

        try {
            const response = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, topic })
            });

            const data = await response.json();
            
            this.currentTopic = topic;
            this.updateTopicIndicator(topic);
            this.enableInputs();
            
            // Show loading results
            const result = data.result;
            const metrics = result.metrics || {};
            
            this.addSystemMessage(`✓ Loaded ${result.abstracts_count} abstracts (${result.chunk_count} chunks)`);
            this.addSystemMessage(`Processing time: ${this.formatTime(metrics.total_time_ms || 0)}`);
            this.addSystemMessage('Ready for your questions!');
            
            // Show sidebar and reload sessions
            //this.showSidebar();
            //sawait this.loadSessions();
            
        } catch (error) {
            console.error('Error starting session:', error);
            this.addSystemMessage('Error starting session', 'error');
        }
    }

    showWelcomeFlow() {
        // Reset and show welcome
        document.getElementById('welcomeContainer').style.display = 'flex';
        document.getElementById('chatContainer').style.display = 'none';
        document.getElementById('inputArea').style.display = 'none';
        document.getElementById('topicIndicator').style.display = 'none';
        
        // Reset steps
        document.getElementById('emailStep').style.display = 'block';
        document.getElementById('topicStep').style.display = 'none';
        
        // Clear inputs
        document.getElementById('welcomeEmailInput').value = this.getSavedEmail() || '';
        document.getElementById('welcomeTopicInput').value = '';
        
        // Hide sidebar
        this.hideSidebar();
        
        // Focus email input
        document.getElementById('welcomeEmailInput').focus();
    }

    hideWelcomeFlow() {
        document.getElementById('welcomeContainer').style.display = 'none';
    }

    showChatInterface() {
        document.getElementById('chatContainer').style.display = 'flex';
        document.getElementById('inputArea').style.display = 'block';
        document.getElementById('topicIndicator').style.display = 'block';
    }

    showSidebar() {
        this.sidebarCollapsed = false;
        const sidebar = document.getElementById('sidebar');
        const mainContent = document.getElementById('mainContent');
        const hamburgerBtn = document.getElementById('hamburgerBtn');
        const toggleBtn = document.getElementById('toggleSidebar');
        
        sidebar.classList.remove('collapsed');
        mainContent.classList.remove('sidebar-collapsed');
        mainContent.classList.add('sidebar-expanded');
        hamburgerBtn.classList.add('active');
        toggleBtn.textContent = '←';
    }

    hideSidebar() {
        this.sidebarCollapsed = true;
        const sidebar = document.getElementById('sidebar');
        const mainContent = document.getElementById('mainContent');
        const hamburgerBtn = document.getElementById('hamburgerBtn');
        const toggleBtn = document.getElementById('toggleSidebar');
        
        sidebar.classList.add('collapsed');
        mainContent.classList.remove('sidebar-expanded');
        mainContent.classList.add('sidebar-collapsed');
        hamburgerBtn.classList.remove('active');
        toggleBtn.textContent = '→';
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    getSavedEmail() {
        return localStorage.getItem('pubmed_email');
    }



    sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message || !this.currentTopic) return;

        messageInput.value = '';
        this.addMessage(message, 'user');
        this.isStreaming = true;
        this.disableInput();

        // Start assistant response
        this.startAssistantResponse();

        // Send via WebSocket
        this.websocket.send(JSON.stringify({
            type: 'question',
            question: message,
            topic: this.currentTopic
        }));
    }

    addMessage(content, sender, scroll = true) {
        const messagesContainer = document.getElementById('messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        messageDiv.innerHTML = `
            <div class="message-content">${content}</div>
            <div class="message-meta">${new Date().toLocaleTimeString()}</div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        
        if (scroll) {
            this.scrollToBottom();
        }
        
        return messageDiv;
    }

    addSystemMessage(content, type = 'info') {
        const messagesContainer = document.getElementById('messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'system-message';
        
        const icon = type === 'error' ? '❌' : type === 'loading' ? '⏳' : '💡';
        messageDiv.innerHTML = `
            <div style="text-align: center; color: #7d8590; padding: 10px; font-size: 14px;">
                ${icon} ${content}
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    startAssistantResponse() {
        const messagesContainer = document.getElementById('messages');
        const responseDiv = document.createElement('div');
        responseDiv.className = 'message assistant-message';
        responseDiv.id = 'current-response';
        
        responseDiv.innerHTML = `
            <div class="message-content"></div>
            <div class="message-meta">${new Date().toLocaleTimeString()}</div>
        `;
        
        messagesContainer.appendChild(responseDiv);
        this.scrollToBottom();
        
        // Add typing indicator
        const content = responseDiv.querySelector('.message-content');
        content.innerHTML = '<div class="typing-indicator">Thinking<div class="typing-dots"><span></span><span></span><span></span></div></div>';
    }

    appendToCurrentResponse(chunk) {
        const currentResponse = document.getElementById('current-response');
        if (!currentResponse) return;
        
        const content = currentResponse.querySelector('.message-content');
        
        // Remove typing indicator if present
        const typingIndicator = content.querySelector('.typing-indicator');
        if (typingIndicator) {
            content.innerHTML = '';
        }
        
        content.textContent += chunk;
        this.scrollToBottom();
    }

    completeResponse(retrievalInfo) {
        const currentResponse = document.getElementById('current-response');
        if (currentResponse) {
            currentResponse.removeAttribute('id');
        }
        
        this.lastRetrievalInfo = retrievalInfo;
        this.enableInput();
        this.isStreaming = false;
        
        // Enable action buttons
        this.enableActionButtons();
        
        // Show metrics briefly
        const metrics = retrievalInfo.metrics || {};
        this.addSystemMessage(`Retrieved ${metrics.chunks_retrieved || 0} chunks | Total: ${this.formatTime(metrics.total_rag_time_ms || 0)}`);
    }

    async showChunksModal() {
        if (!this.lastRetrievalInfo || !this.lastRetrievalInfo.documents) {
            alert('No chunks available. Ask a question first.');
            return;
        }

        const modal = document.getElementById('chunksModal');
        const content = document.getElementById('chunksContent');
        
        content.innerHTML = '';
        
        const documents = this.lastRetrievalInfo.documents[0] || [];
        const distances = this.lastRetrievalInfo.distances[0] || [];
        
        documents.forEach((chunk, index) => {
            const similarity = 1 - (distances[index] || 0);
            const chunkDiv = document.createElement('div');
            chunkDiv.className = 'chunk-item';
            chunkDiv.innerHTML = `
                <div class="chunk-header">
                    <div class="chunk-title">Chunk ${index + 1}</div>
                    <div class="chunk-similarity">Similarity: ${similarity.toFixed(3)}</div>
                </div>
                <div class="chunk-text">${chunk}</div>
            `;
            content.appendChild(chunkDiv);
        });
        
        modal.classList.add('active');
    }

    hideChunksModal() {
        document.getElementById('chunksModal').classList.remove('active');
    }

    showMetricsModal() {
        if (!this.lastRetrievalInfo) {
            alert('No metrics available. Ask a question first.');
            return;
        }

        const modal = document.getElementById('metricsModal');
        const content = document.getElementById('metricsContent');
        
        const metrics = this.lastRetrievalInfo.metrics || {};
        const tokenUsage = this.lastRetrievalInfo.token_usage || {};
        
        content.innerHTML = `
            <div class="metric-group">
                <div class="metric-title">Performance Metrics</div>
                <div class="metric-item">
                    <span class="metric-label">Embedding Time:</span>
                    <span class="metric-value">${this.formatTime(metrics.embedding_time_ms || 0)}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Retrieval Time:</span>
                    <span class="metric-value">${this.formatTime(metrics.retrieval_time_ms || 0)}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Generation Time:</span>
                    <span class="metric-value">${this.formatTime(metrics.generation_time_ms || 0)}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Total Time:</span>
                    <span class="metric-value">${this.formatTime(metrics.total_rag_time_ms || 0)}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Chunks Retrieved:</span>
                    <span class="metric-value">${metrics.chunks_retrieved || 0}</span>
                </div>
            </div>
            <div class="metric-group">
                <div class="metric-title">Token Usage</div>
                <div class="metric-item">
                    <span class="metric-label">Model:</span>
                    <span class="metric-value">${tokenUsage.model || 'N/A'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Prompt Tokens:</span>
                    <span class="metric-value">${tokenUsage.prompt_tokens || 0}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Completion Tokens:</span>
                    <span class="metric-value">${tokenUsage.completion_tokens || 0}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Total Tokens:</span>
                    <span class="metric-value">${tokenUsage.total_tokens || 0}</span>
                </div>
            </div>
        `;
        
        modal.classList.add('active');
    }

    hideMetricsModal() {
        document.getElementById('metricsModal').classList.remove('active');
    }

    async clearData() {
        if (!this.currentTopic) return;
        
        if (!confirm(`Clear all data for topic "${this.currentTopic}"?`)) return;
        
        try {
            const response = await fetch(`/api/data/${this.currentTopic}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                this.addSystemMessage('Data cleared successfully');
                this.disableActionButtons();
            } else {
                this.addSystemMessage('Error clearing data', 'error');
            }
        } catch (error) {
            console.error('Error clearing data:', error);
            this.addSystemMessage('Error clearing data', 'error');
        }
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const mainContent = document.getElementById('mainContent');
        const hamburgerBtn = document.getElementById('hamburgerBtn');
        const toggleBtn = document.getElementById('toggleSidebar');
        
        this.sidebarCollapsed = !this.sidebarCollapsed;
        
        if (this.sidebarCollapsed) {
            sidebar.classList.add('collapsed');
            mainContent.classList.remove('sidebar-expanded');
            mainContent.classList.add('sidebar-collapsed');
            hamburgerBtn.classList.remove('active');
            toggleBtn.textContent = '→';
        } else {
            sidebar.classList.remove('collapsed');
            mainContent.classList.remove('sidebar-collapsed');
            mainContent.classList.add('sidebar-expanded');
            hamburgerBtn.classList.add('active');
            toggleBtn.textContent = '←';
        }
    }

    updateTopicIndicator(topic) {
        document.getElementById('topicIndicator').textContent = topic;
    }

    enableInputs() {
        document.getElementById('messageInput').disabled = false;
        document.getElementById('sendButton').disabled = false;
    }

    disableInput() {
        document.getElementById('messageInput').disabled = true;
        document.getElementById('sendButton').disabled = true;
    }

    enableInput() {
        document.getElementById('messageInput').disabled = false;
        document.getElementById('sendButton').disabled = false;
        document.getElementById('messageInput').focus();
    }

    enableActionButtons() {
        document.getElementById('chunksBtn').disabled = false;
        document.getElementById('metricsBtn').disabled = false;
        document.getElementById('clearBtn').disabled = false;
    }

    disableActionButtons() {
        document.getElementById('chunksBtn').disabled = true;
        document.getElementById('metricsBtn').disabled = true;
        document.getElementById('clearBtn').disabled = true;
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    formatTime(milliseconds) {
        if (!milliseconds) return '0ms';
        if (milliseconds < 1000) {
            return `${milliseconds}ms`;
        } else {
            return `${(milliseconds / 1000).toFixed(1)}s`;
        }
    }

    loadSavedEmail() {
        // Try to load saved email from localStorage
        const savedEmail = localStorage.getItem('pubmed_email');
        if (savedEmail) {
            document.getElementById('emailInput').value = savedEmail;
        }
    }

    saveEmail(email) {
        localStorage.setItem('pubmed_email', email);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.pubmedApp = new PubMedRAGApp();
}); 