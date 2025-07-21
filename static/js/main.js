// Global variables
let isAsking = false;

// DOM elements
const chatMessages = document.getElementById('chat-messages');
const questionInput = document.getElementById('question-input');
const askButton = document.getElementById('ask-button');

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    setupChatInput();
    scrollToBottom();
});

// Chat input setup
function setupChatInput() {
    // Enter key event
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askQuestion();
        }
    });
}

// Ask question
async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question || isAsking) return;
    
    isAsking = true;
    questionInput.disabled = true;
    askButton.disabled = true;
    
    // Add user message to chat
    addMessage('user', question);
    questionInput.value = '';
    
    // Add loading message
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });
        
        const result = await response.json();
        
        // Remove loading message
        removeLoadingMessage(loadingId);
        
        if (result.success) {
            addMessage('assistant', result.answer);
        } else {
            showAlert(result.error || 'Failed to get answer', 'danger');
            addMessage('assistant', 'Sorry, I encountered an error while processing your question. Please try again.');
        }
    } catch (error) {
        console.error('Question error:', error);
        removeLoadingMessage(loadingId);
        showAlert('Network error occurred', 'danger');
        addMessage('assistant', 'Sorry, I encountered a network error. Please check your connection and try again.');
    } finally {
        isAsking = false;
        questionInput.disabled = false;
        askButton.disabled = false;
        questionInput.focus();
    }
}

// Check system status
async function checkStatus() {
    try {
        const response = await fetch('/status');
        const status = await response.json();
        
        let statusHtml = '<div class="small">';
        statusHtml += `<div><strong>Cohere API:</strong> ${status.cohere_api_key ? '✅ Connected' : '❌ Not configured'}</div>`;
        statusHtml += `<div><strong>Pinecone API:</strong> ${status.pinecone_api_key ? '✅ Connected' : '❌ Not configured'}</div>`;
        statusHtml += `<div><strong>Vector Store:</strong> ${status.vectorstore_status === 'ready' ? '✅ Ready' : '❌ Not ready'}</div>`;
        statusHtml += `<div><strong>Index:</strong> ${status.index_name}</div>`;
        if (status.document_count !== undefined) {
            statusHtml += `<div><strong>Documents:</strong> ${status.document_count}</div>`;
        }
        statusHtml += '</div>';
        
        showAlert(statusHtml, status.vectorstore_status === 'ready' ? 'success' : 'warning');
    } catch (error) {
        console.error('Status check error:', error);
        showAlert('Failed to check system status', 'danger');
    }
}

// Add message to chat
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const icon = role === 'user' ? 'bi-person-circle' : 'bi-robot';
    const label = role === 'user' ? 'You' : 'SmartHeal Assistant';
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="message-header">
                <i class="bi ${icon}"></i> ${label}
            </div>
            <div class="message-text">${escapeHtml(content)}</div>
        </div>
    `;
    
    // Remove empty chat message if it exists
    const emptyChat = chatMessages.querySelector('.empty-chat');
    if (emptyChat) {
        emptyChat.remove();
    }
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Add loading message
function addLoadingMessage() {
    const loadingId = 'loading-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = loadingId;
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="message-header">
                <i class="bi bi-robot"></i> SmartHeal Assistant
            </div>
            <div class="message-text loading-message">
                <span>Thinking</span>
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    return loadingId;
}

// Remove loading message
function removeLoadingMessage(loadingId) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) {
        loadingElement.remove();
    }
}

// Clear chat
async function clearChat() {
    try {
        const response = await fetch('/clear', {
            method: 'POST'
        });
        
        if (response.ok) {
            chatMessages.innerHTML = `
                <div class="empty-chat">
                    <i class="bi bi-chat-heart display-4 text-muted"></i>
                    <p class="text-muted mt-3">Start asking questions about your documents!</p>
                    <p class="text-muted small">Ask me anything about health, fitness, or medical topics.</p>
                </div>
            `;
            showAlert('Chat history cleared', 'info');
        }
    } catch (error) {
        console.error('Clear chat error:', error);
        showAlert('Failed to clear chat', 'danger');
    }
}

// Utility functions
function showAlert(message, type) {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert-dismissible');
    existingAlerts.forEach(alert => alert.remove());
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of main content
    const mainContent = document.querySelector('.main-content');
    const firstChild = mainContent.querySelector('.d-flex');
    mainContent.insertBefore(alertDiv, firstChild.nextSibling);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}