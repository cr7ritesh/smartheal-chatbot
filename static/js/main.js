// Global variables
let isAsking = false;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let currentTranscription = '';

// DOM elements
const chatMessages = document.getElementById('chat-messages');
const questionInput = document.getElementById('question-input');
const askButton = document.getElementById('ask-button');

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    setupChatInput();
    scrollToBottom();
});

function setupChatInput() {
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askQuestion();
        }
    });
}

async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question || isAsking) return;
    
    isAsking = true;
    questionInput.disabled = true;
    askButton.disabled = true;
    
    addMessage('user', question);
    questionInput.value = '';
    
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        
        const result = await response.json();
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
    
    const emptyChat = chatMessages.querySelector('.empty-chat');
    if (emptyChat) emptyChat.remove();
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

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

function removeLoadingMessage(loadingId) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) loadingElement.remove();
}

async function clearChat() {
    try {
        const response = await fetch('/clear', { method: 'POST' });
        
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

// Voice Recording Functions
function updateRecordingUI(recording) {
    const recordBtn = document.getElementById('record-btn');
    const micIcon = document.getElementById('mic-icon');
    const recordText = document.getElementById('record-text');
    const recordingStatus = document.getElementById('recording-status');
    
    if (recording) {
        recordBtn.classList.add('btn-danger');
        recordBtn.classList.remove('btn-outline-primary');
        micIcon.className = 'bi bi-stop-circle';
        recordText.textContent = 'Stop Recording';
        recordingStatus.textContent = 'Recording... Click stop when done.';
    } else {
        recordBtn.classList.remove('btn-danger');
        recordBtn.classList.add('btn-outline-primary');
        micIcon.className = 'bi bi-mic';
        recordText.textContent = 'Record';
        recordingStatus.textContent = '';
    }
}

async function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        const options = { mimeType: 'audio/wav' };
        if (!MediaRecorder.isTypeSupported('audio/wav')) {
            if (MediaRecorder.isTypeSupported('audio/webm')) {
                options.mimeType = 'audio/webm';
            } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                options.mimeType = 'audio/mp4';
            }
        }
        
        console.log('Using MIME type:', options.mimeType);
        
        mediaRecorder = new MediaRecorder(stream, options);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: options.mimeType });
            processAudioBlob(audioBlob);
        };
        
        mediaRecorder.start();
        isRecording = true;
        updateRecordingUI(true);
        
    } catch (error) {
        console.error('Error starting recording:', error);
        showAlert('Microphone access denied or not available', 'danger');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        updateRecordingUI(false);
        document.getElementById('recording-status').textContent = 'Processing audio...';
    }
}

async function processAudioBlob(audioBlob) {
    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        
        const response = await fetch('/transcribe_audio', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayTranscriptionResult(result);
        } else {
            showAlert(result.error || 'Transcription failed', 'danger');
        }
    } catch (error) {
        console.error('Error processing audio:', error);
        showAlert('Error processing audio', 'danger');
    } finally {
        document.getElementById('recording-status').textContent = '';
    }
}

async function handleAudioFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    document.getElementById('recording-status').textContent = 'Processing uploaded audio...';
    
    try {
        const formData = new FormData();
        formData.append('audio', file);
        
        const response = await fetch('/transcribe_audio', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayTranscriptionResult(result);
        } else {
            showAlert(result.error || 'Transcription failed', 'danger');
        }
    } catch (error) {
        console.error('Error processing uploaded audio:', error);
        showAlert('Error processing audio file', 'danger');
    } finally {
        document.getElementById('recording-status').textContent = '';
        event.target.value = '';
    }
}

function displayTranscriptionResult(result) {
    currentTranscription = result.transcription;
    
    document.getElementById('transcription-text').innerHTML = `
        <div class="fw-bold text-primary">"${escapeHtml(result.transcription)}"</div>
        <small class="text-muted">Language: ${result.language.toUpperCase()}</small>
    `;
    
    if (result.segments && result.segments.length > 1 && result.speaker_count > 1) {
        let speakerInfo = `<strong>Speakers detected: ${result.speaker_count}</strong><br><div class="mt-2">`;
        
        result.segments.forEach((segment, index) => {
            if (segment.text.trim()) {
                speakerInfo += `<div class="mb-1">
                    <span class="badge bg-secondary">${segment.speaker}</span>
                    <small>"${escapeHtml(segment.text.trim())}"</small>
                </div>`;
            }
        });
        
        speakerInfo += '</div>';
        document.getElementById('speaker-info').innerHTML = speakerInfo;
    } else {
        document.getElementById('speaker-info').innerHTML = '<small class="text-muted">Single speaker detected</small>';
    }
    
    document.getElementById('transcription-result').style.display = 'block';
}

function useTranscription() {
    if (currentTranscription) {
        document.getElementById('question-input').value = currentTranscription;
        document.getElementById('transcription-result').style.display = 'none';
        document.getElementById('question-input').focus();
    }
}

function hideTranscription() {
    document.getElementById('transcription-result').style.display = 'none';
}

// Utility Functions
function showAlert(message, type) {
    const existingAlerts = document.querySelectorAll('.alert-dismissible');
    existingAlerts.forEach(alert => alert.remove());
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const mainContent = document.querySelector('.main-content');
    const firstChild = mainContent.querySelector('.d-flex');
    mainContent.insertBefore(alertDiv, firstChild.nextSibling);
    
    setTimeout(() => {
        if (alertDiv.parentNode) alertDiv.remove();
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
