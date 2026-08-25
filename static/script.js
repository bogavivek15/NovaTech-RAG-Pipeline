// Use the same origin in production (Render) and localhost during local development.
const API_URL = window.location.origin;

const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const fileInput = document.getElementById('file-input');
const chooseFilesBtn = document.getElementById('choose-files-btn');
const uploadBtn = document.getElementById('upload-btn');
const selectedFilesDiv = document.getElementById('selected-files');
const systemStatus = document.getElementById('system-status');
const exampleBtns = document.querySelectorAll('.example-btn');

let selectedFiles = [];

document.addEventListener('DOMContentLoaded', function() {
    console.log('NovaTech assistant started');
    checkBackendHealth();
    setupEventListeners();
    userInput.focus();
});

function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') sendMessage();
    });
    chooseFilesBtn.addEventListener('click', function() {
        fileInput.click();
    });
    fileInput.addEventListener('change', handleFileSelection);
    uploadBtn.addEventListener('click', uploadFiles);
    exampleBtns.forEach(function(btn) {
        btn.addEventListener('click', function() {
            userInput.value = btn.textContent;
            sendMessage();
        });
    });
}

function checkBackendHealth() {
    fetch(API_URL + '/health')
        .then(function(response) {
            if (!response.ok) throw new Error('Health check returned HTTP ' + response.status);
            return response.json();
        })
        .then(function(data) {
            displaySystemStatus(data);
        })
        .catch(function(error) {
            console.error('Backend health check failed:', error);
            systemStatus.innerHTML = `
                <p class="status-warning">⚠️ Cannot connect to backend</p>
                <p class="small-text">Backend health check failed</p>
            `;
        });
}

function displaySystemStatus(data) {
    let html = '';
    if (!data.groq_api_key_configured) {
        html = `
            <p class="status-error">❌ GROQ_API_KEY missing</p>
            <p class="small-text">Add it to Render environment variables</p>
        `;
    } else {
        html = `
            <p class="status-ok">✅ System Ready</p>
            <p style="margin-top: 8px;"><strong>Document Chunks:</strong> ${data.document_chunks || 0}</p>
            <p class="small-text">Model: ${data.model || 'N/A'}</p>
        `;
    }
    systemStatus.innerHTML = html;
}

function sendMessage() {
    const question = userInput.value.trim();
    if (!question) return;

    console.log('Sending question:', question);
    userInput.value = '';

    const welcome = document.querySelector('.welcome');
    if (welcome) welcome.remove();

    addMessage('user', question);
    const loadingDiv = showLoading();

    userInput.disabled = true;
    sendBtn.disabled = true;

    fetch(API_URL + '/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ question: question })
    })
    .then(function(response) {
        return response.json().then(function(data) {
            if (!response.ok) {
                throw new Error(data.detail || 'Backend returned HTTP ' + response.status);
            }
            return data;
        });
    })
    .then(function(data) {
        loadingDiv.remove();
        addMessage('bot', data.answer, data.sources);
    })
    .catch(function(error) {
        console.error('Chat error:', error);
        loadingDiv.remove();
        addMessage('bot', '❌ Error: ' + error.message);
    })
    .finally(function() {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    });
}

function addMessage(role, text, sources) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ' + role + '-message';

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'avatar';
    avatarDiv.textContent = role === 'user' ? '👤' : '🤖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = text;
    contentDiv.appendChild(textDiv);

    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        sourcesDiv.textContent = '📚 Sources: ' + sources.join(', ');
        contentDiv.appendChild(sourcesDiv);
    }

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message';
    loadingDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="message-content">
            <div class="loading">
                <div class="spinner"></div>
                <span>Searching documents...</span>
            </div>
        </div>
    `;
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return loadingDiv;
}

function handleFileSelection(event) {
    selectedFiles = Array.from(event.target.files);
    console.log('Files selected:', selectedFiles.length);

    if (selectedFiles.length === 0) {
        selectedFilesDiv.innerHTML = '';
        uploadBtn.style.display = 'none';
        return;
    }

    let html = '';
    selectedFiles.forEach(function(file) {
        html += `<div class="file-item">📄 ${file.name}</div>`;
    });
    selectedFilesDiv.innerHTML = html;
    uploadBtn.style.display = 'block';
}

function uploadFiles() {
    if (selectedFiles.length === 0) return;

    console.log('Uploading files...');
    uploadBtn.disabled = true;
    uploadBtn.textContent = '⏳ Processing...';

    const formData = new FormData();
    selectedFiles.forEach(function(file) {
        formData.append('files', file);
    });

    fetch(API_URL + '/upload', {
        method: 'POST',
        body: formData
    })
    .then(function(response) {
        return response.json().then(function(data) {
            if (!response.ok) {
                throw new Error(data.detail || 'Upload failed with HTTP ' + response.status);
            }
            return data;
        });
    })
    .then(function(data) {
        alert('✅ ' + data.message + '\n📊 ' + data.chunks_added + ' chunks added');
        fileInput.value = '';
        selectedFiles = [];
        selectedFilesDiv.innerHTML = '';
        uploadBtn.style.display = 'none';
        checkBackendHealth();
    })
    .catch(function(error) {
        console.error('Upload error:', error);
        alert('❌ Upload failed: ' + error.message);
    })
    .finally(function() {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '📤 Upload & Process';
    });
}
