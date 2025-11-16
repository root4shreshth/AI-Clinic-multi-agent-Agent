(function() {
    const API_KEY = 'AIzaSyAU5NkvibezNWNGM5ZT-o_vFtwEtgSzsmw';
    const API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent';
    
    const chatbotToggle = document.getElementById('chatbotToggle');
    const chatbotWindow = document.getElementById('chatbotWindow');
    const chatbotClose = document.getElementById('chatbotClose');
    const chatbotMessages = document.getElementById('chatbotMessages');
    const chatbotInput = document.getElementById('chatbotInput');
    const chatbotSend = document.getElementById('chatbotSend');
    
    let conversationHistory = [];
    
    // Toggle chatbot window
    chatbotToggle.addEventListener('click', () => {
        chatbotWindow.classList.remove('hidden');
        chatbotToggle.style.display = 'none';
        chatbotInput.focus();
    });
    
    chatbotClose.addEventListener('click', () => {
        chatbotWindow.classList.add('hidden');
        chatbotToggle.style.display = 'flex';
    });
    
    // Auto-resize textarea
    chatbotInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
    
    // Send message on Enter (Shift+Enter for new line)
    chatbotInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    chatbotSend.addEventListener('click', sendMessage);
    
    async function sendMessage() {
        const message = chatbotInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        chatbotInput.value = '';
        chatbotInput.style.height = 'auto';
        
        // Disable input while processing
        chatbotInput.disabled = true;
        chatbotSend.disabled = true;
        
        // Show typing indicator
        const typingId = showTypingIndicator();
        
        try {
            // Call Gemini API
            const response = await fetch(`${API_URL}?key=${API_KEY}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    contents: [{
                        parts: [{
                            text: `You are MediScan Assistant, a helpful AI assistant for a medical triage system. 
You help users understand the system, answer medical questions (while being clear you're not a doctor), 
and provide information about the patient intake process. Keep responses concise and friendly.

User question: ${message}`
                        }]
                    }],
                    generationConfig: {
                        temperature: 0.7,
                        maxOutputTokens: 500,
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }
            
            const data = await response.json();
            const botResponse = data.candidates[0].content.parts[0].text;
            
            // Remove typing indicator
            removeTypingIndicator(typingId);
            
            // Add bot response
            addMessage(botResponse, 'bot');
            
        } catch (error) {
            console.error('Error calling Gemini API:', error);
            removeTypingIndicator(typingId);
            addMessage('I apologize, but I encountered an error. Please try again.', 'bot');
        } finally {
            // Re-enable input
            chatbotInput.disabled = false;
            chatbotSend.disabled = false;
            chatbotInput.focus();
        }
    }
    
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const time = new Date().toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit' 
        });
        
        const avatar = sender === 'bot' ? '🏥' : '👤';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <p>${escapeHtml(text)}</p>
                <div class="message-time">${time}</div>
            </div>
        `;
        
        chatbotMessages.appendChild(messageDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }
    
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-typing';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-avatar">🏥</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatbotMessages.appendChild(typingDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        return 'typing-indicator';
    }
    
    function removeTypingIndicator(id) {
        const typingDiv = document.getElementById(id);
        if (typingDiv) {
            typingDiv.remove();
        }
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
})();
