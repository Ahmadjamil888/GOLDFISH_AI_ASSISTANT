import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Set
from utils.tokenizer import load_vocab, tokenize, detokenize

# === Load Configuration ===
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "..", "train_config.json")

if not os.path.isfile(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

EMBED_DIM = config["model"]["embedding_dim"]
HIDDEN_DIM = config["model"]["hidden_dim"]
VOCAB_SIZE = config["model"]["vocab_size"]
MAX_LEN = config["model"]["max_seq_len"]

# === Load Vocabulary ===
VOCAB_PATH = os.path.join(BASE_DIR, "..", config["paths"]["vocab"])
if not os.path.isfile(VOCAB_PATH):
    raise FileNotFoundError(f"❌ Vocabulary file not found: {VOCAB_PATH}")

vocab, word2id, id2word = load_vocab(VOCAB_PATH)
PAD_ID = word2id["<PAD>"]
UNK_ID = word2id["<UNK>"]
BOS_ID = word2id["<BOS>"]
EOS_ID = word2id["<EOS>"]

# Define common words for filtering
COMMON_WORDS = {
    # Greetings and basic responses
    'hello', 'hi', 'hey', 'good', 'morning', 'afternoon', 'evening', 'night',
    'welcome', 'greetings', 'bye', 'goodbye', 'see', 'you', 'later',
    
    # Common verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'do', 'does', 'did', 'done', 'doing',
    'have', 'has', 'had', 'having',
    'can', 'could', 'will', 'would', 'should', 'may', 'might',
    'help', 'need', 'want', 'like', 'love', 'think', 'know',
    
    # Common pronouns and articles
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'this', 'that', 'these', 'those',
    'the', 'a', 'an',
    
    # Common prepositions and conjunctions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'and', 'or', 'but', 'if', 'then', 'than', 'because',
    
    # Question words
    'what', 'where', 'when', 'why', 'who', 'which', 'how',
    
    # Common adverbs
    'very', 'really', 'quite', 'just', 'now', 'here', 'there',
    'today', 'tomorrow', 'yesterday', 'always', 'never', 'sometimes',
    
    # Polite words
    'please', 'thank', 'thanks', 'welcome', 'sorry', 'excuse',
    
    # Common responses
    'yes', 'no', 'maybe', 'sure', 'okay', 'ok', 'alright'
}

# Technical terms to filter out
TECHNICAL_TERMS = {
    'sudo', 'apt', 'git', 'npm', 'pip', 'brew', 'docker', 'kubernetes',
    'linux', 'unix', 'ubuntu', 'debian', 'fedora', 'centos', 'redhat',
    'apache', 'nginx', 'mysql', 'postgres', 'mongodb', 'redis',
    'php', 'html', 'css', 'js', 'javascript', 'python', 'ruby', 'java',
    'api', 'rest', 'soap', 'xml', 'json', 'yaml', 'toml',
    'config', 'conf', 'rc', 'ini', 'yml',
    'wicd', 'kde', 'gnome', 'xfce', 'mate', 'unity',
    'grub', 'boot', 'bios', 'uefi', 'mbr', 'gpt',
    'http', 'https', 'ftp', 'ssh', 'ssl', 'tls',
    'wiki', 'forum', 'irc', 'chat'
}

# === Define Chat Model ===
class SimpleChatModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output[:, -1, :])
        return logits, hidden

# === Load Pretrained Weights ===
WEIGHTS_PATH = os.path.join(BASE_DIR, "..", config["paths"]["save_weights"])
if not os.path.isfile(WEIGHTS_PATH):
    raise FileNotFoundError(f"❌ Model weights not found: {WEIGHTS_PATH}")

model = SimpleChatModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

# Common conversation patterns
CONVERSATION_PATTERNS = {
    'greetings': {
        'patterns': ['hello', 'hi', 'hey', 'greetings'],
        'responses': [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hello! How are you today?",
            "Hi! I'm here to help. What do you need?",
            "Greetings! How may I assist you?"
        ]
    },
    'how_are_you': {
        'patterns': ['how are you', 'how do you do', 'how are things', 'how\'s it going'],
        'responses': [
            "I'm doing well, thank you! How are you?",
            "I'm great, thanks for asking! How about you?",
            "I'm good! Thanks for asking. How can I help you today?",
            "I'm doing fine, thank you! What can I do for you?",
            "I'm well! Thanks for asking. How may I assist you?"
        ]
    },
    'identity': {
        'patterns': ['who are you', 'what are you', 'your name', 'who created you', 'what kind of bot'],
        'responses': [
            "I'm an AI assistant designed to help you with various tasks and conversations.",
            "I'm a friendly AI chatbot, here to assist you with your questions and needs.",
            "I'm an AI companion created to help and chat with you.",
            "I'm your AI assistant, ready to help you with whatever you need.",
            "I'm an AI chatbot focused on being helpful and providing assistance."
        ]
    },
    'capabilities': {
        'patterns': ['what can you do', 'what do you do', 'how can you help', 'your capabilities', 'what are you capable of'],
        'responses': [
            "I can help you with conversations, answer questions, and assist with various tasks. What would you like help with?",
            "I'm designed to engage in conversations, provide information, and help you with different tasks. What do you need?",
            "I can assist you with questions, engage in discussions, and help solve problems. What interests you?",
            "I'm here to help with information, conversations, and various tasks. What would you like to know?",
            "I can help you by answering questions, having conversations, and assisting with different needs. What can I help you with?"
        ]
    },
    'thank_you': {
        'patterns': ['thank you', 'thanks', 'thank', 'appreciate it'],
        'responses': [
            "You're welcome! Let me know if you need anything else.",
            "Happy to help! Is there anything else you'd like to know?",
            "My pleasure! Feel free to ask if you have more questions.",
            "Glad I could help! Let me know if you need more assistance.",
            "You're welcome! Don't hesitate to ask if you need more help."
        ]
    },
    'goodbye': {
        'patterns': ['goodbye', 'bye', 'see you', 'farewell', 'good night'],
        'responses': [
            "Goodbye! Have a great day!",
            "Bye! Take care!",
            "See you later! Have a wonderful day!",
            "Farewell! Feel free to return if you need more help!",
            "Goodbye! It was nice chatting with you!"
        ]
    },
    'help': {
        'patterns': ['help', 'can you help', 'need help', 'assist'],
        'responses': [
            "I'd be happy to help! What do you need assistance with?",
            "Of course! What can I help you with?",
            "I'm here to help! What would you like to know?",
            "Sure thing! What kind of help do you need?",
            "I'll do my best to help! What's on your mind?"
        ]
    },
    'dont_understand': {
        'patterns': ['what', 'dont understand', 'confused', 'unclear', 'what do you mean'],
        'responses': [
            "I'll try to explain more clearly. What specifically would you like me to clarify?",
            "Let me help clarify that for you. Which part is unclear?",
            "I can explain in more detail. What would you like me to explain?",
            "I'll be happy to clarify. What part would you like me to explain better?",
            "Let's break this down together. What part would you like me to explain?"
        ]
    },
    'learning_coding': {
        'patterns': ['learn coding', 'learn programming', 'learn html', 'learn css', 'learn javascript', 'learn python', 'want to code', 'start coding', 'begin programming'],
        'responses': [
            "Learning to code is a great goal! I can suggest some resources and steps to get started. Which programming language interests you most: Python (great for beginners), JavaScript (for web development), or something else?",
            "That's excellent that you want to learn coding! Would you like to focus on web development (HTML, CSS, JavaScript) or general programming (Python)? I can guide you with either path.",
            "I can help guide you on your coding journey! Are you interested in web development, data science, or general programming? This will help me suggest the best resources for you.",
            "Learning to code is an exciting journey! To help you best, could you tell me what you'd like to build? Websites, apps, data analysis, or something else?",
            "Great choice to start coding! To provide the best guidance, could you tell me if you have any previous programming experience, and what kind of projects interest you?"
        ]
    },
    'specific_languages': {
        'patterns': ['html', 'css', 'javascript', 'python', 'java', 'programming language'],
        'responses': [
            "For learning programming languages, I can recommend resources and learning paths. Which aspect would you like to know more about: tutorials, practice exercises, or project ideas?",
            "I can help guide you with programming languages. Would you like to know about online courses, documentation, or hands-on projects to practice with?",
            "There are many great ways to learn programming languages. Should we focus on beginner tutorials, interactive learning platforms, or specific project-based learning?",
            "I can suggest some excellent resources for learning programming. Would you prefer online tutorials, video courses, or interactive coding platforms?",
            "Let's explore programming together! Would you like to start with basic concepts, practical exercises, or see some example projects?"
        ]
    },
    'coding_help': {
        'patterns': ['help with code', 'fix code', 'debug', 'coding problem', 'programming help'],
        'responses': [
            "I can help with coding questions! Could you share the specific code you're working with or describe the problem you're trying to solve?",
            "For coding help, it would be great if you could describe what you're trying to achieve and what issues you're encountering.",
            "I'd be happy to help with your code! Could you explain what you're working on and where you're stuck?",
            "To help with your coding question, could you provide more details about your project and what specific challenges you're facing?",
            "I can assist with coding problems! Please share what you're working on and what specific help you need."
        ]
    },
    'beginner_questions': {
        'patterns': ['beginner', 'starting out', 'new to', 'never coded', 'complete beginner', 'absolute beginner'],
        'responses': [
            "Welcome to the world of coding! The best way to start is with the basics. Would you like to begin with web development (HTML/CSS) or general programming (Python)?",
            "As a beginner, you're at the start of an exciting journey! Should we focus on web development basics or general programming concepts first?",
            "Everyone starts somewhere in coding! Would you like to learn about web basics (HTML), or start with a beginner-friendly language like Python?",
            "For beginners, I recommend starting with either web basics (HTML/CSS) or Python. Which interests you more: building websites or general programming?",
            "Welcome to programming! To help guide you better, what interests you more: creating websites, building applications, or working with data?"
        ]
    },
    'practice_exercises': {
        'patterns': ['practice exercises', 'exercises', 'practice coding', 'coding exercises', 'coding practice', 'practice problems'],
        'responses': [
            "For Python practice exercises, I recommend starting with: 1) CodeWars - great for bite-sized challenges, 2) HackerRank - structured practice problems, 3) Python Challenge - fun algorithmic puzzles. Which interests you?",
            "Here are some great Python practice resources: 1) LeetCode for algorithm practice, 2) Exercism.io for mentored exercises, 3) Project Euler for math-focused problems. Would you like to know more about any of these?",
            "For practicing Python, try these: 1) CodingBat for beginners, 2) Codewars for intermediate challenges, 3) HackerRank's Python track. Would you like specific problem recommendations?",
            "I recommend these Python practice platforms: 1) Exercism.io - great feedback system, 2) PyBites - real-world challenges, 3) CheckiO - game-like exercises. Which style appeals to you?",
            "Here are my top Python practice recommendations: 1) Codewars - daily challenges, 2) HackerRank - structured learning, 3) Project Euler - mathematical focus. Would you like details about any of these?"
        ]
    }
}

def is_valid_token(token: str, generated_tokens: Set[str]) -> bool:
    """Check if a token is valid for generation."""
    token_lower = token.lower()
    
    # Allow technical terms if they're in a learning context
    if is_learning_context(token_lower):
        return True
    
    # Skip technical terms
    if any(tech_term in token_lower for tech_term in TECHNICAL_TERMS):
        return False
    
    # Skip tokens with special characters
    if any(c in token for c in '!@#$%^&*()_+-=[]{}|;:,.<>?/\\'):
        return False
    
    # Skip long tokens
    if len(token) > 12:
        return False
    
    # Skip tokens that look like paths or URLs
    if '/' in token or '\\' in token or '.' in token:
        return False
    
    # Skip numeric tokens
    if any(c.isdigit() for c in token):
        return False
    
    # Strongly prefer common words after the first few tokens
    if len(generated_tokens) >= 2:
        return token_lower in COMMON_WORDS
    
    return True

# Fallback responses when no pattern matches and model generation fails
FALLBACK_RESPONSES = [
    "I understand you're asking something, but could you please rephrase that? I want to make sure I help you correctly.",
    "I want to help you, but I'm not quite sure what you're asking. Could you explain it differently?",
    "I'm not sure I understood that correctly. Could you ask your question in a different way?",
    "To better assist you, could you please provide more details about what you're looking for?",
    "I'd like to help, but I need a bit more clarity. Could you elaborate on your question?"
]

def get_pattern_response(user_input: str) -> str:
    """Check if the input matches any common conversation patterns and return appropriate response."""
    user_input_lower = user_input.lower().strip()
    
    # Check each pattern category
    for category in CONVERSATION_PATTERNS.values():
        if any(pattern in user_input_lower for pattern in category['patterns']):
            responses = category['responses']
            return responses[hash(user_input) % len(responses)]
    
    return ""

def get_response(user_input: str, max_gen_len: int = 15, temperature: float = 0.6) -> str:
    """Generate a response using controlled sampling."""
    # First check for common conversation patterns
    pattern_response = get_pattern_response(user_input)
    if pattern_response:
        return pattern_response
    
    input_ids = tokenize(user_input, word2id, max_len=MAX_LEN)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        # Initial forward pass
        _, hidden = model(input_tensor)
        
        # Generation
        generated_ids = []
        generated_tokens = set()
        last_token = torch.tensor([[BOS_ID]], dtype=torch.long)
        
        for _ in range(max_gen_len):
            # Get next token probabilities
            logits, hidden = model(last_token, hidden)
            logits = logits / temperature
            
            # Filter vocabulary based on valid tokens
            valid_mask = torch.ones(1, VOCAB_SIZE, dtype=torch.bool)
            for i in range(VOCAB_SIZE):
                token = id2word.get(i, "<UNK>")
                if not is_valid_token(token, generated_tokens):
                    valid_mask[0, i] = False
            
            # Apply mask and sample
            logits[~valid_mask] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            if next_token_id == EOS_ID or next_token_id == PAD_ID:
                break
                
            token = id2word.get(next_token_id, "<UNK>")
            if token == "<UNK>" or token in generated_tokens:
                continue
                
            generated_ids.append(next_token_id)
            generated_tokens.add(token.lower())
            last_token = torch.tensor([[next_token_id]], dtype=torch.long)
            
            # Stop if we have a reasonable response
            if len(generated_tokens) >= 3 and any(token.endswith(p) for p in ['.', '!', '?']):
                break
        
        # Format response
        response = detokenize(generated_ids, id2word).strip()
        
        # Use fallback responses if generation failed or is too short
        if not response or len(response.split()) < 2:
            return FALLBACK_RESPONSES[hash(user_input) % len(FALLBACK_RESPONSES)]
            
        # Clean and format response
        response = ' '.join(word.strip('.,!?') + ('.' if '.' in word else '!' if '!' in word else '?' if '?' in word else '')
                          for word in response.split())
        response = response.strip()
        if not any(response.endswith(p) for p in ['.', '!', '?']):
            response += '.'
            
        # If response looks nonsensical, use fallback
        words = response.lower().split()
        if len(set(words) & COMMON_WORDS) < len(words) * 0.5:  # If less than 50% are common words
            return FALLBACK_RESPONSES[hash(user_input) % len(FALLBACK_RESPONSES)]
            
        return response[0].upper() + response[1:]

# === Command Line Test ===
if __name__ == "__main__":
    print("🤖 PyTorch Chatbot ready! Type 'exit' to quit.")
    while True:
        try:
            user_input = input("🧑 You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            print("🤖 Bot:", get_response(user_input))
        except KeyboardInterrupt:
            print("\n🚪 Exiting...")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
