import os
import sys
import random
import re

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Knowledge base
KNOWLEDGE_BASE = {
    'capabilities': [
        "I can help you with:",
        "1. Web Development (HTML, CSS, JavaScript)",
        "2. Programming and coding guidance",
        "3. Answer questions about technology",
        "4. Provide tutorials and learning resources",
        "5. Explain technical concepts",
        "6. Help with development tools and frameworks"
    ],
    'web_development': {
        'getting_started': """To start making a website, you'll need to learn these fundamental technologies:

1. HTML - For structuring your web content
2. CSS - For styling and design
3. JavaScript - For interactivity and functionality

Here's how to begin:
1. Start with HTML basics (structure, tags, elements)
2. Learn CSS for styling (colors, layout, responsive design)
3. Add JavaScript for interactivity
4. Use a code editor like VS Code or Sublime Text
5. Learn about responsive design and mobile-first approach

Would you like specific information about any of these topics?""",
        
        'html': """HTML (HyperText Markup Language) is the foundation of web development. Here are the basics:

1. Structure:
   <!DOCTYPE html>
   <html>
     <head>
       <title>Your Page Title</title>
     </head>
     <body>
       <h1>Your Content Here</h1>
     </body>
   </html>

2. Common Elements:
   - <h1> to <h6> for headings
   - <p> for paragraphs
   - <div> for divisions/sections
   - <a> for links
   - <img> for images

Would you like to know more about specific HTML elements or move on to CSS?""",

        'css': """CSS (Cascading Style Sheets) is used for styling websites. Here are the basics:

1. Ways to add CSS:
   - Inline: <div style="color: blue;">
   - Internal: <style> in HTML
   - External: separate .css file (recommended)

2. Basic Syntax:
   selector {
     property: value;
   }

3. Common Properties:
   - color: for text color
   - background-color: for background
   - font-size: for text size
   - margin/padding: for spacing
   - display: for layout control

4. Modern Features:
   - Flexbox for layouts
   - Grid for complex designs
   - Media queries for responsive design

Would you like specific examples or information about any of these concepts?""",

        'next_steps': """After learning HTML & CSS basics, here are recommended next steps:

1. Learn a CSS Framework:
   - Tailwind CSS
   - Bootstrap
   - Material UI

2. Add JavaScript for:
   - User interaction
   - Form validation
   - Dynamic content
   - API integration

3. Learn Development Tools:
   - Git for version control
   - VS Code or similar editor
   - Browser Developer Tools
   - Package managers (npm)

Would you like more details about any of these topics?"""
    }
}

def get_response(user_input):
    # Convert to lowercase for easier matching
    user_input = user_input.lower().strip()
    
    # Greetings
    greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    if user_input in greetings:
        responses = [
            "Hello! How can I assist you with web development today?",
            "Hi there! Would you like to learn about building websites?",
            "Hey! I can help you with web development. What would you like to know?"
        ]
        return random.choice(responses)
    
    # Web Development Questions
    web_dev_patterns = {
        'getting_started': [
            r'(?:how to|want to|help|can you|).*(?:make|create|build|develop).*(?:website|webpage|web page)',
            r'(?:start|begin|learn).*(?:web|website|development)',
            r'(?:web development|website).*(?:basics|fundamentals|start)'
        ],
        'html': [
            r'(?:what is|learn|teach|show|tell|about).*(?:html)',
            r'html.*(?:basics|tutorial|guide|help)',
            r'(?:html).*(?:elements|tags|structure)'
        ],
        'css': [
            r'(?:what is|learn|teach|show|tell|about).*(?:css)',
            r'css.*(?:basics|tutorial|guide|help|styling)',
            r'(?:style|design).*(?:website|webpage|web page)',
            r'(?:css).*(?:properties|selectors|styles)'
        ],
        'next_steps': [
            r'(?:what|whats|what\'s).*(?:next|after|more)',
            r'(?:advanced|next steps|continue).*(?:web|development)',
            r'(?:learn more|additional|extra).*(?:web|development)'
        ]
    }

    # Check each category of web development questions
    for category, patterns in web_dev_patterns.items():
        for pattern in patterns:
            if re.search(pattern, user_input):
                return KNOWLEDGE_BASE['web_development'][category]

    # What can you do
    if any(phrase in user_input for phrase in ['what can you do', 'what are your capabilities', 'what do you do', 'help me with']):
        return '\n'.join(KNOWLEDGE_BASE['capabilities'])
    
    # Default responses for other inputs
    default_responses = [
        "I can help you with web development. Would you like to learn about HTML, CSS, or getting started with websites?",
        "I'm here to help with web development. Would you like to know about creating websites, HTML, or CSS?",
        "I can guide you through web development. Would you like to start with HTML basics, CSS styling, or general website creation?"
    ]
    return random.choice(default_responses)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
            
        # Get appropriate response
        response = {
            'response': get_response(user_input)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error processing chat request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
