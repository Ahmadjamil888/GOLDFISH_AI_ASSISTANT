# Smart AI Chatbot

A sophisticated chatbot system built with Python, leveraging modern NLP techniques and deep learning for intelligent conversation handling. The system processes user inputs, generates contextually relevant responses, and maintains conversation history.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Technical Architecture](#technical-architecture)
4. [Dataset Analysis](#dataset-analysis)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Development](#development)
8. [Data Visualization](#data-visualization)

## Overview

This project implements an AI-powered chatbot with a modern web interface. The system uses advanced natural language processing techniques to understand user queries and generate appropriate responses. The implementation includes a Flask-based web server, a deep learning model for response generation, and a clean, responsive user interface.

## Features

- **Intelligent Response Generation**: Context-aware responses using advanced NLP models
- **Modern Web Interface**: Clean, responsive design using Tailwind CSS
- **Chat History Management**: Save, load, and manage conversation history
- **Real-time Processing**: Immediate response generation and display
- **Data Visualization**: Comprehensive analysis of conversation patterns

## Technical Architecture

### Backend Components

- **Web Server**: Flask-based REST API
- **Model**: Enhanced chatbot implementation with custom response generation
- **Data Processing**: Tokenization and text preprocessing utilities
- **Visualization**: Data analysis and plotting capabilities

### Frontend Components

- **Interface**: Modern, responsive design with Tailwind CSS
- **Chat Management**: Dynamic chat history and session handling
- **Real-time Updates**: Asynchronous message processing
- **Error Handling**: Robust error management and user feedback

## Dataset Analysis

The chatbot is trained on a comprehensive dataset containing prompt-response pairs. Here are key statistics from our dataset:

- Total Entries: 23,690
- Data Format: CSV
- Columns: prompt, response

### Data Distribution

The dataset has been analyzed for various metrics including length distribution and word counts. Below are the visualization results:

#### Prompt and Response Length Analysis
![Prompt Length Distribution](<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/143940f8-898d-45e4-8dc0-9405be5d97b2" />
)
*Distribution of prompt lengths showing the variety in user inputs*

![Response Length Distribution](<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/e4e74417-a440-48bc-9fdb-d47aada8b007" />
)
*Distribution of response lengths demonstrating output variability*

#### Word Count Analysis
![Prompt Word Count](<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/c3777ae6-5c5d-48cb-815f-ca139a20ae4b" />
)
*Word count distribution in user prompts*

![Response Word Count](<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/9e5d111a-2f28-4b3e-8072-61d10082e54c" />
)
*Word count distribution in system responses*

#### Correlation Analysis
![Length Correlation](<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/be764663-f3bf-415b-b4db-073dd29aa02a" />
)
*Correlation between prompt and response lengths*

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ahmadjamil888/GOLDFISH_AI_ASSISTANT.git
   cd my-smart-ai-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask server:
   ```bash
   python app/app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Begin chatting with the AI assistant through the web interface.

## Development

### Project Structure

```
my-smart-ai-chatbot/
├── app/
│   ├── app.py
│   └── templates/
│       └── chat.html
├── data/
│   ├── cleaned_dataset.csv
│   ├── tokenized_prompts.pkl
│   ├── tokenized_responses.pkl
│   └── plots/
├── model/
│   ├── infer.py
│   └── model_weights.pth
├── utils/
│   ├── tokenizer.py
│   └── visualize_data.py
├── requirements.txt
└── README.md
```

### Key Components

- `app/app.py`: Main Flask application
- `app/templates/chat.html`: Web interface template
- `model/infer.py`: Model inference implementation
- `utils/`: Utility functions and tools
- `data/`: Dataset and visualization storage

## Data Visualization

The project includes comprehensive data visualization tools:

- Distribution analysis of prompt and response lengths
- Word count statistics and distributions
- Correlation analysis between inputs and outputs
- Interactive plotting capabilities

To generate visualizations:

```bash
python utils/visualize_data.py
```

This will create plots in the `data/plots/` directory and display summary statistics.
