from transformers import AutoTokenizer, AutoModel
import torch
import json
import re
from typing import Dict, List, Optional

class EnhancedChatbot:
    def __init__(self):
        """Initialize the chatbot with comprehensive knowledge base."""
        # Basic conversation patterns
        self.knowledge_base = {
            "greetings": {
                "patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
                "responses": [
                    "Hello! How can I assist you today?",
                    "Hi there! What can I help you with?",
                    "Greetings! How may I help you?"
                ]
            },
            "farewell": {
                "patterns": ["goodbye", "bye", "see you", "farewell"],
                "responses": [
                    "Goodbye! Have a great day!",
                    "See you later! Take care!",
                    "Farewell! Feel free to return if you need more help!"
                ]
            }
        }
        
        # General knowledge base
        self.general_knowledge = {
            "transportation": {
                "airplane": {
                    "definition": "An airplane is a powered flying vehicle with fixed wings and a weight greater than that of the air it displaces.",
                    "components": "Key components include wings, fuselage (main body), engines, tail, landing gear, and cockpit.",
                    "how_it_works": "Airplanes fly using four main forces: lift (created by wings), thrust (provided by engines), drag (air resistance), and weight (gravity).",
                    "types": "Types include commercial airliners, military aircraft, private planes, and cargo aircraft.",
                    "history": "The first successful airplane was built by the Wright brothers in 1903, revolutionizing transportation.",
                    "usage": "Used for passenger transport, cargo delivery, military operations, and recreational flying."
                },
                "car": {
                    "definition": "A car is a wheeled motor vehicle used for transportation.",
                    "components": "Main parts include engine, transmission, wheels, chassis, and body.",
                    "types": "Including sedans, SUVs, sports cars, and electric vehicles."
                }
            },
            "animals": {
                "mammals": {
                    "definition": "Warm-blooded vertebrates that give birth to live young and produce milk.",
                    "examples": "Dogs, cats, elephants, whales, humans.",
                    "characteristics": "Hair/fur, warm-blooded, live birth, milk production."
                },
                "birds": {
                    "definition": "Warm-blooded vertebrates with feathers, wings, and beaks.",
                    "examples": "Eagles, sparrows, penguins, ostriches.",
                    "characteristics": "Feathers, wings, beaks, lay eggs."
                }
            },
            "nature": {
                "weather": {
                    "definition": "The state of the atmosphere at a particular time and place.",
                    "types": "Sunny, rainy, cloudy, snowy, windy.",
                    "factors": "Temperature, humidity, air pressure, precipitation."
                },
                "planets": {
                    "definition": "Large celestial bodies that orbit stars.",
                    "solar_system": "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
                    "characteristics": "Spherical shape, orbit around sun, affected by gravity."
                }
            }
        }
        
        # Technical knowledge (keeping existing tech knowledge)
        self.tech_knowledge = {
            "coding": {
                "definition": "Coding, or computer programming, is the process of creating instructions for computers to follow. It's how we create software, websites, apps, and other digital tools.",
                "languages": "Common programming languages include Python, JavaScript, Java, C++, and many others.",
                "importance": "Coding is essential in today's digital world as it powers everything from smartphones to websites to artificial intelligence.",
                "getting_started": "To start coding, you can learn basics of HTML/CSS for web development or Python for general programming."
            },
            "html": {
                "definition": "HTML (HyperText Markup Language) is the standard language for creating web pages and web applications.",
                "structure": "HTML uses tags to structure content on web pages, like headings, paragraphs, lists, and links.",
                "tags": {
                    "head": "The <head> tag contains metadata about the HTML document, including title, character set, styles, and scripts. It's not visible on the page itself.",
                    "body": "The <body> tag contains the visible content of the webpage, including text, images, and links.",
                    "div": "The <div> tag is used to create sections or divisions in the HTML document.",
                    "p": "The <p> tag defines a paragraph of text."
                }
            }
        }

    def get_response(self, user_input: str) -> dict:
        """Generate an informative response based on user input."""
        try:
            text = user_input.lower().strip()
            
            # Check for basic patterns first
            for category, data in self.knowledge_base.items():
                if any(pattern in text for pattern in data["patterns"]):
                    import random
                    return {"response": random.choice(data["responses"]), "status": "success"}

            # Process the question
            words = text.split()
            
            # Check for transportation questions
            if any(word in ["airplane", "aeroplane", "plane", "aircraft"] for word in words):
                airplane_info = self.general_knowledge["transportation"]["airplane"]
                if "definition" in text or "what is" in text:
                    return {
                        "response": f"{airplane_info['definition']}\n\nKey Components:\n{airplane_info['components']}\n\nHow it Works:\n{airplane_info['how_it_works']}",
                        "status": "success"
                    }
                elif "history" in text:
                    return {"response": airplane_info["history"], "status": "success"}
                elif "type" in text:
                    return {"response": airplane_info["types"], "status": "success"}
                else:
                    return {
                        "response": f"Here's what I know about airplanes:\n\n" +
                                  f"Definition: {airplane_info['definition']}\n\n" +
                                  f"How it Works: {airplane_info['how_it_works']}\n\n" +
                                  f"Types: {airplane_info['types']}\n\n" +
                                  f"Would you like to know more about any specific aspect?",
                        "status": "success"
                    }

            # Check for coding/technical questions (keeping existing logic)
            if "coding" in text:
                return {
                    "response": self.tech_knowledge["coding"]["definition"] + "\n\n" + 
                              self.tech_knowledge["coding"]["importance"] + "\n\n" +
                              "Popular programming languages include: " + self.tech_knowledge["coding"]["languages"],
                    "status": "success"
                }

            # If no specific match found, provide a helpful response
            return {
                "response": f"I understand you're asking about {text}. I can provide information about:\n" +
                          "- Transportation (airplanes, cars, etc.)\n" +
                          "- Animals (mammals, birds, etc.)\n" +
                          "- Nature (weather, planets, etc.)\n" +
                          "- Technology (coding, computers, etc.)\n\n" +
                          "Please specify which aspect you'd like to learn about.",
                "status": "success"
            }
            
        except Exception as e:
            return {
                "response": "I apologize, but I encountered an error. Please try rephrasing your question.",
                "status": "error"
            } 