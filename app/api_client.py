# app/api_client.py

import os
from groq import Groq
from dotenv import load_dotenv
from logger import CustomLogger  # Import your custom logger

load_dotenv(override=True)

class GroqClient:
    """Class to interact with the Groq API."""

    def __init__(self):
        self.api_key = os.getenv('API_KEY') or os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=self.api_key)
        self.logger = CustomLogger().get_logger()  # Initialize your custom logger

    def get_response(self, messages):
        """
        Send messages to the Groq API and return the response.

        :param messages: List of messages for the conversation.
        :return: AI response as a string.
        """
        try:
            self.logger.info("Sending messages to Groq API...")
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile"
            )
            response = chat_completion.choices[0].message.content
            self.logger.info("Received response from Groq API.")
            return response
        except Exception as e:
            self.logger.error(f"Error communicating with Groq API: {e!r}")
            return "Sorry, I couldn't get a response at this time."