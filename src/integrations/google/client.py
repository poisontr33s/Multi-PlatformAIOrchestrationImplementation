import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def get_google_client():
    """
    Initializes and returns the Google Generative AI client.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    genai.configure(api_key=api_key)
    return genai

def get_gemini_completion(prompt: str, model: str = "gemini-pro"):
    """
    Gets a completion from the Google Gemini API.
    """
    client = get_google_client()
    try:
        model = client.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
