import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_openai_client():
    """
    Initializes and returns the OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    client = OpenAI(api_key=api_key)
    return client

def get_chat_completion(prompt: str, model: str = "gpt-3.5-turbo"):
    """
    Gets a chat completion from the OpenAI API.
    """
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
