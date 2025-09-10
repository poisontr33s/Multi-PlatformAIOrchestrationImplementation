import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

def get_anthropic_client():
    """
    Initializes and returns the Anthropic client.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")

    client = Anthropic(api_key=api_key)
    return client

def get_claude_completion(prompt: str, model: str = "claude-3-opus-20240229"):
    """
    Gets a completion from the Anthropic Claude API using the messages endpoint.
    """
    client = get_anthropic_client()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
