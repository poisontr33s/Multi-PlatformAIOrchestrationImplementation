from enum import Enum
from integrations.openai.client import get_chat_completion as get_openai_completion
from integrations.google.client import get_gemini_completion
from integrations.claude.client import get_claude_completion

class AIProvider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    CLAUDE = "claude"

def orchestrate_chat(prompt: str, provider: AIProvider):
    """
    Routes a prompt to the specified AI provider and returns the response.
    """
    response = None
    if provider == AIProvider.OPENAI:
        response = get_openai_completion(prompt)
    elif provider == AIProvider.GOOGLE:
        response = get_gemini_completion(prompt)
    elif provider == AIProvider.CLAUDE:
        response = get_claude_completion(prompt)
    return response
