import unittest
from unittest.mock import patch
from orchestrator import orchestrate_chat, AIProvider

class TestOrchestrator(unittest.TestCase):

    @patch('orchestrator.get_openai_completion')
    def test_orchestrate_chat_openai(self, mock_get_completion):
        mock_get_completion.return_value = "Hello from OpenAI"
        response = orchestrate_chat("Hello", AIProvider.OPENAI)
        self.assertEqual(response, "Hello from OpenAI")
        mock_get_completion.assert_called_once_with("Hello")

    @patch('orchestrator.get_gemini_completion')
    def test_orchestrate_chat_google(self, mock_get_completion):
        mock_get_completion.return_value = "Hello from Google"
        response = orchestrate_chat("Hello", AIProvider.GOOGLE)
        self.assertEqual(response, "Hello from Google")
        mock_get_completion.assert_called_once_with("Hello")

    @patch('orchestrator.get_claude_completion')
    def test_orchestrate_chat_claude(self, mock_get_completion):
        mock_get_completion.return_value = "Hello from Claude"
        response = orchestrate_chat("Hello", AIProvider.CLAUDE)
        self.assertEqual(response, "Hello from Claude")
        mock_get_completion.assert_called_once_with("Hello")

if __name__ == '__main__':
    unittest.main()
