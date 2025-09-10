# AI LLM Orchestrator

A simple orchestrator to interact with various Large Language Models (LLMs) from different providers.

## Getting Started

This project uses `uv` for package management.

### Prerequisites

- Python 3.8+
- `uv`: You can install it with `pip install uv` or by following the official instructions on the [uv website](https://github.com/astral-sh/uv).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/poisontr33s/Multi-PlatformAIOrchestrationImplementation.git
    cd Multi-PlatformAIOrchestrationImplementation
    ```

2.  **Create a virtual environment:**
    ```bash
    uv venv
    ```

3.  **Activate the virtual environment:**
    - On macOS and Linux:
      ```bash
      source .venv/bin/activate
      ```
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```

4.  **Install dependencies:**
    ```bash
    uv pip install -e .
    ```

## Usage

## Usage

This project provides two ways to interact with the AI models: a command-line interface (CLI) and a web interface.

### CLI

To use the CLI, run the following command:

```bash
ai-orchestrator chat "Your prompt here" --provider <provider>
```

Replace `"Your prompt here"` with your prompt and `<provider>` with one of `openai`, `google`, or `claude`.

### Web Interface

To use the web interface, first start the server:

```bash
uvicorn src.server:app --reload
```

Then, open your web browser and navigate to `http://127.0.0.1:8000`.

## API Keys

To use the AI models, you will need to provide API keys for the respective services. This project uses a `.env` file to manage environment variables.

1.  Create a file named `.env` in the root of the project.
2.  Add your API keys to the `.env` file like this:

```
OPENAI_API_KEY="your_openai_api_key"
GOOGLE_API_KEY="your_google_api_key"
ANTHROPIC_API_KEY="your_anthropic_api_key"
```

**Note:** The `.env` file is included in the `.gitignore` file, so your keys will not be committed to the repository.
