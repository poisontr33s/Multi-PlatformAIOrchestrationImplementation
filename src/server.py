from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from orchestrator import AIProvider, orchestrate_chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    prompt: str
    provider: AIProvider = AIProvider.OPENAI

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    response = orchestrate_chat(request.prompt, request.provider)

    if response:
        return {"response": response}
    else:
        return {"error": f"Failed to get a response from {request.provider.value}."}

# This will serve the static files from the 'static' directory.
# The 'static' directory should be at the root of the project.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
