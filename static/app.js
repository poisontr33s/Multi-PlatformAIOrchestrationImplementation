document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const promptInput = document.getElementById("prompt-input");
    const sendButton = document.getElementById("send-button");
    const providerSelect = document.getElementById("provider-select");

    sendButton.addEventListener("click", sendMessage);
    promptInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

    function addMessage(text, sender) {
        const message = document.createElement("div");
        message.classList.add("message", `${sender}-message`);
        message.textContent = text;
        chatBox.appendChild(message);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function sendMessage() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        addMessage(prompt, "user");
        promptInput.value = "";

        const provider = providerSelect.value;

        try {
            const response = await fetch("http://127.0.0.1:8000/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ prompt, provider }),
            });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const data = await response.json();
            if (data.response) {
                addMessage(data.response, "ai");
            } else {
                addMessage(data.error || "An unknown error occurred.", "ai");
            }
        } catch (error) {
            console.error("Error:", error);
            addMessage("An error occurred while fetching the response.", "ai");
        }
    }
});
