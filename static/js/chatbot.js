document.addEventListener("DOMContentLoaded", function () {
    const inputField = document.getElementById("input-field");
    const submitButton = document.getElementById("submit-button");
    const conversation = document.getElementById("conversation");
    const form = document.getElementById("input-form");
    const conversationWrapper = document.getElementById("conversation-wrapper");
    const header = document.getElementById("header");
    const inputContainer = document.getElementById("input-container");

    // Function to calculate and set the height of the conversation area
    function setConversationHeight() {
        const headerHeight = header.offsetHeight;
        const inputContainerHeight = inputContainer.offsetHeight;
        const availableHeight = window.innerHeight - headerHeight - inputContainerHeight;
        conversationWrapper.style.height = `${availableHeight}px`;
    }

    // Set initial height
    setConversationHeight();

    // Update height on window resize
    window.addEventListener("resize", setConversationHeight);

    // Function to append a message
    function appendMessage(text, sender) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add(sender === "user" ? "user-message" : "chatbot-message");
        messageDiv.innerHTML = `<p class="${sender}-text"></p>`;
        conversation.appendChild(messageDiv);

        let textElement = messageDiv.querySelector(`.${sender}-text`);

        if (sender === "user") {
            // For user messages, display the text directly (no animation)
            textElement.textContent = text;
        } else {
            // For chatbot messages, display the text dynamically (with animation)
            displayTextDynamically(text, textElement);
        }

        autoScroll(); // Ensure it stays in view
    }

    // Function to display text dynamically (letter by letter) - for chatbot responses only
    function displayTextDynamically(text, element) {
        let index = 0;
        let previousHeight = element.scrollHeight; // Track the height of the element

        let interval = setInterval(() => {
            if (index < text.length) {
                element.textContent += text[index];
                index++;

                // Check if the height of the element has changed
                const newHeight = element.scrollHeight;
                if (newHeight > previousHeight) {
                    autoScroll(); // Scroll if the height increases
                    previousHeight = newHeight; // Update the previous height
                }
            } else {
                clearInterval(interval);
                autoScroll(); // Final scroll after text is fully generated
            }
        }, 20); // Adjust speed if needed
    }

    // Function to auto-scroll to the bottom
    function autoScroll() {
        const isNearBottom =
            conversationWrapper.scrollHeight - conversationWrapper.clientHeight <=
            conversationWrapper.scrollTop + 50; // 50px threshold

        if (isNearBottom) {
            conversationWrapper.scrollTop = conversationWrapper.scrollHeight;
        }
    }

    // Function to show typing indicator
    function showTypingIndicator() {
        const typingDiv = document.createElement("div");
        typingDiv.classList.add("chatbot-message", "typing-indicator");
        typingDiv.id = "typing-indicator";
        typingDiv.innerHTML = ` 
            <span class="typing-dots">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </span>
        `;
        conversation.appendChild(typingDiv);
        autoScroll();
    }

    // Function to remove typing indicator
    function removeTypingIndicator() {
        const typingDiv = document.getElementById("typing-indicator");
        if (typingDiv) typingDiv.remove();
    }

    // Send message function
    async function sendMessage(event) {
        event.preventDefault();
        const userMessage = inputField.value.trim();
        if (!userMessage) return;

        appendMessage(userMessage, "user"); // Show user message directly (no animation)
        inputField.value = ""; // Clear input field
        showTypingIndicator(); // Show typing indicator

        try {
            const response = await fetch("/chatbot", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: userMessage }),
            });

            removeTypingIndicator(); // Remove typing indicator

            if (!response.ok) throw new Error("Response not OK");

            const data = await response.json();
            appendMessage(data.reply, "chatbot"); // Show chatbot response dynamically
        } catch (error) {
            removeTypingIndicator();
            appendMessage("Error: Unable to connect. Please try again.", "chatbot");
            console.error("Fetch error:", error);
        }
    }

    // Event Listeners
    form.addEventListener("submit", sendMessage);

    // Optional: Scroll to the bottom of the chat when the page loads
    window.addEventListener("load", function () {
        autoScroll();
    });
});