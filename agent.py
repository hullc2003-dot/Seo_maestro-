import os
import sys
from groq import Groq

# --- Configuration ---
API_KEY = "GROQ_API_KEY"
MODEL   = "llama-3.3-70b-versatile"
SYSTEM  = "You are a helpful assistant."

# --- Init client ---
client = Groq(api_key=API_KEY)

# --- Conversation history ---
history = [{"role": "system", "content": SYSTEM}]

def chat(user_input):
    history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        temperature=0.7,
        max_tokens=1024,
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply

def main():
    print("=" * 40)
    print(f"  Llama 3.3 70B Chat  ")
    print("=" * 40)
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'clear' to reset history.")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue
        elif user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        elif user_input.lower() == "clear":
            history.clear()
            history.append({"role": "system", "content": SYSTEM})
            print("History cleared.")
            continue

        try:
            reply = chat(user_input)
            print(f"\nLlama: {reply}")
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
