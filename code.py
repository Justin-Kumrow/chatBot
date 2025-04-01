import ollama

def chat_with_character(character_name, model_name="llama3"):
    """
    Start a conversation with a specified character using Ollama.
    
    Args:
        character_name (str): Name or description of the character you want to chat with
        model_name (str): Ollama model to use (default: "llama3")
    """
    # Initial system prompt to set up the character
    system_prompt = f"""You are {character_name}. Stay completely in character at all times. 
    Respond as if you are this character, with their personality, knowledge, and speech patterns.
    Don't break character or acknowledge you're an AI.l"""
    
    print(f"\nStarting conversation with {character_name} (using {model_name})...")
    print("Type 'quit' to end the conversation.\n")
    
    messages = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'assistant',
            'content': f"*{character_name} appears before you* Hello! I'm {character_name}. What would you like to talk about?"
        }
    ]
    
    print(f"{character_name}: Hello! I'm {character_name}. What would you like to talk about?")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print(f"\n{character_name}: Goodbye! It was nice talking with you.")
            break
            
        messages.append({'role': 'user', 'content': user_input})
        
        # Stream the response for a more interactive feel
        print(f"\n{character_name}: ", end="", flush=True)
        
        response = ""
        for chunk in ollama.chat(
            model=model_name,
            messages=messages,
            stream=True
        ):
            chunk_content = chunk['message']['content']
            print(chunk_content, end="", flush=True)
            response += chunk_content
            
        print("\n")
        
        messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    # Example usage
    print("Welcome to Character Chat with Ollama!")
    character = input("Who would you like to talk to? (e.g., 'Sherlock Holmes', 'a wise wizard', 'a friendly robot'): ")
    model = "mannix/llama3.1-8b-abliterated"
    
    chat_with_character(character, model)
