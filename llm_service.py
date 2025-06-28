# backend/llm_service.py
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_description(user_input, module_name):
    prompt = f"Module: {module_name}\nUser Input: {user_input}\n\nGive a technical and clear explanation of the system based on the input."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful telecom design assistant. You return technical, clear, and relevant explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )
    
    return response['choices'][0]['message']['content']
