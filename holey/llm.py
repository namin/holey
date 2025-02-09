import ollama

model = 'qwen2.5-coder'

def generate(prompt, max_tokens=100, temperature=1.0):
    print(f"Sending request to ollama (model={model}, max_tokens={max_tokens}, temp={temperature})")
    print(f"Prompt:\n{prompt}")
    
    try:
        response = ollama.generate(
            model=model, prompt=prompt,
            options={
                'max_tokens': max_tokens,
                'temperature': temperature
            }
        )
        print("Received response from ollama")
        print(f"Response:\n{response['response']}")
        return response['response']
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def extract_code_blocks(response: str) -> str:
    """Extract code blocks from LLM response, removing markdown and explanations."""
    if "```" in response:
        lines = response.split("```")[1:]
        lines = [lines[i] for i in range(0, len(lines)) if i % 2 == 0]
        lines = ["\n".join(line.split('\n')[1:]) if '\n' in line else line for line in lines]
        blocks = lines
    else:
        code = response.strip()
        blocks = [code]
    return blocks
