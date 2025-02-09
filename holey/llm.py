import os
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

if not ANTHROPIC_API_KEY:
    import ollama
    def generate(prompt, max_tokens=1000, temperature=1.0, model='qwen2.5-coder'):
        print(f"Sending request to Ollama (model={model}, max_tokens={max_tokens}, temp={temperature})")
        print(f"Prompt:\n{prompt}")

        try:
            response = ollama.generate(
                model=model, prompt=prompt,
                options={
                    'max_tokens': max_tokens,
                    'temperature': temperature
                }
            )
            print("Received response from Ollama")
            print(f"Response:\n{response['response']}")
            return response['response']
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
else:
    import anthropic
    def generate(prompt, max_tokens=1000, temperature=1.0, model="claude-3-5-sonnet-20241022"):
        print(f"Sending request to Anthropic (model={model}, max_tokens={max_tokens}, temp={temperature})")
        print(f"Prompt:\n{prompt}")

        client = anthropic.Anthropic()

        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are an SMTLIB expert.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        print("Received response from Anthropic")
        print(f"Response:\n{message}")
        return message.content[0].text

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
