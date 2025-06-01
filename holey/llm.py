import os
from typing import List

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY')
PROJECT_ID = os.environ.get('PROJECT_ID') # for Google Cloud

def is_docker():
    path = '/proc/self/cgroup'
    return os.path.exists('/.dockerenv') or (os.path.exists(path) and any('docker' in line for line in open(path)))

# Set Ollama base URL conditionally
if is_docker():
    os.environ['OLLAMA_HOST'] = 'http://host.docker.internal:11434'


def dummy_generate(pkg, extra=""):
    raise ValueError(f"Need to install pip package '{pkg}'"+extra)

generators = {}

def generate(prompt, max_tokens=1000, temperature=1.0, model=None):
    print(f"Prompt:\n{prompt}")
    return None
generators[None] = generate

if PROJECT_ID:
    generate = None
    try:
        from anthropic import AnthropicVertex
    except ModuleNotFoundError:
        generate = dummy_generate('anthropic[vertex]')

    if generate is None:
        LOCATION="us-east5"
        def generate(prompt, max_tokens=1000, temperature=1.0, model="claude-sonnet-4@20250514"):
            print(f"Sending request to Anthropic Vertex (model={model}, max_tokens={max_tokens}, temp={temperature})")

            client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)

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
            print("Received response from Anthropic Vertex")
            print(f"Response:\n{message}")
            return message.content[0].text

    generators['claude_vertex'] = generate

if OPENAI_API_KEY:
    generate = None
    try:
        import openai
    except ModuleNotFoundError:
        generate = dummy_generate('openai')
    if generate is None:
        OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL')
        if OPENAI_BASE_URL:
            openai.base_url = OPENAI_BASE_URL
        def generate(prompt, max_tokens=1000, temperature=1.0, model="gpt-4o"):
            print(f"Sending request to OpenAI (model={model}, max_tokens={max_tokens}, temp={temperature})")

            completion = openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            response = completion.choices[0].message.content
            print("Received response from OpenAI")
            print(f"Response:\n{response}")
            return response
    generators['openai'] = generate

if ANTHROPIC_API_KEY:
    generate = None
    try:
        import anthropic
    except ModuleNotFoundError:
        generate = dummy_generate('anthropic')
    if generate is None:
        def generate(prompt, max_tokens=1000, temperature=1.0, model="claude-3-7-sonnet-20250219"):
            print(f"Sending request to Anthropic (model={model}, max_tokens={max_tokens}, temp={temperature})")

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

    generators['claude'] = generate

if GEMINI_API_KEY:
    generate = None
    try:
        from google import genai
    except ModuleNotFoundError:
        generate = dummy_generate('google-genai')
    if generate is None:
        def generate(prompt, max_tokens=1000, temperature=1.0, model="gemini-2.5-flash-preview-04-17"):
            print(f"Sending request to Google Gemini (model={model}, max_tokens={max_tokens}, temp={temperature})")
            
            client = genai.Client(api_key=GEMINI_API_KEY)

            response = client.models.generate_content(
                model='gemini-2.0-flash', contents=prompt
            )
            text = response.text
            print("Received response from Google Gemini")
            print(f"Response:\n{text}")
            return text

    generators['gemini'] = generate

if OLLAMA_API_KEY:
    generate = None
    try:
        import ollama
    except ModuleNotFoundError:
        generate = dummy_generate('ollama', extra=", or package 'anthropic' while setting ANTHROPIC_API_KEY")
    if generate is None:
        model = os.environ.get('OLLAMA_MODEL', 'gemma3:27b-it-qat')
        def generate(prompt, max_tokens=1000, temperature=1.0, model=model):
            print(f"Sending request to Ollama (model={model}, max_tokens={max_tokens}, temp={temperature})")

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

    generators['ollama'] = generate

def extract_code_blocks(response: str) -> List[str]:
    """Extract code blocks from LLM response, removing markdown and explanations."""
    if not response:
        return []
    if "```" in response:
        lines = response.split("```")[1:]
        lines = [lines[i] for i in range(0, len(lines)) if i % 2 == 0]
        lines = ["\n".join(line.split('\n')[1:]) if '\n' in line else line for line in lines]
        blocks = lines
    elif "`" in response:
        lines = response.split("`")[1:]
        lines = [lines[i] for i in range(0, len(lines)) if i % 2 == 0]
        lines = ["\n".join(line) if '\n' in line else line for line in lines]
        blocks = lines
    else:
        code = response.strip()
        blocks = [code]
    return blocks
