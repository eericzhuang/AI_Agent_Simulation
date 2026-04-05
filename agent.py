# self designed AI-agent using Claude Sonnet

# import API key
from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()

# chat history
messages = []

# add to chat history
user_input = input("you: ")
messages.append({"role": "user", "content": user_input})

# generate response from Claude Sonnet model
response = client.messages.create(
    model="claude-sonnet-4-20250514", max_tokens=1024, messages=messages # input full chat history
)

# print output from the model
print("Claude: ", response.content[0].text)