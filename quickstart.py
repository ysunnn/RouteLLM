import os

from routellm.controller import Controller
from pprint import pprint

os.environ["OPENAI_API_KEY"] = ""

client = Controller(
  routers=["mf", "random"],
  strong_model="gpt-4-1106-preview",
  weak_model="ollama_chat/mistral",
)

response = client.chat.completions.create_with_judge(
  # This tells RouteLLM to use the MF router with a cost threshold of 0.11593
  model="router-mf-0.11593",
  messages=[
    {"role": "user", "content": "what is the capital city of germany?"}
  ]
)

pprint(response)