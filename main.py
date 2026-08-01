from dotenv import load_dotenv
import os

load_dotenv()
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-OpenRouter-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="~openai/gpt-latest",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ],
  max_tokens=1000,
  temperature=0.7,
)

print(completion.choices[0].message.content)