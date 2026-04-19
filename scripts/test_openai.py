from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

resp = client.responses.create(
    model="gpt-5.4-mini",
    input="2+2を一言で答えて"
)

print(resp.output_text)
