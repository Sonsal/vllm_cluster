import aiohttp
import asyncio
import json
import time
from datasets import load_dataset

# Замените 'your_api_key' на ваш реальный API-ключ OpenAI
API_KEY = 'Empty'
API_URL = 'http://localhost:8000/v1/chat/completions'

async def fetch(session, prompt):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    data = {
        "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.01,
        "max_tokens":512
    }

    async with session.post(API_URL, headers=headers, json=data) as response:
        return await response.json()

async def main(dataset, num_requests):
    prompts = dataset['train']['question'][0:num_requests]
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        # for response in responses:
        #     print(json.dumps(response, indent=2))

if __name__ == '__main__':
    num_requests = 10

    dataset = load_dataset('openai/gsm8k', 'main')

    start = time.time()
    asyncio.run(main(dataset, num_requests))
    print(f'Exec time: {time.time() - start} seconds.\n Num requests: {num_requests}')
