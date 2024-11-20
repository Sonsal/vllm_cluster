from datasets import load_dataset
import json

dataset = load_dataset('openai/gsm8k', 'main')['train']['question'][0:150]

print(f'Total rows: {len(dataset)}')

dataset = {
    "prompts":dataset
}
with open('humaneval_prompts.json', 'w', encoding='utf-8') as fp:
    json.dump(dataset, fp, ensure_ascii=False)