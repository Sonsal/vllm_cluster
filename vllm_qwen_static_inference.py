import argparse
import json
import sys
import threading
import torch
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def format_prompt(prompt, tokenizer):
    messages = [
                    {
                        "role": "system", 
                        "content": "You are ProcesetAI, created by Infomaximum AI Team. You are a helpful assistant."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
    text = tokenizer.apply_chat_template(
                                            messages,
                                            tokenize=False,
                                            add_generation_prompt=True
                                        )
    return text

def process_input(model, tokenizer, input_data) -> None:  # Классы нужно прописать!
    input_data = json.loads(input_data)
    # DEV Добавить системные поля для параметров генерации в JSON (количество токенов, ...)
    guid = 123
    try:
        stop_token_ids = None

        inputs = [format_prompt(sub_prompt, tokenizer) for sub_prompt in input_data.split('<batch>')]

        sampling_params = SamplingParams(
                                            temperature=0.01,
                                            max_tokens=512,
                                            stop_token_ids=stop_token_ids
                                        )
        
        start_time = time.time()
        outputs = model.generate(
                                    inputs, 
                                    sampling_params=sampling_params
                                )
        print(f"Generation time: {time.time() - start_time} seconds.\n", flush=True)
        sys.stdout.flush()

        output_text = '<batch>'.join([o.outputs[0].text for o in outputs])
        data = {
                    "guid": guid,
                    "answer": output_text
                }
    except Exception as err:
        data = {
                    "guid": guid,
                    "error": str(err)
                }
    
    json_data = json.dumps(data, ensure_ascii=False)

    response = json_data + "\n"
    print(response, flush=True)
    sys.stdout.flush()

def listen_for_input(llm, tokenizer):
    while True:
        input_data = sys.stdin.readline().strip()
        new_thread = threading.Thread(target=process_input(llm, tokenizer, input_data))
        new_thread.start()


def main():
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type = str,
                        default='.cache/models/MyBestModel',
                        help = 'Path to model folder')

    args = parser.parse_args()

    # Выводим значения аргументов
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}\n', flush=True)
        sys.stdout.flush()

    # Поддержка квантизации
    quantization = None
    load_format = 'auto'
    if args.model.lower().find('awq') != -1:
        quantization = 'awq'
    elif args.model.lower().find('gptq') != -1:
        quantization = 'gptq'
    elif args.model.lower().find('fp8') != -1:
        quantization = 'fp8'
    elif args.model.lower().find('bnb') != -1:
        quantization = 'bitsandbytes'
        load_format="bitsandbytes"
    else:
        pass

    llm = LLM(model=MODEL_NAME)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"vLLM engine started!\n", flush=True)
    sys.stdout.flush()

    # Запуск потока слушателя ввода пользователя
    input_thread = threading.Thread(target=listen_for_input(llm, tokenizer))
    input_thread.start()


# Запуск приложения
if __name__ == "__main__":
    main()