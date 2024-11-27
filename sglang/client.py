# from openai import OpenAI

# # Note: Ray Serve doesn't support all OpenAI client arguments and may ignore some.
# client = OpenAI(
#     # Replace the URL if deploying your app remotely
#     # (e.g., on Anyscale or KubeRay).
#     base_url="http://127.0.0.1:8000/v1",
#     api_key="None",
# )
# chat_completion = client.chat.completions.create(
#     # model="Qwen/Qwen2.5-3B-Instruct",
#     model="SGLANG/Assistant-model",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "What are 2+2?",
#         },
#     ],
#     temperature=0.01,
#     stream=True,
#     max_tokens=100,
# )

# for chat in chat_completion:
#     if chat.choices[0].delta.content is not None:
#         print(chat.choices[0].delta.content, end="")

import openai

client = openai.Client(base_url="http://127.0.0.1:8000/v1", api_key="None")

response = client.chat.completions.create(
    model="SGLANG/Assistant-model",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(f"Response: {response}")