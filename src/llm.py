from openai import OpenAI
from config import (
    NVIDIA_API_KEY,
    BASE_URL,
    MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    REASONING
)


client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=BASE_URL
)



def ask_llm(prompt):

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        top_p=0.95,
        max_tokens=4096,
        seed=42,
        stream=False,
        extra_body={
        "chat_template_kwargs": {
            "enable_thinking": REASONING
        }
    }
    )
    # print(response)
    # print(response.choices[0].message.content)

    return response.choices[0].message.content.strip()


# from openai import OpenAI

# client = OpenAI(
#     api_key=NVIDIA_API_KEY,
#     base_url=BASE_URL
# )


# def ask_llm(prompt):

#     response = client.chat.completions.create(
#         model=MODEL,

#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],

#         temperature=1,
#         top_p=0.95,
#         max_tokens=16384,
#         seed=42,

#         stream=False,

#         reasoning_effort="max",
#     )

#     return response.choices[0].message.content.strip()


