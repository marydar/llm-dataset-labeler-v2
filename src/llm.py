from openai import OpenAI
from config import (
    OPENROUTER_API_KEY,
    BASE_URL,
    MODEL,
    MAX_TOKENS,
    TEMPERATURE
)


client = OpenAI(
    api_key=OPENROUTER_API_KEY,
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
        extra_body={"reasoning": {"enabled": True}},

        # max_tokens=MAX_TOKENS 
    )
    # print(response)
    # print(response.choices[0].message.content)

    return response.choices[0].message.content


# from openai import OpenAI

# def ask_llm(prompt):

#     client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
#     )

#     # First API call with reasoning
#     response = client.chat.completions.create(
#     model="nvidia/nemotron-3.5-content-safety:free",
#     messages=[
#             {
#                 "role": "user",
#                 "content": "How many r's are in the word 'strawberry'?"
#             }
#             ],
#     extra_body={"reasoning": {"enabled": True}}
#     )

#     # Extract the assistant message with reasoning_details
#     response = response.choices[0].message
#     print("RESPONSE:",response)

