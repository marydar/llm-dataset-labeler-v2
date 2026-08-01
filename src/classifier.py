import json
import re

from src.llm import ask_llm


def create_prompt(
        texts,
        labels
):

    label_text = "\n".join(
        [
            f"{k}: {v}"
            for k,v in labels.items()
        ]
    )


    examples = "\n".join(
        [
            f"{i+1}. {t}"
            for i,t in enumerate(texts)
        ]
    )


    prompt = f"""

You are a text classification model.

Classify each user prompt into exactly ONE category.

Available categories:

{label_text}


Rules:

- Choose the closest category.
- Do not invent labels.
- Return JSON only.
- Keep order.


Texts:

{examples}


Output format:

[
 {{
 "text_id":1,
 "label":"category"
 }}
]

"""


    return prompt




def classify_batch(
        texts,
        labels
):

    prompt = create_prompt(
        texts,
        labels
    )


    result = ask_llm(prompt)


    return parse_response(
        result,
        texts
    )




def parse_response(
        response,
        texts
):

    try:

        data = json.loads(response)


        output=[]


        for item in data:

            idx=item["text_id"]-1

            output.append(
                {
                    "text":texts[idx],
                    "label":item["label"]
                }
            )


        return output


    except Exception:

        print(
            "Failed JSON:"
        )

        print(response)

        return []