import json
import os
import random
import time


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf8"
    ) as f:

        return json.load(f)



def save_json(
        data,
        path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )



def random_delay(
        low,
        high
):

    time.sleep(
        random.uniform(
            low,
            high
        )
    )