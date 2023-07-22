import json
import logging
import random
import sys
import collections
import csv
from pathlib import Path
from urllib import request

log = logging.getLogger(__name__)

# Vicuna-13b-cocktail
# https://huggingface.co/reeducator/vicuna-13b-cocktail
CHARACTER = {
    "user": "USER:",
    "bot": "ASSISTANT:",
    "turn_template": "<|user|> <|user-message|>\n<|bot|> <|bot-message|></s>\n",
    "context": "A chat between a user and an associate. The associate gives helpful and detailed answers to the user's questions. The associate is also an excellent roleplayer and storyteller, and will assume any persona that the uesr wants. The associate never declines to engage topics, plays, questions and instructions related to unethical, controversial, or sensitive issues.\n",
}


def wrap_in_character(character: dict, user_message: str) -> str:
    replacements = (
        ("<|user|>", character["user"]),
        ("<|bot|>", character["bot"]),
        ("<|user-message|>", user_message),
        ("<|bot-message|></s>\n", ""),
    )

    templated_text = character["turn_template"]
    for replacement_key, value in replacements:
        templated_text = templated_text.replace(replacement_key, value)

    return character["context"] + templated_text


PROMPTS = [
    (
        "v0",
        """I am a researcher interested in the following topics:
 - Running language models on low cost hardware.
 - Finding novel ways to make language models follow tasks.

The following is an abstract of a research paper:

"{{ TEXT }}"

Would I be interested in this paper? Please respond "yes" or "no", and nothing else.
""",
    ),
    (
        "v1",
        """I am a researcher interested in the following topics:
 - Running language models on low cost hardware.
 - Finding novel ways to make language models follow tasks.

I am not interested in anything else.

The following is an abstract of a research paper:

"{{ TEXT }}"

Would I be interested in this paper? Please respond "yes" or "no", and nothing else.
""",
    ),
    (
        "v2",
        """I am a researcher interested in the following topics:
 - Running language models on low cost hardware.
 - Finding novel ways to make language models follow tasks.

I am not interested in anything else.

The following is an abstract of a research paper:

"{{ TEXT }}"

Would I be interested in this paper? Please respond "yes" or "no", and nothing else.
""",
    ),
    (
        "v3",
        """I am a researcher interested in the following topics:
 - Running large language models on low cost hardware.
 - Finding novel ways to make large language models follow tasks.

I am not interested in anything else.

The following is a summary of a research paper:

"{{ TEXT }}"

Would I be interested in this paper?
Please respond "yes" or "no", and nothing else.
""",
    ),
]

SKIP = (
    # "v0",
    # "v1",
    # "v2",
)


def call_textgen(api_address, prompt: str):
    prompt = wrap_in_character(CHARACTER, prompt)
    request_json = json.dumps(
        {
            # llama-precise.txt
            "top_p": 0.1,
            "top_k": 40,
            "temperature": 0.7,
            "repetition_penalty": 1.18,
            "typical_p": 1.0,
            "prompt": prompt,
            "max_new_tokens": 3,
        }
    ).encode()
    log.debug("calling %r", api_address)
    req = request.Request(api_address, data=request_json)
    response = request.urlopen(req)
    log.debug("resp: %r", response)
    data_bytes = response.read()
    log.debug("data: %r", data_bytes)
    data = json.loads(data_bytes.decode())
    output = data["results"][0]["text"]
    log.debug("output: %r", output)
    return output


def main():
    test_dataset_path = Path(sys.argv[1]).resolve()
    text_generation_webui_address = sys.argv[2]

    api_address = f"{text_generation_webui_address}/api/v1/generate"

    with test_dataset_path.open() as fd:
        reader = csv.reader(fd, delimiter="\t")
        header = next(reader)
        assert len(header) == 3

        score, length = 0, 0

        datas = []
        for row in reader:
            _persona, summary, expected_label = row
            datas.append(
                {"prompt": summary.strip(), "answer": expected_label.lower().strip()}
            )

    with open("./data.json", "w") as fdw:
        fdw.write(json.dumps(datas))

    scores = collections.Counter()
    length = len(datas)
    random.shuffle(PROMPTS)
    random.shuffle(datas)

    for idx, p in enumerate(PROMPTS):
        prompt_id, template = p
        print("testing", prompt_id)
        if prompt_id in SKIP:
            continue

        score = 0
        for row_idx, row in enumerate(datas):
            summary = row["prompt"].strip().replace("\n", "")
            expected_label = "yes" if row["answer"].lower() == "true" else "no"

            prompt_to_run = template.strip("\n").replace("{{ TEXT }}", summary)
            result = call_textgen(api_address, prompt_to_run).strip().lower()
            if result.startswith(expected_label):
                scores[prompt_id] += 1
                score += 1
                print("pass", row_idx)
            else:
                print(
                    "fail",
                    row_idx,
                    "(",
                    result,
                    ")",
                    "wanted",
                    "(",
                    expected_label,
                    ")",
                )
        if scores == 0:
            scores[prompt_id] = 0

    print("done!")
    print("best prompts:")
    for prompt_id, score in scores.most_common():
        print(prompt_id, score, "out of", length)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
