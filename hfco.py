import sys
import json
import logging
from urllib import request
from bs4 import BeautifulSoup
from typing import Tuple
from score import PROMPTS

log = logging.getLogger(__name__)


def post(addr, data):
    req = request.Request(addr, data=json.dumps(data).encode())
    response = request.urlopen(req)
    data_bytes = response.read()
    return json.loads(data_bytes.decode())


def call_classifier(api_address, instruction_prompt: Tuple[str, str], abstract: str):
    log.debug("processing %r", abstract)
    full_prompt = instruction_prompt[1].strip("\n").replace("{{ TEXT }}", abstract)
    log.debug("full prompt: %r", full_prompt)
    info_data = post(api_address + "/v1/model", {"action": "info"})
    log.debug("model: %s", info_data["result"]["model_name"])
    log.debug("system prompt: %s", info_data["result"]["shared.settings"]["context"])
    chat_response = post(api_address + "/v1/chat", {"user_input": full_prompt})
    response = chat_response["results"][0]["history"]["visible"][-1][-1].lower()
    log.info("got %s", response)
    return response.startswith("yes")


def main():
    text_generation_webui_api_address = sys.argv[1]
    api_address = f"{text_generation_webui_api_address}/api"

    resp = request.urlopen("https://huggingface.co/papers")
    assert resp.status == 200
    papers_homepage = resp.read()
    root = BeautifulSoup(papers_homepage, features="lxml")
    papers_links = set()
    for a_element in root.find_all("a"):
        href = a_element.get("href")
        if not href:
            continue
        if href.startswith("/papers/"):
            papers_links.add(href)

    print(papers_links)

    for paper_link in papers_links:
        print(paper_link)
        resp = request.urlopen("https://huggingface.co" + paper_link)
        assert resp.status == 200
        paper_page = resp.read()
        page = BeautifulSoup(paper_page, features="lxml")

        element = page.find(
            lambda tag: tag.name == "h2" and "Abstract" in tag.get_text()
        )
        abstract = element.parent.find("p").get_text()
        print(call_classifier(api_address, PROMPTS[0], abstract))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
