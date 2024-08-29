# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit2
#     language: python
#     name: python3
# ---

# %% [markdown]
# This shows an example of using Bing Image Creator which does not have an API
#
# You need to store your cookie in .env
# TODO: add docs here

# %%
import os
import urllib.parse

from pyrit.models import PromptTemplate
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import HTTP_Target

from pyrit.common import default_values


default_values.load_default_env()

from pyrit.common import default_values
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestPiece

default_values.load_default_env()

http_resp = f"""
Host: www.bing.com
Cookie: {os.getenv("BING_COOKIE")}
Content-Length: 21
Cache-Control: max-age=0
Sec-Ch-Ua: "Chromium";v="127", "Not)A;Brand";v="99"
Sec-Ch-Ua-Mobile: ?0
Sec-Ch-Ua-Full-Version: ""
Sec-Ch-Ua-Arch: ""
Sec-Ch-Ua-Platform: "Windows"
Sec-Ch-Ua-Platform-Version: ""
Sec-Ch-Ua-Model: ""
Sec-Ch-Ua-Bitness: ""
Sec-Ch-Ua-Full-Version-List: 
Accept-Language: en-US
Upgrade-Insecure-Requests: 1
Origin: https://www.bing.com
Content-Type: application/x-www-form-urlencoded
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Sec-Fetch-Site: same-origin
Sec-Fetch-Mode: navigate
Sec-Fetch-User: ?1
Sec-Fetch-Dest: document
Referer: https://www.bing.com/images/create?FORM=GENILP
Accept-Encoding: gzip, deflate, br
Priority: u=0, i
"""

## Add the prompt you want to send to the URL
prompt = "roakey racoon"
prompt_url_safe = urllib.parse.quote(prompt)
url = f"https://www.bing.com/images/create?q={prompt_url_safe}&rt=3&FORM=GENCRE"

# Add the prompt to the body of the request
body_query = prompt.replace(" ", "+")
body = f"q={body_query}&qs=ds"
with HTTP_Target(http_request=http_resp, url=url, body=body) as target_llm:
    request = PromptRequestPiece(
        role="user",
        original_value=prompt,
    ).to_prompt_request_response()

    resp = await target_llm.send_prompt_async(prompt_request=request)  # type: ignore
    print(resp)

# %%
http_prompt_target = HTTP_Target(http_request=http_resp, url=url, body=body)

with PromptSendingOrchestrator(prompt_target=http_prompt_target) as orchestrator:
    response = await orchestrator.send_prompts_async(prompt_list=[prompt])  # type: ignore
    print(response[0])
