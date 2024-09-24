# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit2
#     language: python
#     name: python3
# ---

# # This notebook shows how to interact with the HTTP Target: 

# As a simple example google search is used to show the interaction (this won't result in a successful search because of the anti-bot rules but shows how to use it in a simple case)

# +
import os
import urllib.parse

from pyrit.models import PromptTemplate
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import HTTP_Target
from pyrit.models import PromptRequestPiece


## Add the prompt you want to send to the URL
prompt = "apple"
url = "https://www.google.com/search?q={PROMPT}"
# Add the prompt to the body of the request

with HTTP_Target(http_request={}, url=url, body={}, url_encoding="url", body_encoding="+", method="GET") as target_llm:
    request = PromptRequestPiece(
        role="user",
        original_value=prompt,
    ).to_prompt_request_response()

    resp = await target_llm.send_prompt_async(prompt_request=request)  # type: ignore
    print(resp)
    print
    

# +
import os
import urllib.parse

from pyrit.models import PromptTemplate
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import HTTP_Target
from pyrit.models import PromptRequestPiece
# -

# Bing Image Creator which does not have an API is harder to use
#
# The HTTP request to make needs to be captured and put here in the "http_req" variable (the values you need to get from DevTools or Burp include the Cookie)

# +
http_req = f"""
Host: www.bing.com
Cookie: MUID=3C3E74DA71226CE919B26020700F6D77; SRCHD=AF=NOFORM; SRCHUID=V=2&GUID=D73F8C19C51E4C20BCF7148461855518&dmnchg=1; MUIDB=3C3E74DA71226CE919B26020700F6D77; MSPTC=PQNEcHGh51Or3Xo4_qb5b5ThY9ZeJ8L9dGjyJLYFY0I; BFBUSR=BAWAS=1&BAWFS=1; EDGSRCHHPGUSR=CIBV=1.1678.1; MMCASM=ID=E3099FEE14864F43BDFA1705C6EE8ECA; MSFPC=GUID=2a36dbc594524e50bed80f6b9e511f61&HASH=2a36&LV=202409&V=4&LU=1726335645382; BCBSWITCH=BCB=0; _Rwho=u=d&ts=2024-09-18; SRCHS=PC=U531; OVRTH=; CortanaAppUID=CDA5E56CD69F408118D4C9083296EF19; _UR=QS=0&TQS=0&Pn=0; _HPVN=CS=eyJQbiI6eyJDbiI6MSwiU3QiOjAsIlFzIjowLCJQcm9kIjoiUCJ9LCJTYyI6eyJDbiI6MSwiU3QiOjAsIlFzIjowLCJQcm9kIjoiSCJ9LCJReiI6eyJDbiI6MSwiU3QiOjAsIlFzIjowLCJQcm9kIjoiVCJ9LCJBcCI6dHJ1ZSwiTXV0ZSI6dHJ1ZSwiTGFkIjoiMjAyNC0wOS0xOVQwMDowMDowMFoiLCJJb3RkIjowLCJHd2IiOjAsIlRucyI6MCwiRGZ0IjpudWxsLCJNdnMiOjAsIkZsdCI6MCwiSW1wIjozLCJUb2JuIjowfQ==; ipv6=hit=1726707898799&t=6; MUIDB=3C3E74DA71226CE919B26020700F6D77; _C_ETH=1; BFBFB=F%3D1%26L%3D1%26S%3D1%26SS%3D1%26E%3D1%26O%3D2%26EDS%3D2%26SDS%3D2%26CDS%3D2%26CF%3D2; MicrosoftApplicationsTelemetryDeviceId=a2857dea-a41c-4523-b418-08e7c5d101db; OIDR=gxBL9f3t4dCfY4N7isxB0dRPaYoLNqQ7w6k9BXCB0dUQiX3sdzLm7CorGqpl05foBC37u0DMivKtvU5RBO7fHOpJ2wzjbLIjl_8-76KCU4ERWN3FzxJG1qSEzKGj5cBY0NEPjvIAUTCEC-Bq1KHSpL5CMzQYuaTB7BI7NgkTlnoSH8ghGaEClcKxFi4ngXJuMldSeCCPkLQ8jN8cCJ2p4b76aXb5pRGBdn3M1byvVdWAen27CVx9cOYqSaZfCcX98YVlIejTyTyL6K78TkPWV7gxj5vOYux56B3yZSWmTdgbOWWLYIMwbSPPkZwakUTlOsSsdJOHb0Qn7nDi-YpFodgTl9YOK-g6ETDmnOktglCDXGZgXxOPeEtwZi9Kvh2126qdT16cQ1_suv-euDENtOJz9PsEQnROqKO7SefqL6s8v9r7Hce5I6rUWmEPq3-I_9o5tAzONtMBKGz5UTW013URL96FlF7Ybo5KXItErSN3eAF47GYVKoVwHKIW4ATpZhu4qzoYedtCFgp7_WxMXpVwqKvlybUfttD0Wobui7wryZWXOqSF6OnVoBc1W8vXpXp6BpziM2JtxTG_rZLEjYridfIUBMSyZq6dxKUFKFSHCirkyzPKdqiUvuXssCOtvSFKGUh-sUwiI1ODthOH1nqsWBWDQvBLfReRsrwbqKRpCovgKMe-G-QixWLGsVLg3ZvWvtEdsb_rRXKtpXy3fVj82T7rNNwa9rAvZWOmNfz6OSqMttVCtRhGmA2fWTDQi8NfxU8nKg6P99QIQxpHzLVc9kjSCkUPtWhzYHcY5A62JFqy_2axT5LGbsP5vpkw5miZCcrGtpFeOGtI6KBxcbo7WEKj4Wl8-EzKqtASD4yNJT3KnpNMCzr5gGni-LbBsKwj8OfDjEPNg3cWbIpiCEWhM6tdOcpeliWVjkjW_0IyCQY73Un76S-jR-DvMasP06uIjVZ4lvasLuZrLchuWL5hVvK0rdskKyKpBbZN8uacu1jrkqz3uf8HafRdtVU_ePIrGB42xFBQONtIUaZl1cNVErsBKpMMSezIi0uYIS-SRcVuXMp3xmq2Zxl9MU1P5iadMdzR8CigF2AjhQZHxQTvp2_hyN5a7OVd8L_VcYLzDuoiTAtbxNwamocf8E2iQW9IUWne8Bu6PxPT7blkJ0snL6LsCr3v0w1OZiWjXEzdYXmoRaH0H7JCuKw_goFx5dSBsS6BICuXiLt3r1fSVux2qJA7WuqAsIxoYbuWkSM6n8l2zWNC557jboc9ee9n5DYvb9txb0QDZ5Cjy7_ygTHP8jGcOML5TohGCw2QINjOGYzn33hEWHNIQjL90UgcAlBxm_9o06tFySn8xw3S73EO; CSRFCookie=36dfa334-6b33-431e-a0c1-95162c6f0762; SRCHUSR=DOB=20240914&T=1726499929000&POEX=W; ANON=A=2CC18B9928FACFB213E2035EFFFFFFFF&E=1e47&W=1; NAP=V=1.9&E=1ded&C=6MT6XeBSJSnrfUeMQTHlO8WGaRHVWtK9-LW0Ur4RsmEBkQ5sFHPRxA&W=1; PPLState=1; KievRPSSecAuth=FACKBBRaTOJILtFsMkpLVWSG6AN6C/svRwNmAAAEgAAACN0mXNa6rwGPSATIZYM8xk2GX8WLjaoxAytFfU2NRsS8WwHY9Sbfo2zU3P9K07ebOCO8DZNxbTUarnsna0EDhBZEBMYE74Vq55YJzlISTF8Q3PPq5J2Z+s+90JXIvaidFE6HYMRlmFdyFO3ITR2beeB3BC2lKQwntZTr0lrdmmPMmWXvjw64jVTXZu0TE4ASRw1JmUI19HCNvvG1+CAhTg55dB6aOmPWe/FXzTxZca1ZuaKHsWfzL25uZuZDdkoFMaa/f3QVGxI8eWnE0V8G+9UJpRYXr4yuhIFQdM8td0t9QPkDLkUH7cEy7UUCsNfE9B4lDIhF8n2HMbQXoOONUHCUuO78P2LNbf4ztskVHJJiHHBiOqqycMJBhKoDNAfMgNfw6b/Zb0xpNs/Gl/TuqBg6EWpv3uChQKZ8lfmVXSqpHp+OHp+bh5JHwUz9Rj0Xe6PESgf+VaKTEhNxflxsaRwXzWVIeturfXVxSXGt4O0SKge/2Mx/EaHRsdWkWJrFjIL1fMn2VcX+Vzrr2iVIZ1kCMxyxSxB3upy7nM7K4LrJXxOfkLUjwJA6oBmSgJZkyKJZliekt0lTYZcx093TSf+Y3Rc8tdmQR/E9M/hiAbb4cuhw5I8m/qWDcujRQmyHHraoO9+KxQMl88YTWjaQD86yRfZlfIFLKIkLzyZZ9MdYtcNS6ZzZdpZMXmYMs1Q4Cgea6IGSeIZpn+R3nV5jlzcg0YBaHk23ByAe72Tjr3lKrnCSOpeTYENBLuewem0c0I1OSaOyiQlFMTDWvhYyJxW4Dnyo/jUfda7Togk6fcgEQmCvh5E64WkhlwKBB6YQiqgIte86bqGTM8NOUa7fNzE/fs/yfvckpoPk/pS54npQcz/J12+AUv3kn68s7Rxj4yY/hH+ZLGOq/AAjC+Qg+oExImWDb/l3DTU8tJFiO+1p2qM/G+XTPNfPjjiBY7JfyNC3b2YqIYEX0/ZVOruQHf6MDiEyWm4SqVcqo/BarMlZQ/Re9OUvH5rim0XLWlBtp/gtYvL90k3OCpj1de4n5XHZ9Ona3LLHmgS3tndZ34KzU79gM2XlanJsZDpGPyyPNlXx2spHqhiDRQvp8LVRAx71XWb1rn7faL5aIZoQY/xpoFRc5UkEIT08VrJkWcCqAZaNC785u0hUj2Tn91L+hfQtlMXw1iHbnWtQm38saO7YToSGfvzYRCgvIDOe9JpfhCopBWoXeMO1P4GFq1rwOBBEPUuLviGl0FGagnta2EoMSr4QIlJI3Nk4rWXLLB1g5U3S2n17YWPHox95erA7uTMPUtu8+O42dueazZtpA3M5L0K7WUmWL6VHXb/eh7WgTktNOPqiHtyDeNEo5NKhjXDMDBsni6r0W6hmLryA+9JPq7DVFO+peoI4g2EalnmspN9ra1EQxpVOVUp0u4tOwwrhvxZAQDMxDQvOAQCV93z/6LjXo8B2F2Z7eFj2w+vL04PfFAAvMHZzgKHE6n70OaeOlfvS1zAHxA==; _U=1B_YYoB5RJRSTyI_4m2m3KlLZzasl2WhWpPTNRDTtVxpVdA-f6_Xg6CiV2sI-ic07OPFpWJ26zKnw0vveWSMvKy6RsjPK-DD7IpGxzN78t7svTt9ZyrgUSNdThbv6_5FVtyBpw2dEX48-6GhX6chxWRCqlQkZi0pQQ_87O-WwbNJ69JhWbfMFuzNBfZIXBxbI42AhV2rt01iCqV0aZ2OLGpf8XCZMHgFjC2mY0hJHA-k; WLS=C=0f3786b79f759f46&N=airt; WLID=XJI9SjVKm90FCcWcw2eolskaT5pUOO6bv4bYfTo2UQV/J/FBJNVq2ir1him+z1mFTsUbk63YZ9n9pf9YmpuHJQ7eX2ZXbKA+GNoB2e11hWo=; _SS=SID=25B2615A1ED46DE026E175A41F976C9E&R=40&RB=40&GB=0&RG=0&RP=40&PC=U531; BFB=gxBSWjyBSA1-g_T0EGigZDh6I7YAuiClJbnJa7wnnWN4q7x-ccNNN5y5qBBaijyKK6gcf3Y-rHe5Gps7Bm66LGwvFeufPB4D8DuJF3OlPTzYY9kth9SeMUrQhB-7pTq1tOvLhYAe3-39Y_8t-cCGDwAHvWWwyNBnwQgOKq0xDFcqhbleKmeTGSPawXPg0VmvnHccuDUOZ4qwDfqPwvyj_7p63WACFGP2qjwV-uMf5tYctG7-5gQQ-3l0KKNHT_3lfs8; _EDGE_S=SID=0D5BA0B2088064F03D81B5B109AD6517&mkt=en-us; USRLOC=HS=1&ELOC=LAT=47.65090560913086|LON=-122.13887023925781|N=Redmond%2C%20Washington|ELT=2|&CLOC=LAT=47.65090751452178|LON=-122.13887364216826|A=733.4464586120832|TS=240924163327|SRC=W&BID=MjQwOTI0MDkzMzI3Xzk1NGViMjc2MDAwMzk3OGU5YWE3NWZiODY3NmFhMjk1MTFiNGJlMzJhYTA3MWNkM2M4NWJmMjdlODE3Y2E2Mjc=; dsc=order=BingPages; SRCHHPGUSR=SRCHLANG=en&DM=0&IG=D1A73B4214914236BA1D9E795B61471B&PV=15.0.0&BRW=XW&BRH=T&CW=2552&CH=1280&SCW=2537&SCH=2952&DPR=1.0&UTC=-420&THEME=0&WEBTHEME=0&CIBV=1.1813.0&EXLTT=31&HV=1727197167&WTS=63862096729&PRVCW=2552&PRVCH=1280&AV=14&ADV=14&RB=0&MB=0; SnrOvr=F=msbqfntpanhm; GI_FRE_COOKIE=gi_prompt=5; _clck=9u8c2h%7C2%7Cfpg%7C0%7C1720; GC=U_tpl1omGVvQ4pMt5o6dS_8Y9VNHkTUtAV1qDZi_tFe8QjyyAjG7GseQML3abdvq0D3H2BXud3kc2ZUSYAxSMQ; _clsk=1ypx4b2%7C1727197497624%7C1%7C0%7Cw.clarity.ms%2Fcollect; _RwBf=r=1&ilt=4&ihpd=0&ispd=1&rc=40&rb=40&gb=0&rg=0&pc=40&mtu=0&rbb=0.0&g=0&cid=&clo=0&v=3&l=2024-09-24T07:00:00.0000000Z&lft=0001-01-01T00:00:00.0000000&aof=0&ard=0001-01-01T00:00:00.0000000&rwdbt=-62135539200&rwflt=-62135539200&rwaul2=0&o=0&p=MSAAUTOENROLL&c=MR000T&t=5584&s=2024-09-12T20:53:07.8976531+00:00&ts=2024-09-24T17:04:54.4383559+00:00&rwred=0&wls=2&wlb=0&wle=0&ccp=2&cpt=0&lka=0&lkt=0&aad=0&TH=&mta=0&e=bS6vlg7dmNtJRcHc7-5tB1yd39S0awMGvVYLsvKGy6IJhBGpY15h8YoL1dnRu3vNqwf-2qgeJNhe2x-psguLEA&A=2CC18B9928FACFB213E2035EFFFFFFFF; OID=AxC0LrHrxLgG1CEsoJTCPjbkt-Z11aqMDqOMEfHPFHIqh-Xz20SU2jn5LquvMEvnsxNukB6t-QyK7nW3DBG9-9bJtglvfhfwXuFK8raiN1h5av-9LbkhxwFBm8ftoHJV2Wg; OIDI=AxCIP5afCG8h0U0dDXvpPlPkpHjlbBYxQ4FFVciep3cVL1yiUspKvsDSX70uOXJ5OcvRUixBsE3OERaXZr7E4L8G
Origin: https://www.bing.com
Content-Type: application/x-www-form-urlencoded
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.6613.120 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Sec-Fetch-Site: same-origin
Sec-Fetch-Mode: navigate
Sec-Fetch-User: ?1
Sec-Fetch-Dest: document
Referer: https://www.bing.com/images/create?FORM=GENILP

"""

## Add the prompt you want to send to the URL
prompt = "apple"
url = "https://www.bing.com/images/create?q={PROMPT}&rt=4&FORM=GENCRE"

# Add the prompt to the body of the request

body = "q={PROMPT}&qs=ds"
response_var = None
with HTTP_Target(http_request=http_req, url=url, body=body, url_encoding="url", method="POST") as target_llm:
    # Questions: do i need to call converter on prompt before calling target? ie url encode rather than handling in target itself?
    request = PromptRequestPiece(
        role="user",
        original_value=prompt,
    ).to_prompt_request_response()

    resp = await target_llm.send_prompt_async(prompt_request=request)  # type: ignore
    response_var = resp

    

# +
from bs4 import BeautifulSoup
html_content = response_var.request_pieces[0].original_value
parsed_hmtl_soup = BeautifulSoup(html_content, 'html.parser')

print(parsed_hmtl_soup.prettify())

#TODO: parse & turn this into a parsing function: as an example this is the image 
# <div data-c="/images/create/async/results/1-66f2fad6d7834081a343ac05ae3c1784?q=apple&amp;IG=52AE84FF96F948909718523E5DB8AF89&amp;IID=images.as" data-mc="/images/create/async/mycreation?requestId=1-66f2fad6d7834081a343ac05ae3c1784" data-nfurl="" id="gir">


# +
# Just same thing using orchestrator
http_prompt_target = HTTP_Target(http_request=http_resp, url=url, body=body)

with PromptSendingOrchestrator(prompt_target=http_prompt_target) as orchestrator:
    response = await orchestrator.send_prompts_async(prompt_list=[prompt])  # type: ignore
    print(response[0])
