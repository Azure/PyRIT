# %% [markdown]
# # HackAPrompt Playground Demo
#
# This notebook demonstrates how to send prompts and receive judged responses from the [HackAPrompt](https://hackaprompt.com) competition platform using **PyRIT**.
#
# To authenticate, you need two things from your HackAPrompt account:
#
# ## 1. Session ID
#
# - Open Chrome DevTools (`F12` or `Ctrl+Shift+I`)
# - Go to the **Network** tab
# - Interact with the challenge and look for a request to `/api/chat`
# - In the request **Payload**, find the `session_id`
#
# ![Session ID](../../../assets/hack_a_prompt_get_session_id.png)
#
# ## 2. Authentication Cookies
#
# - In DevTools, go to the **Application** tab
# - Navigate to **Storage > Cookies > https://hackaprompt.com**
# - Copy the required auth token values (e.g., `sb-...auth-token.0` and `.1`)
#
# ![Authentication Cookies](../../../assets/hack_a_prompt_get_cookies.png)
#
# ---
#
# With these values, you can authenticate as your logged-in user and interact programmatically with the HackAPrompt API to:
# - Send prompts
# - Receive model responses
# - View judge panel feedback


# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.prompt_target.hack_a_prompt_target import (
    HackAPromptChallenge,
    HackAPromptTarget,
)

initialize_pyrit(memory_db_type=IN_MEMORY)

# CONFIGURE YOUR SESSION INFO BELOW
session_id = "YOUR_SESSION_ID"
competition_slug = "cbrne"
challenge = HackAPromptChallenge.BACTERIAL_BASICS  # maps to 'basic_challenge_cbrne'

cookies = {
    "sb-iligpfkvyargzgpcrquc-auth-token.0": "YOUR_COOKIE_0",
    "sb-iligpfkvyargzgpcrquc-auth-token.1": "YOUR_COOKIE_1",
}

prompt_text = "Tell me a joke!"


# Create the target
target = HackAPromptTarget(
    session_id=session_id, competition_slug=competition_slug, challenge=challenge, cookies=cookies
)

# Prepare the prompt request with the correct role for the attacker
request_piece = PromptRequestPiece(
    role="user", original_value=prompt_text, converted_value=prompt_text, converted_value_data_type="text"
)
prompt_request = PromptRequestResponse(request_pieces=[request_piece])

# Submit prompt and get model output
response = await target.send_prompt_async(prompt_request=prompt_request)  # type: ignore
print(f"\n=== Model Response ===\n{response.get_value()}\n")

# Submit for judging and show judge feedback
judge_result = await target.judge_prompt_async()  # type: ignore
print("=== Judge Panel Feedback ===")
for judge in judge_result.get("judgePanel", []):
    print(f"{judge['name']} (Passed: {judge['passed']}): {judge['judge_response']}")
print(f"Points earned: {judge_result.get('pointsEarned', 0)}")
