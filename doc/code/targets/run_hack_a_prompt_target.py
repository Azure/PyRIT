# %% [markdown]
# # HackAPrompt Playground Demo
#
# This notebook demonstrates how to send prompts and receive judged responses from the HackAPrompt competition platform using PyRIT. To authenticate, you need a valid session ID and authentication cookies from your HackAPrompt account.
#
# The session ID can be found in Chrome DevTools under the Network tab by inspecting any `/api/chat` request payload while you are logged in and interacting with the challenge, look for the `session_id` value in the JSON body. Your authentication cookies are visible under the Application tab in DevTools, inside Storage > Cookies for `hackaprompt.com`. Copy the names and values (such as `sb-...auth-token.0` and `.1`) directly from there.
#
# With these details, you can programmatically interact with the HackAPrompt API as your logged-in user, send prompts, and see how the judge panel scores your attack.


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
