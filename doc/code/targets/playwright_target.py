# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: pyrit-311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Playwright Target:
#
# This notebook demonstrates how to interact with the **Playwright Target** in PyRIT.
#
# The `PlaywrightTarget` class allows you to interact with web applications using
# [Playwright](https://playwright.dev/python/docs/intro).
# This is useful for testing interactions with web applications, such as chatbots or other web interfaces,
# especially for red teaming purposes.
#
# Before you begin, ensure you have the correct version of PyRIT installed and any necessary secrets configured as
# described [here](../../setup/populating_secrets.md).
#
# **Installation:**
#
# If you plan to use the Playwright integration (`PlaywrightTarget`), install PyRIT with the `playwright` extra:
#
# ```bash
# pip install -e '.[dev,playwright]'
# ```
#
# After installing PyRIT with Playwright, install the browser binaries:
#
# ```bash
# playwright install
# ```
#
# **Note:** The `-e` flag installs PyRIT in editable mode, which is useful if you're modifying the source code.
# If you are not modifying the source, you can omit the `-e` flag.


# %% [markdown]
# ## Example: Interacting with a Web Application using `PlaywrightTarget`
#
# In this example, we'll interact with a simple web application running locally at `http://127.0.0.1:5000`.
# We'll define an interaction function that navigates to the web application, inputs a prompt, and retrieves the
# response. We'll also use a scorer to evaluate the responses.

# %%
from pyrit.common import default_values
from playwright.async_api import Page, async_playwright
from pyrit.prompt_target import PlaywrightTarget, OllamaChatTarget
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.score import SelfAskLikertScorer, LikertScalePaths
from pyrit.models import PromptRequestPiece

# Load environment variables (optional)
default_values.load_environment_files()


# Define the interaction function
async def interact_with_my_app(page: Page, request_piece: PromptRequestPiece) -> str:
    # Define selectors
    input_selector = '#message-input'
    send_button_selector = '#send-button'
    bot_message_selector = '.bot-message'

    # Count existing messages
    initial_messages = await page.query_selector_all(bot_message_selector)
    initial_message_count = len(initial_messages)

    # Wait for the page to be ready
    await page.wait_for_selector(input_selector)

    # Send the prompt text
    prompt_text = request_piece.converted_value
    await page.fill(input_selector, prompt_text)
    await page.click(send_button_selector)

    # Wait for new messages (bot messages)
    await page.wait_for_function(
        f"document.querySelectorAll('{bot_message_selector}').length > {initial_message_count}"
    )

    # Extract the bot's response text
    bot_message_element = await page.query_selector(f'{bot_message_selector}:last-child')
    bot_response = await bot_message_element.text_content()
    return bot_response.strip()


# Define the scoring target (replace with your model and endpoint)
scoring_target = OllamaChatTarget(
    model_name="dolphin-llama3:70b",
    endpoint="http://127.0.0.1:11434/api/chat"
)

# Set up the scorer with a Likert scale for hate speech detection
bias_scorer = SelfAskLikertScorer(
    likert_scale_path=LikertScalePaths.HATE_SPEECH_SCALE.value,
    chat_target=scoring_target
)


# %% [markdown]
# ### Using `PlaywrightTarget` with the Interaction Function and Scorer
#
# Now, we can use the `PlaywrightTarget` by passing the interaction function we defined.
# We'll use the `PromptSendingOrchestrator` to send prompts to the target, collect responses, and score them.

# %%
# Using PlaywrightTarget with the interaction function and scorer
async def main(page: Page):
    target = PlaywrightTarget(
        interaction_func=interact_with_my_app,
        page=page
    )

    with PromptSendingOrchestrator(
            prompt_target=target,
            scorers=[bias_scorer],
    ) as orchestrator:
        all_prompts = [
            "Tell me a joke about computer programming.",
        ]

        await orchestrator.send_prompts_async(prompt_list=all_prompts)

        memory = orchestrator.get_memory()
        score_memory = orchestrator.get_score_memory()

        for entry in memory:
            for score_entry in score_memory:
                if entry.id == score_entry.prompt_request_response_id:
                    print(
                        f"Scored text: {entry.converted_value}\n"
                        f"Score category: {score_entry.score_category}\n"
                        f"Score value: {score_entry.get_value()}\n\n"
                    )

                # Run the main function


async with async_playwright() as playwright:
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page: Page = await context.new_page()
    await page.goto('http://127.0.0.1:5000')
    await main(page)