# %%NBQA-CELL-SEP52c935
from pyrit.chat_message_normalizer import ChatMessageNormalizerChatML
from pyrit.models import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a helpful AI assistant"),
    ChatMessage(role="user", content="Hello, how are you?"),
    ChatMessage(role="assistant", content="I'm doing well, thanks for asking."),
]

normalizer = ChatMessageNormalizerChatML()
chatml_messages = normalizer.normalize(messages)
# chatml_messages is a string in chatml format

print(chatml_messages)

# %%NBQA-CELL-SEP52c935
chat_messages = normalizer.from_chatml(
    """\
    <|im_start|>system
    You are a helpful AI assistant<|im_end|>
    <|im_start|>user
    Hello, how are you?<|im_end|>
    <|im_start|>assistant
    I'm doing well, thanks for asking.<|im_end|>"""
)

print(chat_messages)

# %%NBQA-CELL-SEP52c935
import os

from transformers import AutoTokenizer

from pyrit.chat_message_normalizer import ChatMessageNormalizerTokenizerTemplate

messages = [
    ChatMessage(role="user", content="Hello, how are you?"),
    ChatMessage(role="assistant", content="I'm doing well, thanks for asking."),
    ChatMessage(role="user", content="What is your favorite food?"),
]

# Load the tokenizer. If you are not logged in via CLI (huggingface-cli login), you can pass in your access token here
# via the HUGGINGFACE_TOKEN environment variable to access the gated model.
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
                                          token = os.environ.get("HUGGINGFACE_TOKEN"))

# create the normalizer and pass in the tokenizer
tokenizer_normalizer = ChatMessageNormalizerTokenizerTemplate(tokenizer)

tokenizer_template_messages = tokenizer_normalizer.normalize(messages)
print(tokenizer_template_messages)
