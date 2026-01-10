# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Literal


# Voices supported by OpenAI Chat Completions API audio output.
# See: https://platform.openai.com/docs/guides/text-to-speech#voice-options
CompletionsAudioVoice = Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"]
CompletionsAudioFormat = Literal["wav", "mp3", "flac", "opus", "pcm16"]


@dataclass
class OpenAICompletionsAudioConfig:
    """
    Configuration for audio output from OpenAI Chat Completions API.

    When provided to OpenAIChatTarget, this enables audio output from models
    that support it (e.g., gpt-4o-audio-preview).

    Note: This is specific to the Chat Completions API. The Responses API does not
    support audio input or output. For real-time audio, use RealtimeTarget instead.

    Attributes:
        voice: The voice to use for audio output. Supported voices are:
            "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar".
        format: The audio format for the response. Supported formats are:
            "wav", "mp3", "flac", "opus", "pcm16". Defaults to "wav".
        prefer_transcript_for_history: If True, historical user messages that contain
            both audio and text will only send the text (transcript) to reduce bandwidth
            and token usage. The current (last) user message will still include audio.
            Defaults to True.
    """

    voice: CompletionsAudioVoice
    format: CompletionsAudioFormat = "wav"
    prefer_transcript_for_history: bool = True

    def to_extra_body_parameters(self) -> dict:
        """
        Convert the config to extra_body_parameters format for OpenAI API.

        Returns:
            dict: Parameters to include in the request body for audio output.
        """
        return {
            "modalities": ["text", "audio"],
            "audio": {"voice": self.voice, "format": self.format},
        }
