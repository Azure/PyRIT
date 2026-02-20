#!/usr/bin/env python3

# Test the core modality architecture without initializing targets

print("=== Testing Core Modality Architecture ===")

# Test the base architecture
from pyrit.models import PromptDataType

# Test frozenset creation and matching
text_only = frozenset(["text"])
text_image = frozenset(["text", "image_path"])
text_audio = frozenset(["text", "audio_path"])

print(f"Text-only frozenset: {text_only}")
print(f"Text+Image frozenset: {text_image}")
print(f"Text+Audio frozenset: {text_audio}")

# Test set of frozensets
modality_combinations = {text_only, text_image}
print(f"Modality combinations set: {modality_combinations}")

# Test exact matching
test_modalities = frozenset({"text", "image_path"})
print(f"Test modalities: {test_modalities}")
print(f"Exact match found: {test_modalities in modality_combinations}")

# Test order independence
test_modalities_different_order = frozenset({"image_path", "text"})
print(f"Different order: {test_modalities_different_order}")
print(f"Still matches: {test_modalities_different_order in modality_combinations}")

# Test PromptDataType literals
text_type: PromptDataType = "text"
image_type: PromptDataType = "image_path"
audio_type: PromptDataType = "audio_path"

print(f"\nPromptDataType literals:")
print(f"Text: {text_type}")
print(f"Image: {image_type}")
print(f"Audio: {audio_type}")

# Test with literal types
literal_frozenset = frozenset([text_type, image_type])
print(f"Literal frozenset: {literal_frozenset}")
print(f"Matches string frozenset: {literal_frozenset == text_image}")

print("\n=== Architecture Test Complete ===")