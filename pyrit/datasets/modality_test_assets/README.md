# Modality Test Assets

Benign, minimal test files used by `pyrit.prompt_target.modality_verification` to
verify which modalities a target actually supports at runtime.

- **test_image.png** — 1×1 white pixel PNG
- **test_audio.wav** — TTS-generated speech: "raccoons are extraordinary creatures"
- **test_video.mp4** — 1-frame, 16×16 solid color video

These are intentionally simple and non-controversial so they won't be blocked by
content filters during modality verification.
