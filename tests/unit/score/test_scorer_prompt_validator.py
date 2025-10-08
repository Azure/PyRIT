# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.score import ScorerPromptValidator


class TestScorerPromptValidatorDataTypes:
    """Test data type filtering functionality."""

    def test_default_supports_all_data_types(self):
        """Test that validator with no supported_data_types supports all types."""
        validator = ScorerPromptValidator()

        text_piece = PromptRequestPiece(role="assistant", original_value="text", converted_value_data_type="text")
        image_piece = PromptRequestPiece(
            role="assistant", original_value="image.png", converted_value_data_type="image_path"
        )
        audio_piece = PromptRequestPiece(
            role="assistant", original_value="audio.wav", converted_value_data_type="audio_path"
        )

        assert validator.is_request_piece_supported(text_piece) is True
        assert validator.is_request_piece_supported(image_piece) is True
        assert validator.is_request_piece_supported(audio_piece) is True

    def test_filters_to_text_only(self):
        """Test that validator correctly filters to only text pieces."""
        validator = ScorerPromptValidator(supported_data_types=["text"])

        text_piece = PromptRequestPiece(role="assistant", original_value="text", converted_value_data_type="text")
        image_piece = PromptRequestPiece(
            role="assistant", original_value="image.png", converted_value_data_type="image_path"
        )
        audio_piece = PromptRequestPiece(
            role="assistant", original_value="audio.wav", converted_value_data_type="audio_path"
        )

        assert validator.is_request_piece_supported(text_piece) is True
        assert validator.is_request_piece_supported(image_piece) is False
        assert validator.is_request_piece_supported(audio_piece) is False

    def test_filters_to_multiple_types(self):
        """Test that validator correctly filters to multiple specified types."""
        validator = ScorerPromptValidator(supported_data_types=["text", "image_path"])

        text_piece = PromptRequestPiece(role="assistant", original_value="text", converted_value_data_type="text")
        image_piece = PromptRequestPiece(
            role="assistant", original_value="image.png", converted_value_data_type="image_path"
        )
        audio_piece = PromptRequestPiece(
            role="assistant", original_value="audio.wav", converted_value_data_type="audio_path"
        )

        assert validator.is_request_piece_supported(text_piece) is True
        assert validator.is_request_piece_supported(image_piece) is True
        assert validator.is_request_piece_supported(audio_piece) is False

    def test_filters_to_image_only(self):
        """Test that validator correctly filters to only image pieces."""
        validator = ScorerPromptValidator(supported_data_types=["image_path"])

        text_piece = PromptRequestPiece(role="assistant", original_value="text", converted_value_data_type="text")
        image_piece = PromptRequestPiece(
            role="assistant", original_value="image.png", converted_value_data_type="image_path"
        )

        assert validator.is_request_piece_supported(text_piece) is False
        assert validator.is_request_piece_supported(image_piece) is True


class TestScorerPromptValidatorMetadata:
    """Test metadata filtering functionality."""

    def test_no_metadata_required_accepts_all(self):
        """Test that validator with no required_metadata accepts all pieces."""
        validator = ScorerPromptValidator()

        piece_with_metadata = PromptRequestPiece(
            role="assistant",
            original_value="text",
            converted_value_data_type="text",
            prompt_metadata={"key": "value"},
        )
        piece_without_metadata = PromptRequestPiece(
            role="assistant", original_value="text", converted_value_data_type="text"
        )

        assert validator.is_request_piece_supported(piece_with_metadata) is True
        assert validator.is_request_piece_supported(piece_without_metadata) is True

    def test_required_metadata_filters_correctly(self):
        """Test that validator correctly filters based on required metadata."""
        validator = ScorerPromptValidator(required_metadata=["category"])

        piece_with_metadata = PromptRequestPiece(
            role="assistant",
            original_value="text",
            converted_value_data_type="text",
            prompt_metadata={"category": "test"},
        )
        piece_without_metadata = PromptRequestPiece(
            role="assistant", original_value="text", converted_value_data_type="text"
        )

        assert validator.is_request_piece_supported(piece_with_metadata) is True
        assert validator.is_request_piece_supported(piece_without_metadata) is False

    def test_multiple_required_metadata_all_must_be_present(self):
        """Test that all required metadata keys must be present."""
        validator = ScorerPromptValidator(required_metadata=["category", "source"])

        piece_with_all_metadata = PromptRequestPiece(
            role="assistant",
            original_value="text",
            converted_value_data_type="text",
            prompt_metadata={"category": "test", "source": "test"},
        )
        piece_with_partial_metadata = PromptRequestPiece(
            role="assistant",
            original_value="text",
            converted_value_data_type="text",
            prompt_metadata={"category": "test"},
        )
        piece_without_metadata = PromptRequestPiece(
            role="assistant", original_value="text", converted_value_data_type="text"
        )

        assert validator.is_request_piece_supported(piece_with_all_metadata) is True
        assert validator.is_request_piece_supported(piece_with_partial_metadata) is False
        assert validator.is_request_piece_supported(piece_without_metadata) is False


class TestScorerPromptValidatorValidate:
    """Test the validate() method."""

    def test_validate_passes_with_valid_pieces(self):
        """Test that validate passes when there are valid pieces."""
        validator = ScorerPromptValidator(supported_data_types=["text"])

        text_piece = PromptRequestPiece(
            role="assistant", original_value="text", converted_value_data_type="text", conversation_id="test"
        )
        response = PromptRequestResponse(request_pieces=[text_piece])

        # Should not raise
        validator.validate(response, objective=None)

    def test_validate_raises_when_no_valid_pieces(self):
        """Test that validate raises error when no pieces are valid."""
        validator = ScorerPromptValidator(supported_data_types=["text"])

        image_piece = PromptRequestPiece(
            role="assistant",
            original_value="image.png",
            converted_value_data_type="image_path",
            conversation_id="test",
        )
        response = PromptRequestResponse(request_pieces=[image_piece])

        with pytest.raises(ValueError, match="There are no valid pieces to score"):
            validator.validate(response, objective=None)

    def test_validate_passes_with_mixed_pieces_when_enforce_false(self):
        """Test that validate passes with mixed pieces when enforce_all_pieces_valid=False."""
        validator = ScorerPromptValidator(supported_data_types=["text"], enforce_all_pieces_valid=False)

        text_piece = PromptRequestPiece(
            role="assistant", original_value="text", converted_value_data_type="text", conversation_id="test"
        )
        image_piece = PromptRequestPiece(
            role="assistant",
            original_value="image.png",
            converted_value_data_type="image_path",
            conversation_id="test",
        )
        response = PromptRequestResponse(request_pieces=[text_piece, image_piece])

        # Should not raise
        validator.validate(response, objective=None)

    def test_validate_raises_with_unsupported_piece_when_enforce_true(self):
        """Test that validate raises error for unsupported pieces when enforce_all_pieces_valid=True."""
        validator = ScorerPromptValidator(supported_data_types=["text"], enforce_all_pieces_valid=True)

        text_piece = PromptRequestPiece(
            role="assistant",
            original_value="text",
            converted_value_data_type="text",
            conversation_id="test",
            id="text-1",
        )
        image_piece = PromptRequestPiece(
            role="assistant",
            original_value="image.png",
            converted_value_data_type="image_path",
            conversation_id="test",
            id="image-1",
        )
        response = PromptRequestResponse(request_pieces=[text_piece, image_piece])

        with pytest.raises(ValueError, match="Request piece image-1 with data type image_path is not supported"):
            validator.validate(response, objective=None)

    def test_validate_raises_when_exceeds_max_pieces(self):
        """Test that validate raises error when response exceeds max_pieces_in_response."""
        validator = ScorerPromptValidator(max_pieces_in_response=2)

        pieces = [
            PromptRequestPiece(
                role="assistant", original_value=f"text{i}", converted_value_data_type="text", conversation_id="test"
            )
            for i in range(3)
        ]
        response = PromptRequestResponse(request_pieces=pieces)

        with pytest.raises(ValueError, match="exceeding the limit of 2"):
            validator.validate(response, objective=None)

    def test_validate_passes_when_within_max_pieces(self):
        """Test that validate passes when response is within max_pieces_in_response."""
        validator = ScorerPromptValidator(max_pieces_in_response=3)

        pieces = [
            PromptRequestPiece(
                role="assistant", original_value=f"text{i}", converted_value_data_type="text", conversation_id="test"
            )
            for i in range(2)
        ]
        response = PromptRequestResponse(request_pieces=pieces)

        # Should not raise
        validator.validate(response, objective=None)

    def test_validate_raises_when_objective_required_but_missing(self):
        """Test that validate raises error when objective is required but not provided."""
        validator = ScorerPromptValidator(is_objective_required=True)

        text_piece = PromptRequestPiece(
            role="assistant", original_value="text", converted_value_data_type="text", conversation_id="test"
        )
        response = PromptRequestResponse(request_pieces=[text_piece])

        with pytest.raises(ValueError, match="Objective is required but not provided"):
            validator.validate(response, objective=None)

    def test_validate_passes_when_objective_provided(self):
        """Test that validate passes when objective is required and provided."""
        validator = ScorerPromptValidator(is_objective_required=True)

        text_piece = PromptRequestPiece(
            role="assistant", original_value="text", converted_value_data_type="text", conversation_id="test"
        )
        response = PromptRequestResponse(request_pieces=[text_piece])

        # Should not raise
        validator.validate(response, objective="test objective")


class TestScorerPromptValidatorCombined:
    """Test combined filtering scenarios."""

    def test_combined_data_type_and_metadata_filtering(self):
        """Test that both data type and metadata filtering work together."""
        validator = ScorerPromptValidator(supported_data_types=["text"], required_metadata=["category"])

        # Valid: correct type and metadata
        valid_piece = PromptRequestPiece(
            role="assistant",
            original_value="text",
            converted_value_data_type="text",
            prompt_metadata={"category": "test"},
        )

        # Invalid: wrong type but has metadata
        wrong_type_piece = PromptRequestPiece(
            role="assistant",
            original_value="image.png",
            converted_value_data_type="image_path",
            prompt_metadata={"category": "test"},
        )

        # Invalid: correct type but missing metadata
        missing_metadata_piece = PromptRequestPiece(
            role="assistant", original_value="text", converted_value_data_type="text"
        )

        assert validator.is_request_piece_supported(valid_piece) is True
        assert validator.is_request_piece_supported(wrong_type_piece) is False
        assert validator.is_request_piece_supported(missing_metadata_piece) is False

    def test_all_validator_options_combined(self):
        """Test validator with all options configured."""
        validator = ScorerPromptValidator(
            supported_data_types=["text"],
            required_metadata=["category"],
            max_pieces_in_response=2,
            enforce_all_pieces_valid=False,
            is_objective_required=True,
        )

        valid_piece = PromptRequestPiece(
            role="assistant",
            original_value="text",
            converted_value_data_type="text",
            prompt_metadata={"category": "test"},
            conversation_id="test",
        )
        invalid_piece = PromptRequestPiece(
            role="assistant",
            original_value="image.png",
            converted_value_data_type="image_path",
            conversation_id="test",
        )

        response = PromptRequestResponse(request_pieces=[valid_piece, invalid_piece])

        # Should pass with valid objective and mixed pieces (enforce_all_pieces_valid=False)
        validator.validate(response, objective="test objective")

        # Should fail without objective
        with pytest.raises(ValueError, match="Objective is required"):
            validator.validate(response, objective=None)
