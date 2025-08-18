# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import MagicMock, patch

from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece
from pyrit.score.plagiarism_scorer import PlagiarismScorer, PlagiarismMetric, _plagiarism_score, _tokenize, _lcs_length, _levenshtein_distance, _ngram_set


@pytest.mark.usefixtures("patch_central_database")
class TestPlagiarismScorer:
    """Test cases for the PlagiarismScorer class."""

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        reference_text = "This is a sample reference text"
        scorer = PlagiarismScorer(reference_text=reference_text)
        
        assert scorer.reference_text == reference_text
        assert scorer.metric == PlagiarismMetric.LCS
        assert scorer.n == 5
        assert scorer.scorer_type == "float_scale"

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        reference_text = "Custom reference text"
        metric = PlagiarismMetric.JACCARD
        n = 3
        
        scorer = PlagiarismScorer(
            reference_text=reference_text,
            metric=metric,
            n=n
        )
        
        assert scorer.reference_text == reference_text
        assert scorer.metric == metric
        assert scorer.n == n
        assert scorer.scorer_type == "float_scale"

    @pytest.mark.asyncio
    async def test_score_async_lcs_metric(self):
        """Test scoring with LCS metric."""
        reference_text = "The quick brown fox jumps over the lazy dog"
        response_text = "The quick brown fox runs over the lazy dog"
        
        scorer = PlagiarismScorer(
            reference_text=reference_text,
            metric=PlagiarismMetric.LCS
        )
        
        request_piece = PromptRequestPiece(
            role="assistant",
            original_value=response_text,
            converted_value=response_text,
            converted_value_data_type="text"
        )
        
        scores = await scorer._score_async(request_response=request_piece)
        
        assert len(scores) == 1
        score = scores[0]
        assert score.score_type == "float_scale"
        assert "Plagiarism score using lcs metric" in score.score_value_description
        assert score.score_rationale == "Score is deterministic."
        assert score.prompt_request_response_id == request_piece.id
        
        # Verify the score value is reasonable (should be high due to similarity)
        score_value = float(score.score_value)
        assert 0.0 <= score_value <= 1.0
        assert score_value > 0.8  # Should be high similarity

    @pytest.mark.asyncio
    async def test_score_async_levenshtein_metric(self):
        """Test scoring with Levenshtein metric."""
        reference_text = "Hello world"
        response_text = "Hello world test"
        
        scorer = PlagiarismScorer(
            reference_text=reference_text,
            metric=PlagiarismMetric.LEVENSHTEIN
        )
        
        request_piece = PromptRequestPiece(
            role="assistant",
            original_value=response_text,
            converted_value=response_text,
            converted_value_data_type="text"
        )
        
        scores = await scorer._score_async(request_response=request_piece)
        
        assert len(scores) == 1
        score = scores[0]
        assert "Plagiarism score using levenshtein metric" in score.score_value_description
        
        score_value = float(score.score_value)
        assert 0.0 <= score_value <= 1.0

    @pytest.mark.asyncio
    async def test_score_async_jaccard_metric(self):
        """Test scoring with Jaccard metric."""
        reference_text = "The quick brown fox jumps over the lazy dog"
        response_text = "The quick brown fox runs over the lazy cat"
        
        scorer = PlagiarismScorer(
            reference_text=reference_text,
            metric=PlagiarismMetric.JACCARD,
            n=3
        )
        
        request_piece = PromptRequestPiece(
            role="assistant",
            original_value=response_text,
            converted_value=response_text,
            converted_value_data_type="text"
        )
        
        scores = await scorer._score_async(request_response=request_piece)
        
        assert len(scores) == 1
        score = scores[0]
        assert "Plagiarism score using jaccard metric" in score.score_value_description
        
        score_value = float(score.score_value)
        assert 0.0 <= score_value <= 1.0

    @pytest.mark.asyncio
    async def test_score_async_non_text_response(self):
        """Test scoring with non-text response."""
        reference_text = "Sample reference text"
        scorer = PlagiarismScorer(reference_text=reference_text)
        
        request_piece = PromptRequestPiece(
            role="assistant",
            original_value="image_data",
            converted_value="image_data",
            converted_value_data_type="image_path"
        )
        
        scores = await scorer._score_async(request_response=request_piece)
        
        assert len(scores) == 1
        score = scores[0]
        assert score.score_value == "0.0"
        assert score.score_value_description == "Non text response"
        assert score.score_rationale == "Model response is not text."
        assert score.score_metadata == "None"

    @pytest.mark.asyncio
    async def test_score_async_empty_response(self):
        """Test scoring with empty response."""
        reference_text = "Sample reference text"
        scorer = PlagiarismScorer(reference_text=reference_text)
        
        request_piece = PromptRequestPiece(
            role="assistant",
            original_value="",
            converted_value="",
            converted_value_data_type="text"
        )
        
        scores = await scorer._score_async(request_response=request_piece)
        
        assert len(scores) == 1
        score = scores[0]
        score_value = float(score.score_value)
        assert score_value == 0.0

    @pytest.mark.asyncio
    async def test_score_async_identical_texts(self):
        """Test scoring with identical texts."""
        reference_text = "This is exactly the same text"
        
        scorer = PlagiarismScorer(
            reference_text=reference_text,
            metric=PlagiarismMetric.LCS
        )
        
        request_piece = PromptRequestPiece(
            role="assistant",
            original_value=reference_text,
            converted_value=reference_text,
            converted_value_data_type="text"
        )
        
        scores = await scorer._score_async(request_response=request_piece)
        
        assert len(scores) == 1
        score = scores[0]
        score_value = float(score.score_value)
        assert score_value == 1.0  # Should be perfect match

    @pytest.mark.asyncio
    async def test_score_async_completely_different_texts(self):
        """Test scoring with completely different texts."""
        reference_text = "Apple banana cherry"
        response_text = "Dog elephant fox"
        
        scorer = PlagiarismScorer(
            reference_text=reference_text,
            metric=PlagiarismMetric.LCS
        )
        
        request_piece = PromptRequestPiece(
            role="assistant",
            original_value=response_text,
            converted_value=response_text,
            converted_value_data_type="text"
        )
        
        scores = await scorer._score_async(request_response=request_piece)
        
        assert len(scores) == 1
        score = scores[0]
        score_value = float(score.score_value)
        assert score_value == 0.0  # Should be no similarity

    @pytest.mark.asyncio
    async def test_score_async_adds_to_memory(self):
        """Test that scoring adds results to memory."""
        memory = MagicMock(MemoryInterface)
        reference_text = "Test reference text"
        scorer = PlagiarismScorer(reference_text=reference_text)
        
        request_piece = PromptRequestPiece(
            role="assistant",
            original_value="Test response text",
            converted_value="Test response text",
            converted_value_data_type="text"
        )
        
        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
            await scorer.score_async(request_piece)
            memory.add_scores_to_memory.assert_called_once()

    def test_validate_text_data_type(self):
        """Test validation with text data type."""
        reference_text = "Test reference text"
        scorer = PlagiarismScorer(reference_text=reference_text)
        
        request_piece = PromptRequestPiece(
            role="assistant",
            original_value="Test response text",
            converted_value="Test response text",
            converted_value_data_type="text"
        )
        
        # Should not raise an exception
        scorer.validate(request_piece)

    def test_validate_non_text_data_type_raises_error(self):
        """Test validation with non-text data type raises ValueError."""
        reference_text = "Test reference text"
        scorer = PlagiarismScorer(reference_text=reference_text)
        
        request_piece = PromptRequestPiece(
            role="assistant",
            original_value="image_data",
            converted_value="image_data",
            converted_value_data_type="image_path"
        )
        
        with pytest.raises(ValueError, match="PlagiarismScorer only supports text data type"):
            scorer.validate(request_piece)

    @pytest.mark.asyncio
    async def test_score_text_async_integration(self):
        """Test scoring using the convenience method score_text_async."""
        reference_text = "The quick brown fox"
        scorer = PlagiarismScorer(reference_text=reference_text)
        
        scores = await scorer.score_text_async("The quick brown dog")
        
        assert len(scores) == 1
        score = scores[0]
        assert score.score_type == "float_scale"
        score_value = float(score.score_value)
        assert 0.0 <= score_value <= 1.0
        assert score_value > 0.5  # Should have some similarity


class TestPlagiarismScorerUtilityFunctions:
    """Test cases for utility functions in the plagiarism scorer."""

    def test_tokenize_basic(self):
        """Test basic tokenization functionality."""
        text = "Hello World Test"
        tokens = _tokenize(text)
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_with_punctuation(self):
        """Test tokenization with punctuation removal."""
        text = "Hello, world! How are you?"
        tokens = _tokenize(text)
        assert tokens == ["hello", "world", "how", "are", "you"]

    def test_tokenize_empty_string(self):
        """Test tokenization with empty string."""
        tokens = _tokenize("")
        assert tokens == []

    def test_lcs_length_identical(self):
        """Test LCS with identical sequences."""
        a = ["hello", "world", "test"]
        b = ["hello", "world", "test"]
        length = _lcs_length(a, b)
        assert length == 3

    def test_lcs_length_different(self):
        """Test LCS with different sequences."""
        a = ["hello", "world", "test"]
        b = ["hello", "test", "case"]
        length = _lcs_length(a, b)
        assert length == 2  # "hello" and "test"

    def test_lcs_length_empty(self):
        """Test LCS with empty sequences."""
        a = []
        b = ["hello", "world"]
        length = _lcs_length(a, b)
        assert length == 0

    def test_levenshtein_distance_identical(self):
        """Test Levenshtein distance with identical sequences."""
        a = ["hello", "world"]
        b = ["hello", "world"]
        distance = _levenshtein_distance(a, b)
        assert distance == 0

    def test_levenshtein_distance_different(self):
        """Test Levenshtein distance with different sequences."""
        a = ["hello", "world"]
        b = ["hello", "test"]
        distance = _levenshtein_distance(a, b)
        assert distance == 1  # One substitution

    def test_levenshtein_distance_empty(self):
        """Test Levenshtein distance with empty sequences."""
        a = []
        b = ["hello", "world"]
        distance = _levenshtein_distance(a, b)
        assert distance == 2  # Two insertions

    def test_ngram_set_basic(self):
        """Test n-gram set generation."""
        tokens = ["the", "quick", "brown", "fox"]
        ngrams = _ngram_set(tokens, 2)
        expected = {("the", "quick"), ("quick", "brown"), ("brown", "fox")}
        assert ngrams == expected

    def test_ngram_set_longer_n(self):
        """Test n-gram set with n longer than token list."""
        tokens = ["hello", "world"]
        ngrams = _ngram_set(tokens, 5)
        assert ngrams == set()

    def test_ngram_set_empty_tokens(self):
        """Test n-gram set with empty token list."""
        tokens = []
        ngrams = _ngram_set(tokens, 2)
        assert ngrams == set()

    def test_plagiarism_score_lcs(self):
        """Test plagiarism score with LCS metric."""
        response = "The quick brown fox"
        reference = "The quick brown dog"
        score = _plagiarism_score(response, reference, metric="lcs")
        assert 0.0 <= score <= 1.0
        assert score == 0.75  # 3/4 words match

    def test_plagiarism_score_levenshtein(self):
        """Test plagiarism score with Levenshtein metric."""
        response = "hello world"
        reference = "hello world"
        score = _plagiarism_score(response, reference, metric="levenshtein")
        assert score == 1.0  # Perfect match

    def test_plagiarism_score_jaccard(self):
        """Test plagiarism score with Jaccard metric."""
        response = "the quick brown fox jumps"
        reference = "the quick brown dog runs"
        score = _plagiarism_score(response, reference, metric="jaccard", n=2)
        assert 0.0 <= score <= 1.0

    def test_plagiarism_score_empty_texts(self):
        """Test plagiarism score with empty texts."""
        score = _plagiarism_score("", "hello world", metric="lcs")
        assert score == 0.0

    def test_plagiarism_score_invalid_metric(self):
        """Test plagiarism score with invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be 'lcs', 'levenshtein', or 'jaccard'"):
            _plagiarism_score("hello", "world", metric="invalid")

    def test_plagiarism_score_case_insensitive(self):
        """Test that plagiarism score is case insensitive."""
        response = "Hello World"
        reference = "hello world"
        score = _plagiarism_score(response, reference, metric="lcs")
        assert score == 1.0  # Should be perfect match despite case difference

    def test_plagiarism_score_lcs_reference_contained_in_response(self):
        """Test LCS metric returns 1.0 when reference text is contained in response."""
        reference = "It was a bright cold day in April"
        response = "The famous opening line states: It was a bright cold day in April, and the clocks were striking thirteen."
        score = _plagiarism_score(response, reference, metric="lcs")
        assert score == 1.0  # Should be perfect match when reference is contained

    def test_plagiarism_score_levenshtein_reference_contained_in_response(self):
        """Test Levenshtein metric returns 1.0 when reference text is contained in response."""
        reference = "The quick brown fox jumps"
        response = "Here is the sentence: The quick brown fox jumps over the lazy dog."
        score = _plagiarism_score(response, reference, metric="levenshtein")
        assert score == 1.0  # Should be perfect match when reference is contained

    def test_plagiarism_score_jaccard_reference_contained_in_response(self):
        """Test Jaccard metric returns 1.0 when reference text is contained in response."""
        reference = "Hello world this is a test"
        response = "The AI model responded with: Hello world this is a test message for validation."
        score = _plagiarism_score(response, reference, metric="jaccard", n=3)
        assert score == 1.0  # Should be perfect match when reference is contained


class TestPlagiarismMetricEnum:
    """Test cases for the PlagiarismMetric enum."""

    def test_plagiarism_metric_values(self):
        """Test that enum values are correct."""
        assert PlagiarismMetric.LCS.value == "lcs"
        assert PlagiarismMetric.LEVENSHTEIN.value == "levenshtein"
        assert PlagiarismMetric.JACCARD.value == "jaccard"

    def test_plagiarism_metric_membership(self):
        """Test enum membership."""
        assert PlagiarismMetric.LCS in PlagiarismMetric
        assert PlagiarismMetric.LEVENSHTEIN in PlagiarismMetric
        assert PlagiarismMetric.JACCARD in PlagiarismMetric
