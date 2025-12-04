# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_converter.text_selection_strategy import (
    IndexSelectionStrategy,
    KeywordSelectionStrategy,
    PositionSelectionStrategy,
    ProportionSelectionStrategy,
    RangeSelectionStrategy,
    RegexSelectionStrategy,
)


class TestIndexSelectionStrategy:
    def test_select_range_full_text(self):
        strategy = IndexSelectionStrategy(start=0, end=None)
        result = strategy.select_range(text="Hello World")
        assert result == (0, 11)

    def test_select_range_partial(self):
        strategy = IndexSelectionStrategy(start=0, end=5)
        result = strategy.select_range(text="Hello World")
        assert result == (0, 5)

    def test_select_range_middle(self):
        strategy = IndexSelectionStrategy(start=6, end=11)
        result = strategy.select_range(text="Hello World")
        assert result == (6, 11)

    def test_select_range_beyond_length(self):
        strategy = IndexSelectionStrategy(start=0, end=100)
        result = strategy.select_range(text="Hello World")
        assert result == (0, 11)

    def test_select_range_negative_start(self):
        strategy = IndexSelectionStrategy(start=-5, end=5)
        result = strategy.select_range(text="Hello World")
        assert result == (0, 5)

    def test_select_range_start_after_end(self):
        strategy = IndexSelectionStrategy(start=10, end=5)
        result = strategy.select_range(text="Hello World")
        assert result == (10, 10)


class TestRegexSelectionStrategy:
    def test_select_range_simple_match(self):
        strategy = RegexSelectionStrategy(pattern=r"World")
        result = strategy.select_range(text="Hello World")
        assert result == (6, 11)

    def test_select_range_pattern_match(self):
        strategy = RegexSelectionStrategy(pattern=r"\d+")
        result = strategy.select_range(text="The number is 12345 here")
        assert result == (14, 19)

    def test_select_range_no_match(self):
        strategy = RegexSelectionStrategy(pattern=r"xyz")
        result = strategy.select_range(text="Hello World")
        assert result == (0, 0)

    def test_select_range_first_match_only(self):
        strategy = RegexSelectionStrategy(pattern=r"\d+")
        result = strategy.select_range(text="First 123 and second 456")
        assert result == (6, 9)

    def test_select_range_complex_pattern(self):
        strategy = RegexSelectionStrategy(pattern=r"password:\s*\w+")
        result = strategy.select_range(text="The password: secret123 is here")
        assert result == (4, 23)


class TestKeywordSelectionStrategy:
    def test_select_range_exact_keyword(self):
        strategy = KeywordSelectionStrategy(keyword="secret")
        result = strategy.select_range(text="The secret is here")
        assert result == (4, 10)

    def test_select_range_with_context_before(self):
        strategy = KeywordSelectionStrategy(keyword="secret", context_before=4)
        result = strategy.select_range(text="The secret is here")
        assert result == (0, 10)

    def test_select_range_with_context_after(self):
        strategy = KeywordSelectionStrategy(keyword="secret", context_after=3)
        result = strategy.select_range(text="The secret is here")
        assert result == (4, 13)

    def test_select_range_with_both_contexts(self):
        strategy = KeywordSelectionStrategy(keyword="secret", context_before=4, context_after=3)
        result = strategy.select_range(text="The secret is here")
        assert result == (0, 13)

    def test_select_range_case_insensitive(self):
        strategy = KeywordSelectionStrategy(keyword="SECRET", case_sensitive=False)
        result = strategy.select_range(text="The secret is here")
        assert result == (4, 10)

    def test_select_range_case_sensitive_no_match(self):
        strategy = KeywordSelectionStrategy(keyword="SECRET", case_sensitive=True)
        result = strategy.select_range(text="The secret is here")
        assert result == (0, 0)

    def test_select_range_keyword_not_found(self):
        strategy = KeywordSelectionStrategy(keyword="xyz")
        result = strategy.select_range(text="The secret is here")
        assert result == (0, 0)

    def test_select_range_context_at_boundaries(self):
        strategy = KeywordSelectionStrategy(keyword="secret", context_before=100, context_after=100)
        result = strategy.select_range(text="The secret is here")
        assert result == (0, 18)


class TestPositionSelectionStrategy:
    def test_select_range_first_half(self):
        strategy = PositionSelectionStrategy(position="first_half")
        result = strategy.select_range(text="0123456789")
        assert result == (0, 5)

    def test_select_range_second_half(self):
        strategy = PositionSelectionStrategy(position="second_half")
        result = strategy.select_range(text="0123456789")
        assert result == (5, 10)

    def test_select_range_first_third(self):
        strategy = PositionSelectionStrategy(position="first_third")
        result = strategy.select_range(text="012345678")
        assert result == (0, 3)

    def test_select_range_second_third(self):
        strategy = PositionSelectionStrategy(position="second_third")
        result = strategy.select_range(text="012345678")
        assert result == (3, 6)

    def test_select_range_last_third(self):
        strategy = PositionSelectionStrategy(position="last_third")
        result = strategy.select_range(text="012345678")
        assert result == (6, 9)

    def test_select_range_first_quarter(self):
        strategy = PositionSelectionStrategy(position="first_quarter")
        result = strategy.select_range(text="0123456789ABCDEF")
        assert result == (0, 4)

    def test_select_range_last_quarter(self):
        strategy = PositionSelectionStrategy(position="last_quarter")
        result = strategy.select_range(text="0123456789ABCDEF")
        assert result == (12, 16)

    def test_invalid_position_raises_error(self):
        with pytest.raises(ValueError, match="Invalid position 'invalid'"):
            PositionSelectionStrategy(position="invalid")


class TestProportionSelectionStrategy:
    def test_select_range_start_anchor(self):
        strategy = ProportionSelectionStrategy(proportion=0.5, anchor="start")
        result = strategy.select_range(text="0123456789")
        assert result == (0, 5)

    def test_select_range_end_anchor(self):
        strategy = ProportionSelectionStrategy(proportion=0.5, anchor="end")
        result = strategy.select_range(text="0123456789")
        assert result == (5, 10)

    def test_select_range_middle_anchor(self):
        strategy = ProportionSelectionStrategy(proportion=0.4, anchor="middle")
        result = strategy.select_range(text="0123456789")
        assert result == (3, 7)

    def test_select_range_random_anchor_with_seed(self):
        strategy = ProportionSelectionStrategy(proportion=0.3, anchor="random", seed=42)
        result1 = strategy.select_range(text="0123456789")
        strategy2 = ProportionSelectionStrategy(proportion=0.3, anchor="random", seed=42)
        result2 = strategy2.select_range(text="0123456789")
        assert result1 == result2  # Same seed should give same result

    def test_select_range_random_anchor_different_seeds(self):
        strategy1 = ProportionSelectionStrategy(proportion=0.3, anchor="random", seed=42)
        strategy2 = ProportionSelectionStrategy(proportion=0.3, anchor="random", seed=43)
        result1 = strategy1.select_range(text="0123456789")
        result2 = strategy2.select_range(text="0123456789")
        # Different seeds may give different results (not guaranteed, but highly likely)
        # Just ensure both are valid ranges
        assert 0 <= result1[0] <= result1[1] <= 10
        assert 0 <= result2[0] <= result2[1] <= 10

    def test_invalid_proportion_too_high_raises_error(self):
        with pytest.raises(ValueError, match="Proportion must be between 0.0 and 1.0"):
            ProportionSelectionStrategy(proportion=1.5, anchor="start")

    def test_invalid_proportion_negative_raises_error(self):
        with pytest.raises(ValueError, match="Proportion must be between 0.0 and 1.0"):
            ProportionSelectionStrategy(proportion=-0.1, anchor="start")

    def test_invalid_anchor_raises_error(self):
        with pytest.raises(ValueError, match="Invalid anchor 'invalid'"):
            ProportionSelectionStrategy(proportion=0.5, anchor="invalid")

    def test_proportion_zero(self):
        strategy = ProportionSelectionStrategy(proportion=0.0, anchor="start")
        result = strategy.select_range(text="0123456789")
        assert result == (0, 0)

    def test_proportion_one(self):
        strategy = ProportionSelectionStrategy(proportion=1.0, anchor="start")
        result = strategy.select_range(text="0123456789")
        assert result == (0, 10)


class TestRangeSelectionStrategy:
    def test_select_range_full_range(self):
        strategy = RangeSelectionStrategy(start_proportion=0.0, end_proportion=1.0)
        result = strategy.select_range(text="0123456789")
        assert result == (0, 10)

    def test_select_range_middle_half(self):
        strategy = RangeSelectionStrategy(start_proportion=0.25, end_proportion=0.75)
        result = strategy.select_range(text="0123456789")
        assert result == (2, 7)

    def test_select_range_first_quarter(self):
        strategy = RangeSelectionStrategy(start_proportion=0.0, end_proportion=0.25)
        result = strategy.select_range(text="0123456789ABCDEF")
        assert result == (0, 4)

    def test_select_range_last_quarter(self):
        strategy = RangeSelectionStrategy(start_proportion=0.75, end_proportion=1.0)
        result = strategy.select_range(text="0123456789ABCDEF")
        assert result == (12, 16)

    def test_invalid_start_proportion_too_high_raises_error(self):
        with pytest.raises(ValueError, match="start_proportion must be between 0.0 and 1.0"):
            RangeSelectionStrategy(start_proportion=1.5, end_proportion=1.0)

    def test_invalid_end_proportion_too_high_raises_error(self):
        with pytest.raises(ValueError, match="end_proportion must be between 0.0 and 1.0"):
            RangeSelectionStrategy(start_proportion=0.0, end_proportion=1.5)

    def test_invalid_start_proportion_negative_raises_error(self):
        with pytest.raises(ValueError, match="start_proportion must be between 0.0 and 1.0"):
            RangeSelectionStrategy(start_proportion=-0.1, end_proportion=1.0)

    def test_invalid_start_equals_end_raises_error(self):
        with pytest.raises(ValueError, match="start_proportion .* must be less than end_proportion"):
            RangeSelectionStrategy(start_proportion=0.5, end_proportion=0.5)

    def test_invalid_start_greater_than_end_raises_error(self):
        with pytest.raises(ValueError, match="start_proportion .* must be less than end_proportion"):
            RangeSelectionStrategy(start_proportion=0.75, end_proportion=0.25)
