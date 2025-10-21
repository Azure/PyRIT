# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Base class for scenario attack strategies with group-based aggregation.

This module provides a generic base class for creating enum-based attack strategy
hierarchies where strategies can be grouped by categories (e.g., complexity, encoding type)
and automatically expanded during scenario initialization.
"""

from enum import Enum
from typing import Set, TypeVar


# TypeVar for the enum subclass itself
T = TypeVar("T", bound="ScenarioAttackStrategy")


class ScenarioAttackStrategy(Enum):
    """
    Base class for attack strategies with tag-based categorization and aggregation.

    This class provides a pattern for defining attack strategies as enums where each
    strategy has a set of tags for flexible categorization. It supports aggregate tags
    (like "easy", "moderate", "difficult" or "fast", "medium") that automatically expand 
    to include all strategies with that tag.

    **Tags**: Flexible categorization system where strategies can have multiple tags
    (e.g., {"easy", "converter"}, {"difficult", "multi_turn"})

    Subclasses should define their enum members with (value, tags) tuples and
    override the get_aggregate_tags() classmethod to specify which tags
    represent aggregates that should expand.

    **Convention**: All subclasses should include `ALL = ("all", {"all"})` as the first
    aggregate member. The base class automatically handles expanding "all" to
    include all non-aggregate strategies.

    The normalization process automatically:
    1. Expands aggregate tags into their constituent strategies
    2. Excludes the aggregate tag enum members themselves from the final set
    3. Handles the special "all" tag by expanding to all non-aggregate strategies

    Example:
        >>> class MyAttackStrategy(ScenarioAttackStrategy):
        ...     '''Attack strategies for my scenario.'''
        ...
        ...     # Aggregate members (special markers that expand)
        ...     ALL = ("all", {"all"})
        ...     EASY = ("easy", {"easy"})
        ...     MODERATE = ("moderate", {"moderate"})
        ...     CONVERTER = ("converter", {"converter"})
        ...
        ...     # Baseline strategy
        ...     Baseline = ("baseline", {"baseline"})
        ...
        ...     # Strategies with multiple tags
        ...     Base64 = ("base64", {"easy", "converter", "encoding"})
        ...     ROT13 = ("rot13", {"easy", "converter", "encoding"})
        ...     Advanced = ("advanced", {"moderate", "multi_turn"})
        ...
        ...     @classmethod
        ...     def get_aggregate_tags(cls):
        ...         return super().get_aggregate_tags() | {"easy", "moderate", "converter"}
        ...
        >>> # User specifies aggregate tag
        >>> strategies = {MyAttackStrategy.EASY}
        >>> normalized = MyAttackStrategy.normalize_strategies(strategies)
        >>> # Returns: {Base64, ROT13} (all strategies tagged with "easy")
        >>>
        >>> # Filter by any tag
        >>> converter_strategies = MyAttackStrategy.get_strategies_by_tag("converter")
        >>> # Returns: {Base64, ROT13}
        >>>
        >>> # Get all strategies
        >>> all_strategies = MyAttackStrategy.normalize_strategies({MyAttackStrategy.ALL})
        >>> # Returns: {Baseline, Base64, ROT13, Advanced}
    """

    _tags: set[str]

    def __new__(cls, value: str, tags: set[str] | None = None) -> "ScenarioAttackStrategy":
        """
        Create a new ScenarioAttackStrategy with value and tags.

        Args:
            value (str): The strategy identifier/name.
            tags (set[str] | None): Tags for categorization (e.g., {"easy", "converter", "encoding"}).

        Returns:
            ScenarioAttackStrategy: The new enum member.
        """
        obj = object.__new__(cls)
        obj._value_ = value
        obj._tags = tags or set()  # type: ignore[misc]
        return obj

    @property
    def tags(self) -> set[str]:
        """
        Get the tags for this attack strategy.

        Tags provide a flexible categorization system, allowing strategies
        to be classified along multiple dimensions (e.g., by complexity, type, or technique).

        Returns:
            set[str]: The tags (e.g., {"easy", "converter", "encoding"}).
        """
        return self._tags

    @classmethod
    def get_aggregate_tags(cls: type[T]) -> Set[str]:
        """
        Get the set of tags that represent aggregate categories.

        Subclasses should override this method to specify which tags
        are aggregate markers (e.g., {"easy", "moderate", "difficult"} for complexity-based
        scenarios or {"fast", "medium"} for speed-based scenarios).

        The base class automatically includes "all" as an aggregate tag that expands
        to all non-aggregate strategies.

        Returns:
            Set[str]: Set of tags that represent aggregates.
        """
        return {"all"}

    @classmethod
    def get_strategies_by_tag(cls: type[T], tag: str) -> Set[T]:
        """
        Get all attack strategies that have a specific tag.

        This method returns concrete attack strategies (not aggregate markers)
        that include the specified tag.

        Args:
            tag (str): The tag to filter by (e.g., "easy", "converter", "multi_turn").

        Returns:
            Set[T]: Set of strategies that include the specified tag, excluding
                    any aggregate markers.
        """
        aggregate_tags = cls.get_aggregate_tags()
        return {
            strategy
            for strategy in cls
            if tag in strategy.tags and strategy.value not in aggregate_tags
        }

    @classmethod
    def normalize_strategies(cls: type[T], strategies: Set[T]) -> Set[T]:
        """
        Normalize a set of attack strategies by expanding aggregate tags.

        This method processes a set of strategies and expands any aggregate tags
        (like EASY, MODERATE, DIFFICULT or FAST, MEDIUM) into their constituent concrete strategies.
        The aggregate tag markers themselves are removed from the result.

        The special "all" tag is automatically supported and expands to all non-aggregate strategies.

        Args:
            strategies (Set[T]): The initial set of attack strategies, which may include
                                aggregate tags.

        Returns:
            Set[T]: The normalized set of concrete attack strategies with aggregate tags
                   expanded and removed.

        Example:
            >>> strategies = {MyAttackStrategy.EASY, MyAttackStrategy.Base64}
            >>> normalized = MyAttackStrategy.normalize_strategies(strategies)
            >>> # EASY is expanded to all strategies with "easy" tag, EASY itself is removed
        """
        normalized_strategies = set(strategies)

        # Find aggregate tags in the input and expand them
        aggregate_tags = cls.get_aggregate_tags()
        aggregates_to_expand = {
            tag 
            for strategy in strategies 
            if strategy.value in aggregate_tags
            for tag in strategy.tags
        }

        for aggregate_tag in aggregates_to_expand:
            # Remove the aggregate marker itself
            aggregate_marker = next(
                (s for s in normalized_strategies if s.value == aggregate_tag), None
            )
            if aggregate_marker:
                normalized_strategies.remove(aggregate_marker)

            # Special handling for "all" tag - expand to all non-aggregate strategies
            if aggregate_tag == "all":
                normalized_strategies.update({
                    strategy for strategy in cls 
                    if strategy.value not in aggregate_tags
                })
            else:
                # Add all strategies with that tag
                normalized_strategies.update(cls.get_strategies_by_tag(aggregate_tag))

        return normalized_strategies
