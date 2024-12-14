# 5. Human in the Loop Scoring

This is possible using the `HITLScorer` class. It can take input from a csv file or directly via standard input. See the [tests](../../../tests/unit/score/test_hitl.py) for an explicit example; the csv format should have the following headers in any order, followed by the data separated by commas:

```
score_value, score_value_description, score_type, score_category, score_rationale, score_metadata, scorer_class_identifier ,prompt_request_response_id
```
