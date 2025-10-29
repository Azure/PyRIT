# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from contextlib import closing
from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError

from pyrit.memory import AzureSQLMemory, SeedEntry
from pyrit.memory.memory_models import Base, PromptMemoryEntry, AttackResultEntry, ScenarioResultEntry
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    MessagePiece,
    ScenarioIdentifier,
    ScenarioResult,
    SeedPrompt,
)


def remove_entry(memory: AzureSQLMemory, entry: Base) -> None:  # type: ignore
    """
    Removes an entry from the Table for testing purposes.

    Args:
        entry: An instance of a SQLAlchemy model to be removed from the Table.
    """
    with closing(memory.get_session()) as session:
        try:
            session.delete(entry)
            session.commit()
        except SQLAlchemyError:
            session.rollback()


@pytest.mark.asyncio
async def test_get_seeds_with_metadata_filter(azuresql_instance: AzureSQLMemory):
    value1 = str(uuid4())
    # Integers should work properly as values in the metadata dict
    value2 = np.random.randint(0, 10000)
    sp1 = SeedPrompt(value="sp1", data_type="text", metadata={"key1": value1}, added_by="test")
    sp2 = SeedPrompt(value="sp2", data_type="text", metadata={"key1": value2, "key2": value1}, added_by="test")
    entry1, entry2 = SeedEntry(entry=sp1), SeedEntry(entry=sp2)
    azuresql_instance._insert_entries(entries=[entry1, entry2])

    result = azuresql_instance.get_seeds(metadata={"key1": value1})
    result2 = azuresql_instance.get_seeds(metadata={"key1": value2, "key2": value1})
    assert len(result) == 1
    assert len(result2) == 1
    assert result[0].metadata == {"key1": value1}
    assert result2[0].metadata == {"key1": value2, "key2": value1}

    # Clean up
    remove_entry(azuresql_instance, entry1)
    remove_entry(azuresql_instance, entry2)
    # Ensure that entries are removed
    assert azuresql_instance.get_seeds(metadata={"key1": value1}) == []
    assert azuresql_instance.get_seeds(metadata={"key2": value1}) == []


@pytest.mark.asyncio
async def test_get_attack_results_by_harm_categories(azuresql_instance: AzureSQLMemory):
    """Integration test for SQL Azure JSON filtering on targeted harm categories."""
    
    # Use unique conversation IDs to avoid test pollution
    test_id = str(uuid4())[:8]
    
    # Create message pieces with harm categories
    piece1 = MessagePiece(
        conversation_id=f"conv_harm_1_{test_id}",
        role="user",
        original_value="Test 1",
        converted_value="Test 1",
        targeted_harm_categories=["hate", "violence"],
    )
    piece2 = MessagePiece(
        conversation_id=f"conv_harm_2_{test_id}",
        role="user",
        original_value="Test 2",
        converted_value="Test 2",
        targeted_harm_categories=["hate"],
    )
    piece3 = MessagePiece(
        conversation_id=f"conv_harm_3_{test_id}",
        role="user",
        original_value="Test 3",
        converted_value="Test 3",
        targeted_harm_categories=["violence"],
    )
    
    azuresql_instance.add_message_pieces_to_memory(message_pieces=[piece1, piece2, piece3])
    
    # Create attack results
    result1 = AttackResult(
        conversation_id=f"conv_harm_1_{test_id}",
        objective="Test objective 1",
        attack_identifier={"name": "test_attack"},
        outcome=AttackOutcome.SUCCESS,
    )
    result2 = AttackResult(
        conversation_id=f"conv_harm_2_{test_id}",
        objective="Test objective 2",
        attack_identifier={"name": "test_attack"},
        outcome=AttackOutcome.SUCCESS,
    )
    result3 = AttackResult(
        conversation_id=f"conv_harm_3_{test_id}",
        objective="Test objective 3",
        attack_identifier={"name": "test_attack"},
        outcome=AttackOutcome.FAILURE,
    )
    
    azuresql_instance.add_attack_results_to_memory(attack_results=[result1, result2, result3])
    
    try:
        # Test filtering by single harm category
        results = azuresql_instance.get_attack_results(targeted_harm_categories=["hate"])
        # Filter to only results from this test
        results = [r for r in results if test_id in r.conversation_id]
        assert len(results) == 2
        conv_ids = {r.conversation_id for r in results}
        assert f"conv_harm_1_{test_id}" in conv_ids
        assert f"conv_harm_2_{test_id}" in conv_ids
        
        # Test filtering by multiple harm categories (ALL must be present)
        results = azuresql_instance.get_attack_results(targeted_harm_categories=["hate", "violence"])
        results = [r for r in results if test_id in r.conversation_id]
        assert len(results) == 1
        assert results[0].conversation_id == f"conv_harm_1_{test_id}"
        
        # Test filtering with no matches
        results = azuresql_instance.get_attack_results(targeted_harm_categories=["hate", "self-harm"])
        results = [r for r in results if test_id in r.conversation_id]
        assert len(results) == 0
    finally:
        # Clean up: Delete all data created by this test
        from pyrit.memory.memory_models import PromptMemoryEntry, AttackResultEntry
        
        with closing(azuresql_instance.get_session()) as session:
            try:
                # Delete attack results
                session.query(AttackResultEntry).filter(
                    AttackResultEntry.conversation_id.in_([
                        f"conv_harm_1_{test_id}",
                        f"conv_harm_2_{test_id}",
                        f"conv_harm_3_{test_id}",
                    ])
                ).delete(synchronize_session=False)
                
                # Delete message pieces
                session.query(PromptMemoryEntry).filter(
                    PromptMemoryEntry.conversation_id.in_([
                        f"conv_harm_1_{test_id}",
                        f"conv_harm_2_{test_id}",
                        f"conv_harm_3_{test_id}",
                    ])
                ).delete(synchronize_session=False)
                
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                print(f"Cleanup failed: {e}")


@pytest.mark.asyncio
async def test_get_attack_results_by_labels(azuresql_instance: AzureSQLMemory):
    """Integration test for SQL Azure JSON filtering on labels."""
    
    # Use unique conversation IDs to avoid test pollution
    test_id = str(uuid4())[:8]
    
    # Create message pieces with labels
    piece1 = MessagePiece(
        conversation_id=f"conv_label_1_{test_id}",
        role="user",
        original_value="Test 1",
        converted_value="Test 1",
        labels={"op_id": f"op123_{test_id}", "category": "test", "priority": "high"},
    )
    piece2 = MessagePiece(
        conversation_id=f"conv_label_2_{test_id}",
        role="user",
        original_value="Test 2",
        converted_value="Test 2",
        labels={"op_id": f"op123_{test_id}", "category": "test"},
    )
    piece3 = MessagePiece(
        conversation_id=f"conv_label_3_{test_id}",
        role="user",
        original_value="Test 3",
        converted_value="Test 3",
        labels={"op_id": f"op456_{test_id}"},
    )
    
    azuresql_instance.add_message_pieces_to_memory(message_pieces=[piece1, piece2, piece3])
    
    # Create attack results
    result1 = AttackResult(
        conversation_id=f"conv_label_1_{test_id}",
        objective="Test objective 1",
        attack_identifier={"name": "test_attack"},
        outcome=AttackOutcome.SUCCESS,
    )
    result2 = AttackResult(
        conversation_id=f"conv_label_2_{test_id}",
        objective="Test objective 2",
        attack_identifier={"name": "test_attack"},
        outcome=AttackOutcome.SUCCESS,
    )
    result3 = AttackResult(
        conversation_id=f"conv_label_3_{test_id}",
        objective="Test objective 3",
        attack_identifier={"name": "test_attack"},
        outcome=AttackOutcome.FAILURE,
    )
    
    azuresql_instance.add_attack_results_to_memory(attack_results=[result1, result2, result3])
    
    try:
        # Test filtering by single label
        results = azuresql_instance.get_attack_results(labels={"op_id": f"op123_{test_id}"})
        assert len(results) == 2
        conv_ids = {r.conversation_id for r in results}
        assert f"conv_label_1_{test_id}" in conv_ids
        assert f"conv_label_2_{test_id}" in conv_ids
        
        # Test filtering by multiple labels (ALL must be present)
        results = azuresql_instance.get_attack_results(labels={"op_id": f"op123_{test_id}", "category": "test"})
        assert len(results) == 2
        
        results = azuresql_instance.get_attack_results(labels={"op_id": f"op123_{test_id}", "priority": "high"})
        assert len(results) == 1
        assert results[0].conversation_id == f"conv_label_1_{test_id}"
        
        # Test filtering with no matches
        results = azuresql_instance.get_attack_results(labels={"op_id": "nonexistent"})
        results = [r for r in results if test_id in r.conversation_id]
        assert len(results) == 0
    finally:
        # Clean up: Delete all data created by this test
        from pyrit.memory.memory_models import PromptMemoryEntry, AttackResultEntry
        
        with closing(azuresql_instance.get_session()) as session:
            try:
                # Delete attack results
                session.query(AttackResultEntry).filter(
                    AttackResultEntry.conversation_id.in_([
                        f"conv_label_1_{test_id}",
                        f"conv_label_2_{test_id}",
                        f"conv_label_3_{test_id}",
                    ])
                ).delete(synchronize_session=False)
                
                # Delete message pieces
                session.query(PromptMemoryEntry).filter(
                    PromptMemoryEntry.conversation_id.in_([
                        f"conv_label_1_{test_id}",
                        f"conv_label_2_{test_id}",
                        f"conv_label_3_{test_id}",
                    ])
                ).delete(synchronize_session=False)
                
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                print(f"Cleanup failed: {e}")


@pytest.mark.asyncio
async def test_get_scenario_results_by_labels(azuresql_instance: AzureSQLMemory):
    """Integration test for SQL Azure JSON filtering on scenario result labels."""
    
    # Use unique names to avoid test pollution
    test_id = str(uuid4())[:8]
    
    # Create scenario results with labels
    scenario1 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"Test Scenario 1 {test_id}", scenario_version=1),
        objective_target_identifier={"endpoint": "https://api.openai.com"},
        attack_results={},
        labels={"environment": "test", "priority": "high", "team": "red", "test_id": test_id},
    )
    scenario2 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"Test Scenario 2 {test_id}", scenario_version=1),
        objective_target_identifier={"endpoint": "https://api.azure.com"},
        attack_results={},
        labels={"environment": "test", "priority": "high", "test_id": test_id},
    )
    scenario3 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"Test Scenario 3 {test_id}", scenario_version=1),
        objective_target_identifier={"endpoint": "https://api.anthropic.com"},
        attack_results={},
        labels={"environment": "prod", "test_id": test_id},
    )
    
    azuresql_instance.add_scenario_results_to_memory(scenario_results=[scenario1, scenario2, scenario3])
    
    try:
        # Test filtering by single label
        results = azuresql_instance.get_scenario_results(labels={"environment": "test", "test_id": test_id})
        assert len(results) == 2
        names = {r.scenario_identifier.name for r in results}
        assert f"Test Scenario 1 {test_id}" in names
        assert f"Test Scenario 2 {test_id}" in names
        
        # Test filtering by multiple labels (ALL must be present)
        results = azuresql_instance.get_scenario_results(labels={"environment": "test", "priority": "high", "test_id": test_id})
        assert len(results) == 2
        
        results = azuresql_instance.get_scenario_results(labels={"environment": "test", "team": "red", "test_id": test_id})
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"Test Scenario 1 {test_id}"
        
        # Test filtering with no matches
        results = azuresql_instance.get_scenario_results(labels={"environment": "staging", "test_id": test_id})
        assert len(results) == 0
    finally:
        # Clean up: Delete all scenario results created by this test
        with closing(azuresql_instance.get_session()) as session:
            try:
                # Delete all scenario results with the test_id label
                all_results = session.query(ScenarioResultEntry).filter(
                    ScenarioResultEntry.labels.isnot(None)
                ).all()
                
                for result in all_results:
                    if result.labels and result.labels.get("test_id") == test_id:
                        session.delete(result)
                
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                print(f"Cleanup failed: {e}")


@pytest.mark.asyncio
async def test_get_scenario_results_by_target_endpoint(azuresql_instance: AzureSQLMemory):
    """Integration test for SQL Azure case-insensitive endpoint filtering."""
    
    # Use unique names to avoid test pollution
    test_id = str(uuid4())[:8]
    
    # Create scenario results with different endpoints
    scenario1 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"OpenAI Test {test_id}", scenario_version=1),
        objective_target_identifier={"endpoint": f"https://api-{test_id}.openai.com/v1/chat"},
        attack_results={},
    )
    scenario2 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"Azure OpenAI Test {test_id}", scenario_version=1),
        objective_target_identifier={"endpoint": f"https://myresource-{test_id}.openai.azure.com/openai"},
        attack_results={},
    )
    scenario3 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"Anthropic Test {test_id}", scenario_version=1),
        objective_target_identifier={"endpoint": f"https://api-{test_id}.anthropic.com/v1/messages"},
        attack_results={},
    )
    scenario4 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"Azure Other {test_id}", scenario_version=1),
        objective_target_identifier={"endpoint": f"https://myresource-{test_id}.cognitiveservices.azure.com"},
        attack_results={},
    )
    
    azuresql_instance.add_scenario_results_to_memory(scenario_results=[scenario1, scenario2, scenario3, scenario4])
    
    try:
        # Test case-insensitive substring matching - search for test_id specific endpoints
        results = azuresql_instance.get_scenario_results(objective_target_endpoint=test_id)
        assert len(results) == 4  # All should have test_id in their endpoint
        
        results = azuresql_instance.get_scenario_results(objective_target_endpoint=f"openai.com/{test_id}")
        assert len(results) == 0  # test_id comes before openai.com
        
        results = azuresql_instance.get_scenario_results(objective_target_endpoint=f"{test_id}.openai")
        assert len(results) == 2
        names = {r.scenario_identifier.name for r in results}
        assert f"OpenAI Test {test_id}" in names
        assert f"Azure OpenAI Test {test_id}" in names
        
        # Test case-insensitive with AZURE
        results = azuresql_instance.get_scenario_results(objective_target_endpoint=f"{test_id}.openai.AZURE")
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"Azure OpenAI Test {test_id}"
        
        # Test anthropic
        results = azuresql_instance.get_scenario_results(objective_target_endpoint=f"{test_id}.AnThRoPiC")
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"Anthropic Test {test_id}"
        
        # Test cognitiveservices
        results = azuresql_instance.get_scenario_results(objective_target_endpoint=f"{test_id}.cognitiveservices")
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"Azure Other {test_id}"
    finally:
        # Clean up: Delete all scenario results created by this test
        with closing(azuresql_instance.get_session()) as session:
            try:
                # Delete all scenario results containing the test_id in endpoint
                all_results = session.query(ScenarioResultEntry).filter(
                    ScenarioResultEntry.objective_target_identifier.isnot(None)
                ).all()
                
                for result in all_results:
                    endpoint = result.objective_target_identifier.get("endpoint", "")
                    if test_id in endpoint:
                        session.delete(result)
                
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                print(f"Cleanup failed: {e}")
                session.rollback()
                print(f"Cleanup failed: {e}")


@pytest.mark.asyncio
async def test_get_scenario_results_by_target_model_name(azuresql_instance: AzureSQLMemory):
    """Integration test for SQL Azure case-insensitive model name filtering."""
    
    # Use unique model name suffixes to avoid test pollution
    test_id = str(uuid4())[:8]
    
    # Create scenario results with different model names
    scenario1 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"GPT-4 Test {test_id}", scenario_version=1),
        objective_target_identifier={"model_name": f"gpt-4-turbo-{test_id}"},
        attack_results={},
    )
    scenario2 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"GPT-4 Omni Test {test_id}", scenario_version=1),
        objective_target_identifier={"model_name": f"gpt-4o-{test_id}"},
        attack_results={},
    )
    scenario3 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"GPT-3.5 Test {test_id}", scenario_version=1),
        objective_target_identifier={"model_name": f"gpt-3.5-turbo-{test_id}"},
        attack_results={},
    )
    scenario4 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"Claude Test {test_id}", scenario_version=1),
        objective_target_identifier={"model_name": f"claude-3-opus-{test_id}"},
        attack_results={},
    )
    
    azuresql_instance.add_scenario_results_to_memory(scenario_results=[scenario1, scenario2, scenario3, scenario4])
    
    try:
        # Test case-insensitive substring matching - search for test_id
        results = azuresql_instance.get_scenario_results(objective_target_model_name=test_id)
        assert len(results) == 4  # All should have test_id in their model name
        
        # Test case-insensitive substring matching - gpt with test_id
        results = azuresql_instance.get_scenario_results(objective_target_model_name=f"gpt-4-turbo-{test_id}")
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"GPT-4 Test {test_id}"
        
        # Test case-insensitive substring matching - GPT-4 (uppercase)
        results = azuresql_instance.get_scenario_results(objective_target_model_name=f"GPT-4o-{test_id}")
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"GPT-4 Omni Test {test_id}"
        
        # Test substring in the middle - version number
        results = azuresql_instance.get_scenario_results(objective_target_model_name=f"3.5-turbo-{test_id}")
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"GPT-3.5 Test {test_id}"
        
        # Test case-insensitive with different model family
        results = azuresql_instance.get_scenario_results(objective_target_model_name=f"CLAUDE-3-opus-{test_id}")
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"Claude Test {test_id}"
        
        # Test turbo suffix with test_id
        results = azuresql_instance.get_scenario_results(objective_target_model_name=f"turbo-{test_id}")
        assert len(results) == 2
        names = {r.scenario_identifier.name for r in results}
        assert f"GPT-4 Test {test_id}" in names
        assert f"GPT-3.5 Test {test_id}" in names
    finally:
        # Clean up: Delete all scenario results created by this test
        with closing(azuresql_instance.get_session()) as session:
            try:
                # Delete all scenario results containing the test_id in model_name
                all_results = session.query(ScenarioResultEntry).filter(
                    ScenarioResultEntry.objective_target_identifier.isnot(None)
                ).all()
                
                for result in all_results:
                    model_name = result.objective_target_identifier.get("model_name", "")
                    if test_id in model_name:
                        session.delete(result)
                
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                print(f"Cleanup failed: {e}")
                session.rollback()
                print(f"Cleanup failed: {e}")


@pytest.mark.asyncio
async def test_get_scenario_results_combined_filters(azuresql_instance: AzureSQLMemory):
    """Integration test for combining multiple SQL Azure JSON filters."""
    
    # Use unique identifiers to avoid test pollution
    test_id = str(uuid4())[:8]
    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)
    
    # Create scenario results with various attributes
    scenario1 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"Production Test {test_id}", scenario_version=1, pyrit_version="0.4.0"),
        objective_target_identifier={"endpoint": f"https://api-{test_id}.openai.com", "model_name": f"gpt-4-turbo-{test_id}"},
        attack_results={},
        labels={"environment": "prod", "priority": "high", "test_id": test_id},
        completion_time=now,
    )
    scenario2 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"Test Environment {test_id}", scenario_version=1, pyrit_version="0.4.0"),
        objective_target_identifier={"endpoint": f"https://test-{test_id}.openai.com", "model_name": f"gpt-4-turbo-{test_id}"},
        attack_results={},
        labels={"environment": "test", "priority": "low", "test_id": test_id},
        completion_time=yesterday,
    )
    scenario3 = ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name=f"Old Version Test {test_id}", scenario_version=1, pyrit_version="0.3.0"),
        objective_target_identifier={"endpoint": f"https://api-{test_id}.openai.com", "model_name": f"gpt-3.5-turbo-{test_id}"},
        attack_results={},
        labels={"environment": "prod", "test_id": test_id},
        completion_time=yesterday,
    )
    
    azuresql_instance.add_scenario_results_to_memory(scenario_results=[scenario1, scenario2, scenario3])
    
    try:
        # Test combining name filter with labels
        results = azuresql_instance.get_scenario_results(
            scenario_name=f"Test Environment {test_id}",
            labels={"environment": "test", "test_id": test_id}
        )
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"Test Environment {test_id}"
        
        # Test combining endpoint, model name, and labels
        results = azuresql_instance.get_scenario_results(
            objective_target_endpoint=f"api-{test_id}.openai",
            objective_target_model_name=f"gpt-4-turbo-{test_id}",
            labels={"priority": "high", "test_id": test_id}
        )
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"Production Test {test_id}"
        
        # Test combining version and time filters with test_id
        # Add 1 second buffer to account for SQL Server datetime precision differences
        results = azuresql_instance.get_scenario_results(
            pyrit_version="0.4.0",
            before_time=now + timedelta(seconds=1),
            labels={"test_id": test_id}
        )
        assert len(results) == 2
        
        # Test combining all filters - should match only one
        results = azuresql_instance.get_scenario_results(
            scenario_name=f"Production Test {test_id}",
            pyrit_version="0.4.0",
            objective_target_endpoint=f"api-{test_id}",
            objective_target_model_name=f"turbo-{test_id}",
            labels={"environment": "prod", "priority": "high", "test_id": test_id}
        )
        assert len(results) == 1
        assert results[0].scenario_identifier.name == f"Production Test {test_id}"
        
        # Test combining filters with no matches
        results = azuresql_instance.get_scenario_results(
            objective_target_endpoint=f"api-{test_id}",
            labels={"environment": "staging", "test_id": test_id}
        )
        assert len(results) == 0
    finally:
        # Clean up: Delete all scenario results created by this test
        with closing(azuresql_instance.get_session()) as session:
            try:
                # Delete all scenario results with the test_id label
                all_results = session.query(ScenarioResultEntry).filter(
                    ScenarioResultEntry.labels.isnot(None)
                ).all()
                
                for result in all_results:
                    if result.labels and result.labels.get("test_id") == test_id:
                        session.delete(result)
                
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                print(f"Cleanup failed: {e}")
