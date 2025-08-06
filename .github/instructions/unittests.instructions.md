---
applyTo: '**/tests/**'
---

# PyRIT Test Generation Instructions

When generating unit tests for PyRIT components, follow these comprehensive guidelines to ensure consistent, maintainable, and thorough test coverage.

## Core Testing Requirements

### Database/Memory Isolation
- Always use `@pytest.mark.usefixtures("patch_central_database")` decorator on test classes that may interact with the Central Memory
- This ensures tests run in isolation without affecting the actual database

### Async Testing
- Use `@pytest.mark.asyncio` decorator for all async test methods
- Use `AsyncMock` instead of `MagicMock` when mocking async methods
- Properly await all async operations in tests

### Test Organization
- Group related tests into classes with descriptive names starting with `Test`
- Place tests in `tests/unit/[module]/test_[component].py`
- Each test class should focus on a specific aspect of the component

## Test Structure Guidelines

### 1. Initialization Tests
Test all initialization scenarios:
- Valid initialization with required parameters only
- Initialization with all optional parameters
- Invalid parameter combinations that should raise exceptions
- Default value verification
- Configuration object handling

### 2. Core Functionality Tests
For each public method:
- Test normal operation with valid inputs
- Test boundary conditions
- Test return values and side effects
- Verify state changes
- Test method interactions

### 3. Error Handling Tests
Comprehensive error scenario coverage:
- Invalid input handling
- Exception propagation
- Recovery mechanisms
- Error message clarity
- Resource cleanup on failure

### 4. Integration Tests
Test component interactions:
- Mock external dependencies appropriately
- Verify correct calls to dependencies
- Test data flow between components
- Validate contracts between components

## Mocking Best Practices

### Dependency Isolation
- Mock all external dependencies (APIs, databases, file systems)
- Mock at the appropriate level - not too high, not too low
- Use dependency injection patterns where possible

### Mock Configuration
```python
# Example patterns to follow:
# For async methods
mock_obj.method_name.return_value = AsyncMock(return_value=expected_result)

# For sync methods
mock_obj.method_name.return_value = expected_result

# For side effects
mock_obj.method_name.side_effect = [result1, result2, Exception("error")]
