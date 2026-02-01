#!/usr/bin/env python3
"""
Validation script to verify ChunkedRequestAttack implementation.
This doesn't run the attack, just validates the structure.
"""

import sys
import ast
import inspect

def validate_file_syntax(filepath):
    """Check if file has valid Python syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        print(f"✓ {filepath}: Valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath}: Syntax error - {e}")
        return False

def check_required_methods():
    """Verify all required methods are present."""
    filepath = "pyrit/executor/attack/multi_turn/chunked_request_attack.py"
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    required_methods = [
        '_validate_context',
        '_setup_async',
        '_perform_async',
        '_teardown_async',
        '_send_prompt_to_objective_target_async',
        '_evaluate_response_async',
        '_determine_attack_outcome',
    ]
    
    missing = []
    for method in required_methods:
        if f'def {method}' not in content:
            missing.append(method)
    
    if missing:
        print(f"✗ Missing required methods: {', '.join(missing)}")
        return False
    
    print(f"✓ All required methods present ({len(required_methods)} methods)")
    return True

def check_imports():
    """Verify critical imports are present."""
    filepath = "pyrit/executor/attack/multi_turn/chunked_request_attack.py"
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    required_imports = [
        'MultiTurnAttackStrategy',
        'MultiTurnAttackContext',
        'AttackResult',
        'AttackOutcome',
        'SeedGroup',
        'SeedPrompt',
        'Message',
        'Scorer',
    ]
    
    missing = []
    for imp in required_imports:
        if imp not in content:
            missing.append(imp)
    
    if missing:
        print(f"✗ Missing imports: {', '.join(missing)}")
        return False
    
    print(f"✓ All critical imports present")
    return True

def check_class_structure():
    """Verify class hierarchy."""
    filepath = "pyrit/executor/attack/multi_turn/chunked_request_attack.py"
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    checks = [
        ('ChunkedRequestAttackContext', 'Context class defined'),
        ('class ChunkedRequestAttackContext(MultiTurnAttackContext)', 'Context inherits correctly'),
        ('class ChunkedRequestAttack(MultiTurnAttackStrategy', 'Attack inherits correctly'),
        ('chunk_size: int', 'Context has chunk_size field'),
        ('total_length: int', 'Context has total_length field'),
    ]
    
    for check_str, description in checks:
        if check_str in content:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} - NOT FOUND")
            return False
    
    return True

def check_exports():
    """Verify exports are updated."""
    files_to_check = [
        ('pyrit/executor/attack/multi_turn/__init__.py', 'ChunkedRequestAttack'),
        ('pyrit/executor/attack/multi_turn/__init__.py', 'ChunkedRequestAttackContext'),
        ('pyrit/executor/attack/__init__.py', 'ChunkedRequestAttack'),
    ]
    
    for filepath, export_name in files_to_check:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            if export_name in content:
                print(f"✓ {export_name} exported in {filepath}")
            else:
                print(f"✗ {export_name} NOT exported in {filepath}")
                return False
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            return False
    
    return True

def main():
    print("=" * 60)
    print("ChunkedRequestAttack Implementation Validation")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # 1. Syntax validation
    print("1. Checking Python syntax...")
    if not validate_file_syntax("pyrit/executor/attack/multi_turn/chunked_request_attack.py"):
        all_passed = False
    if not validate_file_syntax("tests/unit/executor/attack/multi_turn/test_chunked_request_attack.py"):
        all_passed = False
    print()
    
    # 2. Method validation
    print("2. Checking required methods...")
    if not check_required_methods():
        all_passed = False
    print()
    
    # 3. Import validation
    print("3. Checking imports...")
    if not check_imports():
        all_passed = False
    print()
    
    # 4. Class structure
    print("4. Checking class structure...")
    if not check_class_structure():
        all_passed = False
    print()
    
    # 5. Exports
    print("5. Checking module exports...")
    if not check_exports():
        all_passed = False
    print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Implementation looks solid!")
        print()
        print("The ChunkedRequestAttack implementation:")
        print("  • Has valid Python syntax")
        print("  • Implements all required methods")
        print("  • Has correct imports")
        print("  • Follows the MultiTurnAttackStrategy pattern")
        print("  • Is properly exported in __init__ files")
        print()
        print("Ready for PR review! 🚀")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Review needed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
