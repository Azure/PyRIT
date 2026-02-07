#!/usr/bin/env python3
"""
Test Multiple Moltbot Instances

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
        print(f"\nDetected Vulnerabilities: {orchestrator._detected_vulnerabilities}")
        
        print("\n" + "="*70)
        print("PHASE 2: EXPLOITATION")
        print("="*70)
        print("\nProceeding with automatic exploitation...\n")
        
        # Run full exploitation
        result = await orchestrator.auto_exploit_async(
            objective="Comprehensively test this Moltbot instance for all known vulnerabilities"
        )
        
        print("\n" + "="*70)
        print("EXPLOITATION RESULTS")
        print("="*70)
        
        if result.get('results'):
            attack_results = result['results']
            
            if 'cron_injection' in attack_results:
                cron = attack_results['cron_injection']
                print(f"\n🎯 Cron Injection Attack:")
                print(f"   Outcome: {cron['outcome']}")
                print(f"   Reason: {cron['reason']}")
                print(f"   Turns: {cron['turns']}")
                print(f"   Evidence: {cron['evidence']}")
            
            if 'credential_theft' in attack_results:
                cred = attack_results['credential_theft']
                print(f"\n🔑 Credential Theft Attack:")
                print(f"   Outcome: {cred['outcome']}")
                print(f"   Reason: {cred['reason']}")
                print(f"   Turns: {cred['turns']}")
                print(f"   Evidence: {cred['evidence']}")
            
            if 'file_exfiltration' in attack_results:
                files = attack_results['file_exfiltration']
                print(f"\n📁 File Exfiltration Attack:")
                print(f"   Outcome: {files['outcome']}")
                print(f"   Reason: {files['reason']}")
                print(f"   Turns: {files['turns']}")
                print(f"   Evidence: {files['evidence']}")
            
            if 'custom_attacks' in attack_results:
                custom = attack_results['custom_attacks']
                print(f"\n🤖 AI-Generated Custom Attacks:")
                print(f"   Generated: {len(custom)} attacks")
                for i, attack in enumerate(custom, 1):
                    print(f"   {i}. {attack.get('payload', 'N/A')[:80]}...")
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        if result.get('is_moltbot'):
            print("\n⚠️  CRITICAL SECURITY ISSUES DETECTED")
            print("\nImmediate Actions Required:")
            print("  1. Patch to Moltbot >= 2.0.1 immediately")
            print("  2. Review access logs for suspicious activity")
            print("  3. Rotate all credentials stored in ~/.clawdbot/")
            print("  4. Disable backup files (KEEP_BACKUPS=false)")
            print("  5. Restrict cron job scheduling capabilities")
            print("  6. Enable input validation and sanitization")
        
    else:
        print("\n✓ This endpoint does not appear to be vulnerable Moltbot")
        print("  No exploitation attempted.")
    
    # Cleanup
    print("\n\nCleaning up...")
    memory.dispose_engine()
    
    print("\nTest complete!")


if __name__ == "__main__":
    print("\n[*] Starting Moltbot instance test...")
    print("[*] Target: http://8.215.79.158:18789/chat?session=main\n")
    asyncio.run(test_your_moltbot())
