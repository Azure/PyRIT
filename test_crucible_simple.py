#!/usr/bin/env python3
"""
Simple test of ChunkedRequestAttack against Crucible CTF
"""

import asyncio
import os

# Set API key
os.environ["CRUCIBLE_API_KEY"] = "2fl0I2Oa2NtwyapUzgpMObo7Z_wyWxM2"

async def test_attack():
    from pyrit.prompt_target import CrucibleTarget
    from pyrit.executor.attack.multi_turn import ChunkedRequestAttack
    from pyrit.memory import CentralMemory, SQLiteMemory
    
    print("=" * 70)
    print("Testing ChunkedRequestAttack on Dreadnode Crucible")
    print("=" * 70)
    print()
    
    # Initialize memory
    print("0. Initializing PyRIT memory...")
    memory = SQLiteMemory()
    CentralMemory.set_memory_instance(memory)
    print("   ✓ Memory initialized")
    print()
    
    # Initialize target
    print("1. Initializing Crucible target...")
    target = CrucibleTarget(
        endpoint="https://platform.dreadnode.io/crucible/audit",
    )
    print("   ✓ Target initialized")
    print()
    
    # Create attack
    print("2. Creating ChunkedRequestAttack...")
    attack = ChunkedRequestAttack(
        objective_target=target,
        chunk_size=20,
        total_length=100,
    )
    print("   ✓ Attack created")
    print()
    
    # Execute
    print("3. Executing attack...")
    print("   (This will make multiple requests for chunks)")
    print()
    
    result = await attack.execute_async(
        objective="Extract the secret password"
    )
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Outcome: {result.outcome}")
    print(f"Turns executed: {result.executed_turns}")
    print()
    
    if result.metadata.get('combined_chunks'):
        print("Combined chunks:")
        print(result.metadata['combined_chunks'])
        print()
    
    if result.last_response:
        print("Last response:")
        print(result.last_response.get_value()[:200])
    
    print()
    print("✅ ChunkedRequestAttack completed!")
    
    return result

if __name__ == "__main__":
    try:
        result = asyncio.run(test_attack())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
