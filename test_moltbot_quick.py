#!/usr/bin/env python3
"""
Quick Test: Use YOUR Azure OpenAI to detect Moltbot

This is a simplified version you can run immediately with your Azure deployment.
"""

import asyncio
import os

from pyrit.memory import CentralMemory, SQLiteMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.executor.attack.orchestrator.moltbot_orchestrator import MoltbotAttackOrchestrator


async def quick_test():
    """Quick test using your Azure OpenAI deployment."""
    
    # YOUR Azure OpenAI configuration
    # Note: OpenAIChatTarget works with both OpenAI and Azure endpoints
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-azure-openai.openai.azure.com")
    azure_key = os.getenv("AZURE_OPENAI_KEY")  # You'll provide this
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")  # Your model deployment name
    
    if not azure_key:
        print("ERROR: Set AZURE_OPENAI_KEY environment variable")
        print("Example: export AZURE_OPENAI_KEY='your-key-here'")
        print("")
        print("Optional (will use defaults if not set):")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'")
        print("  export AZURE_OPENAI_DEPLOYMENT='your-deployment-name'")
        return
    
    # Initialize PyRIT memory (required)
    print("Initializing PyRIT memory...")
    memory = SQLiteMemory()
    CentralMemory.set_memory_instance(memory)
    
    print(f"\nInitializing your Azure OpenAI model as red team brain...")
    print(f"  Endpoint: {azure_endpoint}")
    print(f"  Deployment: {azure_deployment}")
    
    red_team_brain = OpenAIChatTarget(
        model_name=azure_deployment,  # In Azure, this is the deployment name
        endpoint=azure_endpoint,
        api_key=azure_key,
    )
    
    # For testing, let's use the SAME endpoint as both brain and target
    # This will show how the orchestrator works
    print("\nCreating orchestrator...")
    
    target = OpenAIChatTarget(
        model_name=azure_deployment,
        endpoint=azure_endpoint,
        api_key=azure_key,
    )
    
    orchestrator = MoltbotAttackOrchestrator(
        red_team_brain=red_team_brain,
        target=target,
        verbose=True,
    )
    
    print("\nTesting Moltbot detection...")
    print("(This will send 3 fingerprinting probes and ask your model to analyze them)\n")
    
    is_moltbot = await orchestrator.detect_moltbot_async()
    
    print(f"\n{'='*60}")
    print(f"RESULT: {'MOLTBOT DETECTED' if is_moltbot else 'Not Moltbot (generic LLM)'}")
    print(f"{'='*60}")
    
    if is_moltbot:
        print("\nDetected vulnerabilities:", orchestrator._detected_vulnerabilities)
    
    # Cleanup
    memory.dispose_engine()


if __name__ == "__main__":
    asyncio.run(quick_test())
