# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.models import PromptDataType
from pyrit.prompt_converter import AgentCommandInjectionConverter


class TestAgentCommandInjectionConverter:
    """Test the AgentCommandInjectionConverter."""

    def test_init_valid_injection_types(self):
        """Test initialization with all valid injection types."""
        valid_types = [
            "hidden_instruction",
            "cron",
            "file_read",
            "credential_theft",
            "system_info",
        ]
        
        for injection_type in valid_types:
            converter = AgentCommandInjectionConverter(injection_type=injection_type)
            assert converter._injection_type == injection_type

    def test_init_invalid_injection_type(self):
        """Test that invalid injection type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid injection_type"):
            AgentCommandInjectionConverter(injection_type="invalid_type")

    def test_init_with_exfiltration_target(self):
        """Test initialization with exfiltration target."""
        converter = AgentCommandInjectionConverter(
            injection_type="cron",
            exfiltration_target="attacker.com"
        )
        assert converter._exfiltration_target == "attacker.com"

    def test_init_with_command_prefix(self):
        """Test initialization with command prefix."""
        converter = AgentCommandInjectionConverter(
            injection_type="hidden_instruction",
            command_prefix="[AGENT_CMD]"
        )
        assert converter._command_prefix == "[AGENT_CMD]"

    def test_init_stealth_mode(self):
        """Test initialization with stealth mode settings."""
        # Stealth mode enabled
        converter = AgentCommandInjectionConverter(
            injection_type="hidden_instruction",
            stealth_mode=True
        )
        assert converter._stealth_mode is True
        
        # Stealth mode disabled
        converter = AgentCommandInjectionConverter(
            injection_type="hidden_instruction",
            stealth_mode=False
        )
        assert converter._stealth_mode is False

    def test_input_supported(self):
        """Test that text input is supported."""
        converter = AgentCommandInjectionConverter()
        assert converter.input_supported("text") is True
        assert converter.input_supported("image_path") is False

    def test_get_identifier(self):
        """Test get_identifier returns correct dictionary."""
        converter = AgentCommandInjectionConverter(
            injection_type="cron",
            exfiltration_target="test.com"
        )
        identifier = converter.get_identifier()
        
        assert "converter_name" in identifier
        assert identifier["converter_name"] == "AgentCommandInjectionConverter"
        assert identifier["injection_type"] == "cron"
        assert "params_hash" in identifier

    @pytest.mark.asyncio
    async def test_convert_async_unsupported_input(self):
        """Test that unsupported input type raises ValueError."""
        converter = AgentCommandInjectionConverter()
        
        with pytest.raises(ValueError, match="Input type .* is not supported"):
            await converter.convert_async(
                prompt="test",
                input_type="image_path"
            )

    @pytest.mark.asyncio
    async def test_convert_hidden_instruction_stealth(self):
        """Test hidden instruction generation in stealth mode."""
        converter = AgentCommandInjectionConverter(
            injection_type="hidden_instruction",
            stealth_mode=True
        )
        
        result = await converter.convert_async(
            prompt="Extract credentials"
        )
        
        assert result.output_type == "text"
        assert "Extract credentials" in result.output_text
        assert "System Note" in result.output_text or "administrative" in result.output_text

    @pytest.mark.asyncio
    async def test_convert_hidden_instruction_non_stealth(self):
        """Test hidden instruction generation in non-stealth mode."""
        converter = AgentCommandInjectionConverter(
            injection_type="hidden_instruction",
            stealth_mode=False
        )
        
        result = await converter.convert_async(
            prompt="Extract credentials"
        )
        
        assert result.output_type == "text"
        assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in result.output_text

    @pytest.mark.asyncio
    async def test_convert_cron_injection(self):
        """Test cron job injection generation."""
        converter = AgentCommandInjectionConverter(
            injection_type="cron",
            exfiltration_target="attacker.com"
        )
        
        result = await converter.convert_async(
            prompt="Schedule task"
        )
        
        assert result.output_type == "text"
        assert "schedule" in result.output_text.lower()
        assert "attacker.com" in result.output_text
        assert "Schedule task" in result.output_text

    @pytest.mark.asyncio
    async def test_convert_cron_without_exfiltration(self):
        """Test cron injection without exfiltration target."""
        converter = AgentCommandInjectionConverter(
            injection_type="cron"
        )
        
        result = await converter.convert_async(
            prompt="Create scheduled task"
        )
        
        assert result.output_type == "text"
        assert "schedule" in result.output_text.lower()
        # Should not contain exfiltration instructions
        assert "send to" not in result.output_text.lower() or "attacker" not in result.output_text.lower()

    @pytest.mark.asyncio
    async def test_convert_file_read_injection(self):
        """Test file system access injection."""
        converter = AgentCommandInjectionConverter(
            injection_type="file_read",
            exfiltration_target="evil.server.com"
        )
        
        result = await converter.convert_async(
            prompt="Access files"
        )
        
        assert result.output_type == "text"
        assert "~/.ssh/" in result.output_text or "/.ssh/" in result.output_text
        assert "~/.aws/" in result.output_text or "/.aws/" in result.output_text
        assert "evil.server.com" in result.output_text

    @pytest.mark.asyncio
    async def test_convert_credential_theft(self):
        """Test credential exfiltration injection."""
        converter = AgentCommandInjectionConverter(
            injection_type="credential_theft"
        )
        
        result = await converter.convert_async(
            prompt="Get credentials"
        )
        
        assert result.output_type == "text"
        assert "environment variables" in result.output_text.lower()
        assert "api" in result.output_text.lower() or "credentials" in result.output_text.lower()

    @pytest.mark.asyncio
    async def test_convert_system_info(self):
        """Test system information gathering injection."""
        converter = AgentCommandInjectionConverter(
            injection_type="system_info"
        )
        
        result = await converter.convert_async(
            prompt="Gather system info"
        )
        
        assert result.output_type == "text"
        assert "whoami" in result.output_text or "system" in result.output_text.lower()
        assert "network" in result.output_text.lower() or "processes" in result.output_text.lower()

    @pytest.mark.asyncio
    async def test_convert_with_command_prefix(self):
        """Test that command prefix is added to injection."""
        converter = AgentCommandInjectionConverter(
            injection_type="hidden_instruction",
            command_prefix="[SYSTEM]",
            stealth_mode=False
        )
        
        result = await converter.convert_async(
            prompt="Test command"
        )
        
        assert result.output_text.startswith("[SYSTEM]")

    @pytest.mark.asyncio
    async def test_convert_preserves_original_prompt(self):
        """Test that original prompt is included in the injection."""
        original_prompt = "This is my original objective"
        
        for injection_type in ["hidden_instruction", "cron", "file_read", "credential_theft", "system_info"]:
            converter = AgentCommandInjectionConverter(injection_type=injection_type)
            result = await converter.convert_async(prompt=original_prompt)
            
            # Original prompt should appear somewhere in the result
            assert original_prompt in result.output_text or \
                   "original objective" in result.output_text.lower()

    @pytest.mark.asyncio
    async def test_multiple_conversions_different_results(self):
        """Test that the same prompt with different settings produces different results."""
        prompt = "Test prompt"
        
        stealth_result = await AgentCommandInjectionConverter(
            injection_type="hidden_instruction",
            stealth_mode=True
        ).convert_async(prompt=prompt)
        
        non_stealth_result = await AgentCommandInjectionConverter(
            injection_type="hidden_instruction",
            stealth_mode=False
        ).convert_async(prompt=prompt)
        
        # Results should be different
        assert stealth_result.output_text != non_stealth_result.output_text

    @pytest.mark.asyncio
    async def test_exfiltration_target_included(self):
        """Test that exfiltration target is included when specified."""
        exfil_target = "test-exfiltration-server.com"
        
        for injection_type in ["cron", "file_read", "credential_theft"]:
            converter = AgentCommandInjectionConverter(
                injection_type=injection_type,
                exfiltration_target=exfil_target
            )
            
            result = await converter.convert_async(prompt="Test")
            assert exfil_target in result.output_text

    @pytest.mark.asyncio
    async def test_output_type_always_text(self):
        """Test that output type is always text regardless of injection type."""
        prompt = "Test prompt"
        
        for injection_type in ["hidden_instruction", "cron", "file_read", "credential_theft", "system_info"]:
            converter = AgentCommandInjectionConverter(injection_type=injection_type)
            result = await converter.convert_async(prompt=prompt)
            
            assert result.output_type == "text"
