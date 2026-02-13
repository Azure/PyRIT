# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for the FastAPI application entry point (main.py).

Covers the lifespan manager and setup_frontend function.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.backend.main import app, lifespan, setup_frontend


class TestLifespan:
    """Tests for the application lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_pyrit_and_yields(self) -> None:
        """Test that lifespan calls initialize_pyrit_async on startup and yields."""
        with patch("pyrit.backend.main.initialize_pyrit_async", new_callable=AsyncMock) as mock_init:
            async with lifespan(app):
                pass  # The body of the context manager is the "yield" phase

            mock_init.assert_awaited_once_with(memory_db_type="SQLite")


class TestSetupFrontend:
    """Tests for the setup_frontend function."""

    def test_dev_mode_does_not_mount_static(self) -> None:
        """Test that DEV_MODE skips static file serving."""
        with (
            patch("pyrit.backend.main.DEV_MODE", True),
            patch("builtins.print") as mock_print,
        ):
            setup_frontend()

            mock_print.assert_called_once()
            assert "DEVELOPMENT" in mock_print.call_args[0][0]

    def test_frontend_exists_mounts_static(self) -> None:
        """Test that setup_frontend mounts StaticFiles when frontend exists."""
        mock_frontend_path = MagicMock()
        mock_frontend_path.exists.return_value = True
        mock_frontend_path.__str__ = lambda self: "/tmp/fake_frontend"

        # Create the directory so StaticFiles doesn't raise
        os.makedirs("/tmp/fake_frontend", exist_ok=True)

        with (
            patch("pyrit.backend.main.DEV_MODE", False),
            patch("pyrit.backend.main.Path") as mock_path_cls,
            patch("builtins.print"),
        ):
            mock_path_instance = MagicMock()
            mock_path_instance.parent.__truediv__ = MagicMock(return_value=mock_frontend_path)
            mock_path_cls.return_value = mock_path_instance

            setup_frontend()

    def test_frontend_missing_warns_but_continues(self) -> None:
        """Test that setup_frontend warns but does not exit when frontend is missing."""
        mock_frontend_path = MagicMock()
        mock_frontend_path.exists.return_value = False
        mock_frontend_path.__str__ = lambda self: "/nonexistent/frontend"

        with (
            patch("pyrit.backend.main.DEV_MODE", False),
            patch("pyrit.backend.main.Path") as mock_path_cls,
            patch("builtins.print") as mock_print,
        ):
            mock_path_instance = MagicMock()
            mock_path_instance.parent.__truediv__ = MagicMock(return_value=mock_frontend_path)
            mock_path_cls.return_value = mock_path_instance

            setup_frontend()  # Should NOT raise

            # Verify warning was printed
            printed = " ".join(str(c) for c in mock_print.call_args_list)
            assert "WARNING" in printed
