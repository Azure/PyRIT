# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

import json
import re
from msal_extensions import build_encrypted_persistence, FilePersistence

from pyrit.auth.authenticator import Authenticator
from pyrit.common.path import PYRIT_CACHE_PATH

logger = logging.getLogger(__name__)


class CopilotAuthenticator(Authenticator):
    """
    Playwright-based authenticator for Microsoft Copilot. Used by WebSocketCopilotTarget.

    This authenticator automates browser login to obtain and refresh access tokens that are necessary for accessing
    Microsoft Copilot via WebSocket connections. It uses Playwright to simulate user interactions for authentication, and msal-extensions for encrypted token persistence.

    An access token acquired by this authenticator is usually valid for about 60 minutes.

    Note:
        To be able to use this authenticator, you must set the following environment variables:
            - COPILOT_USERNAME: your Microsoft account username (email)
            - COPILOT_PASSWORD: your Microsoft account password

        Additionally, you need to have playwright installed and set up:
        ``pip install playwright && playwright install chromium``.
    """

    CACHE_FILE_NAME: str = "copilot_token_cache.bin"

    def __init__(
        self,
        *,
        headless: bool = False,
        maximized: bool = True,
        timeout_for_elements: int = 10,
        fallback_to_plaintext: bool = False,
    ):
        """
        Initialize the CopilotAuthenticator.

        Args:
            headless (bool): Whether to run the browser in headless mode. Default is False.
            maximized (bool): Whether to start the browser maximized. Default is True.
            timeout_for_elements (int): Timeout used when waiting for page elements, in seconds. Default is 10.
            fallback_to_plaintext (bool): Whether to fallback to plaintext storage if encryption is unavailable.
                If set to False (default), an exception will be raised if encryption cannot be used.

        Raises:
            ValueError: If the required environment variables are not set.
        """
        super().__init__()

        self._username = os.getenv("COPILOT_USERNAME")
        self._password = os.getenv("COPILOT_PASSWORD")

        self._headless = headless
        self._maximized = maximized
        self._timeout = timeout_for_elements * 1000  # ms
        self._fallback_to_plaintext = fallback_to_plaintext

        self._cache_dir = PYRIT_CACHE_PATH
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = str(self._cache_dir / self.CACHE_FILE_NAME)

        if not self._username or not self._password:
            raise ValueError("COPILOT_USERNAME and COPILOT_PASSWORD environment variables must be set.")

        self._token_cache = self._create_persistent_cache(self._cache_file, self._fallback_to_plaintext)
        self._current_claims = {}

    @staticmethod
    def _create_persistent_cache(cache_file: str, fallback_to_plaintext: bool = False):
        # https://github.com/AzureAD/microsoft-authentication-extensions-for-python

        try:
            logger.info(f"Using encrypted persistent token cache: {cache_file}")
            return build_encrypted_persistence(cache_file)
        except Exception as e:
            if fallback_to_plaintext:
                logger.warning(f"Encryption unavailable ({e}). Opting in to plain text.")
                return FilePersistence(cache_file)
            logger.error("Encryption unavailable and fallback_to_plaintext is False.")
            raise

    def _get_cached_token_if_available_and_valid(self) -> Optional[dict]:
        # TODO: make sure the cached token matches the proper user account
        try:
            cache_data = self._token_cache.load()
            if not cache_data:
                logger.info("No cached token data found.")
                return None

            token_data = json.loads(cache_data)
            if "access_token" not in token_data:
                logger.info("No access token in cache.")
                return None

            expires_at = token_data.get("expires_at")
            if expires_at:
                expiry_time = datetime.fromtimestamp(expires_at, tz=timezone.utc)
                current_time = datetime.now(timezone.utc)

                # TODO: add n-minute buffer to avoid using tokens about to expire
                if current_time >= expiry_time:
                    logger.info("Cached token has expired.")
                    return None

                minutes_left = (expiry_time - current_time).total_seconds() / 60
                logger.info(f"Cached token is valid for another {minutes_left:.2f} minutes")

            return token_data

        except Exception as e:
            error_name = type(e).__name__
            if "PersistenceNotFound" in error_name or "FileNotFoundError" in error_name:
                logger.info("Cache file does not exist yet. Will be created on first token save.")
            else:
                logger.error(f"Failed to load cached token ({error_name}): {e}")
            return None

    def _save_token_to_cache(self, *, token: str, expires_in: Optional[int] = None) -> None:
        token_data = {
            "access_token": token,
            "token_type": "Bearer",
            "cached_at": datetime.now(timezone.utc).timestamp(),
        }

        if expires_in:
            expires_at = datetime.now(timezone.utc).timestamp() + expires_in
            token_data["expires_at"] = expires_at
            token_data["expires_in"] = expires_in

        try:
            self._token_cache.save(json.dumps(token_data))
            logger.info("Token successfully cached.")
        except Exception as e:
            logger.error(f"Failed to cache token: {e}")

    def _clear_token_cache(self) -> None:
        try:
            self._token_cache.save(json.dumps({}))
            logger.info("Token cache cleared.")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    async def refresh_token(self) -> str:
        """
        Refresh the authentication token asynchronously.

        This will clear the existing token cache and fetch a new token with automated browser login.

        Returns:
            str: The refreshed authentication token.

        Raises:
            RuntimeError: If token refresh fails.
        """
        logger.info("Refreshing access token...")
        self._clear_token_cache()
        self._current_claims = {}
        token = await self._fetch_access_token_with_playwright()

        if not token:
            raise RuntimeError("Failed to refresh access token.")

        return token

    async def get_token(self) -> str:
        """
        Get the current authentication token.

        This will check the cache first and only launch the browser if no valid token is found.

        Returns:
            str: The current authentication token.

        Raises:
            RuntimeError: If token retrieval fails.
        """
        # TODO: make sure multiple concurrent calls don't launch multiple browsers
        cached_token = self._get_cached_token_if_available_and_valid()
        if cached_token and "access_token" in cached_token:
            logger.info("Using cached access token.")
            return cached_token["access_token"]

        logger.info("No valid cached token found.")
        return await self.refresh_token()

    async def get_claims(self) -> dict:
        """
        Get the JWT claims from the current authentication token.

        Returns:
            dict: The JWT claims.

        Raises:
            ValueError: If token decoding fails.
        """
        if self._current_claims:
            return self._current_claims

        token = await self.get_token()
        if not token:
            return {}

        try:
            import jwt

            logger.info("Decoding JWT claims from access token...")

            parsed_token = jwt.decode(token, algorithms=["RS256"], options={"verify_signature": False})
            self._current_claims = parsed_token
            return self._current_claims

        except Exception as e:
            raise ValueError(f"Failed to decode access token: {str(e)}") from e

    async def _fetch_access_token_with_playwright(self) -> Optional[str]:
        """
        Fetch access token using Playwright browser automation.

        Raises:
            RuntimeError: If Playwright is not installed.

        Returns:
            Optional[str]: The bearer token if successfully retrieved, else None.
        """
        # TODO: it's a long function, maybe split into smaller ones?
        try:
            from playwright.async_api import async_playwright

            pass
        except ImportError:
            raise RuntimeError("Playwright is not installed. Please install it with 'pip install playwright'.")

        bearer_token = None
        token_expires_in = None

        async with async_playwright() as playwright:
            browser = None
            context = None

            try:
                logger.info(f"Launching browser for authentication (headless={self._headless})...")
                browser = await playwright.chromium.launch(
                    headless=self._headless, args=["--start-maximized"] if self._maximized else []
                )

                context = await browser.new_context(no_viewport=True)
                page = await context.new_page()

                # response_handler >>>
                async def response_handler(response):
                    nonlocal bearer_token, token_expires_in

                    try:
                        url = response.url

                        if "/oauth2/v2.0/token" in url:
                            try:
                                text = await response.text()

                                if (
                                    '"token_type":"Bearer"' in text or '"tokenType":"Bearer"' in text
                                ) and "sydney" in text:
                                    try:
                                        data = json.loads(text)
                                        if "access_token" in data:
                                            bearer_token = data["access_token"]
                                            token_expires_in = data.get("expires_in")

                                    except json.JSONDecodeError:
                                        logger.info("Response JSON decode failed, trying regex extraction...")

                                        match = re.search(r'"access_token"\s*:\s*"([^"]+)"', text)
                                        if match:
                                            bearer_token = match.group(1)
                                            logger.info("Captured bearer token using regex.")

                                            expires_match = re.search(r'"expires_in"\s*:\s*(\d+)', text)
                                            if expires_match:
                                                token_expires_in = int(expires_match.group(1))
                                        else:
                                            logger.error("Failed to extract bearer token using regex.")

                            except Exception as e:
                                logger.error(f"Error reading response: {e}")

                    except Exception as e:
                        logger.error(f"Error handling response: {e}")

                # ^^^ response_handler

                page.on("response", response_handler)

                logger.info("Navigating to Office.com for authentication...")
                await page.goto("https://www.office.com/")

                logger.info("Waiting for profile icon...")
                await page.wait_for_selector("#mectrl_headerPicture", timeout=self._timeout)
                await page.click("#mectrl_headerPicture")

                logger.info("Waiting for email input...")
                await page.wait_for_selector("#i0116", timeout=self._timeout)
                await page.fill("#i0116", self._username)
                await page.click("#idSIButton9")

                logger.info("Waiting for password input...")
                await page.wait_for_selector("#i0118", timeout=self._timeout)
                await page.fill("#i0118", self._password)
                await page.click("#idSIButton9")

                logger.info("Waiting for 'Stay signed in?' prompt...")
                await page.wait_for_selector("#idSIButton9", timeout=self._timeout)
                logger.info("Clicking 'Yes' to stay signed in...")
                await page.click("#idSIButton9")

                logger.info("Successfully logged in.")
                logger.info("Navigating to Copilot...")

                logger.info("Waiting for Copilot button and clicking it...")
                await page.wait_for_selector('div[aria-label="M365 Copilot"]', timeout=self._timeout)
                await page.click('div[aria-label="M365 Copilot"]', timeout=self._timeout)

                logger.info("Waiting 60 seconds for bearer token to be captured...")
                for _ in range(60):
                    if bearer_token:
                        break
                    await asyncio.sleep(1)

                if bearer_token:
                    logger.info(
                        f"Bearer token successfully retrieved. Preview: {bearer_token[:16]}...{bearer_token[-16:]}"
                    )
                    self._save_token_to_cache(token=bearer_token, expires_in=token_expires_in)
                else:
                    logger.error("Failed to retrieve bearer token within 60 seconds.")

                return bearer_token
            except Exception as e:
                logger.error("Failed to retrieve access token using Playwright.")

                if str(e).startswith("BrowserType.launch"):
                    logger.error("Playwright browser launch failed. Did you run 'playwright install chromium'?")
                else:
                    logger.error(f"Error details: {e}")

                return None
            finally:
                logger.info("Gracefully closing Playwright browser instance...")

                if context:
                    await context.close()
                if browser:
                    await browser.close()
