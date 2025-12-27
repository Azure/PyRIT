# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import asyncio
from typing import Optional

import json
import re

from pyrit.auth.authenticator import Authenticator

logger = logging.getLogger(__name__)


class CopilotAuthenticator(Authenticator):
    """
    Playwright-based authenticator for Microsoft Copilot.

    You need to have playwright installed and set up: ``pip install playwright && playwright install chromium``.
    """

    def __init__(self, *, headless: bool = False, maximized: bool = True, timeout: int = 10):
        """
        Initialize the CopilotAuthenticator.

        You must set the following environment variables for authentication:
            - COPILOT_USERNAME: Your Microsoft account username (email).
            - COPILOT_PASSWORD: Your Microsoft account password.

        Args:
            headless (bool): Whether to run the browser in headless mode. Default is False.
            maximized (bool): Whether to start the browser maximized. Default is True.
            timeout (int): Timeout for page operations in seconds. Default is 10.

        Raises:
            ValueError: If the required environment variables are not set.
        """
        super().__init__()

        import os

        self._username = os.getenv("COPILOT_USERNAME")
        self._password = os.getenv("COPILOT_PASSWORD")

        self._headless = headless
        self._maximized = maximized
        self._timeout = timeout * 1000  # ms

        if not self._username or not self._password:
            raise ValueError("COPILOT_USERNAME and COPILOT_PASSWORD environment variables must be set.")

    def refresh_token(self) -> str:
        """
        Refresh the authentication token.

        Returns:
            str: The refreshed authentication token.
        """
        raise NotImplementedError("refresh_token method not implemented")

    async def get_token(self) -> str:
        """
        Get the current authentication token.

        Returns:
            str: The current authentication token.
        """
        return await self._fetch_access_token_with_playwright()

    async def _fetch_access_token_with_playwright(self) -> Optional[str]:
        """
        Fetch access token using Playwright browser automation.

        Raises:
            RuntimeError: If Playwright is not installed.

        Returns:
            Optional[str]: The bearer token if successfully retrieved, else None.
        """
        try:
            from playwright.async_api import async_playwright

            pass
        except ImportError:
            raise RuntimeError("Playwright is not installed. Please install it with 'pip install playwright'.")

        bearer_token = None

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
                    nonlocal bearer_token

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

                                    except json.JSONDecodeError:
                                        logger.info("Response JSON decode failed, trying regex extraction...")

                                        match = re.search(r'"access_token"\s*:\s*"([^"]+)"', text)
                                        if match:
                                            bearer_token = match.group(1)
                                            logger.info("Captured bearer token using regex.")
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
