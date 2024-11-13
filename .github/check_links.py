import re
import sys
import os
import time
import requests
from urllib.parse import urlsplit, urlunsplit
from concurrent.futures import ThreadPoolExecutor, as_completed


skipped_urls = [
    "https://cognitiveservices.azure.com/.default",
    "https://gandalf.lakera.ai/api/send-message",
    "https://code.visualstudio.com/Download",  # This will block python requests
    "https://platform.openai.com/docs/api-reference/introduction",  # blocks python requests
]

custom_myst_references = ["notebook_tests"]

# Updated regex pattern to capture URLs from Markdown and HTML
URL_PATTERN = re.compile(r'\[.*?\]\((.*?)\)|href="([^"]+)"|src="([^"]+)"')


def extract_urls(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    matches = URL_PATTERN.findall(content)
    # Flatten the list of tuples and filter out empty strings
    urls = [strip_fragment(url) for match in matches for url in match if url]
    return urls


def strip_fragment(url):
    """
    Removes the fragment (#...) from the URL, so the base URL can be checked.
    """
    parsed_url = urlsplit(url)
    return urlunsplit((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.query, ""))


def resolve_relative_url(base_path, url):
    if not url.startswith(("http://", "https://", "mailto:")):
        return os.path.abspath(os.path.join(os.path.dirname(base_path), url))
    return url


def check_url(url, retries=2, delay=2):
    """
    Check the validity of a URL, with retries if it fails.

    Args:
        url (str): URL to check.
        retries (int, optional): Number of retries if the URL check fails. Defaults to 2.
        delay (int, optional): Delay in seconds between retries. Defaults to 2.
    Returns:
        tuple: A tuple containing the URL and a boolean indicating whether it is valid.
    """

    if (
        "http://localhost:" in url
        or url in skipped_urls
        or any(url.endswith(reference) for reference in custom_myst_references)
        or os.path.isfile(url)
        or os.path.isdir(url)
        or url.startswith("mailto:")
    ):
        return url, True

    attempts = 0
    while attempts <= retries:
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            if response.status_code >= 400:
                attempts += 1
                if attempts > retries:
                    return url, False
                time.sleep(delay)
            else:
                return url, True
        except requests.RequestException:
            attempts += 1
            if attempts > retries:
                return url, False
            time.sleep(delay)


def check_links_in_file(file_path):
    urls = extract_urls(file_path)
    resolved_urls = [resolve_relative_url(file_path, url) for url in urls]
    broken_urls = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_url, url): url for url in resolved_urls}
        for future in as_completed(futures):
            url, is_valid = future.result()
            if not is_valid:
                broken_urls.append(url)
    return broken_urls


if __name__ == "__main__":
    files = sys.argv[1:]
    all_broken_urls = {}
    for file_path in files:
        print(f"Checking links in {file_path}")
        broken_urls = check_links_in_file(file_path)
        if broken_urls:
            all_broken_urls[file_path] = broken_urls
    if all_broken_urls:
        for file_path, urls in all_broken_urls.items():
            print(f"Broken links in {file_path}:")
            for url in urls:
                print(f"  - {url}")
        sys.exit(1)
    else:
        print("No broken links found.")
