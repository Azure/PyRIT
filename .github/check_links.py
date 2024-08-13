import re
import sys
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Updated regex pattern to capture URLs from Markdown and HTML
URL_PATTERN = re.compile(r'\[.*?\]\((.*?)\)|href="([^"]+)"|src="([^"]+)"')


def extract_urls(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    matches = URL_PATTERN.findall(content)
    # Flatten the list of tuples and filter out empty strings
    urls = [url for match in matches for url in match if url]
    return urls


def resolve_relative_url(base_path, url):
    if not url.startswith(("http://", "https://", "mailto:")):
        return os.path.abspath(os.path.join(os.path.dirname(base_path), url))
    return url


def check_url(url):
    if (
        "http://localhost:" in url
        or url
        in [
            "https://cognitiveservices.azure.com/.default",
            "https://gandalf.lakera.ai/api/send-message",
            "https://code.visualstudio.com/Download",
        ]
        or os.path.isfile(url)
        or os.path.isdir(url)
        or url.startswith("mailto:")
    ):
        return url, True
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code >= 400:
            return url, False
        return url, True
    except requests.RequestException:
        return url, False


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
