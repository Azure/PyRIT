import re
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

URL_PATTERN = re.compile(r'https?://[^\s)"]+')


def extract_urls(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return URL_PATTERN.findall(content)


def check_url(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code >= 400:
            return url, False
        return url, True
    except requests.RequestException:
        return url, False


def check_links_in_file(file_path):
    urls = extract_urls(file_path)
    broken_urls = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_url, url): url for url in urls}
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
