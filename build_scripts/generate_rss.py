# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from html.parser import HTMLParser
from pathlib import Path

from feedgen.feed import FeedGenerator


# HTML parser to extract title and description
class BlogEntryParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.description = ""
        self.current_tag = ""
        self.description_step = 0

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.description_step += 1
        self.current_tag = tag

    def handle_endtag(self, tag):
        if tag == "p":
            self.description_step += 1
        self.current_tag = ""

    def handle_data(self, data):
        if self.current_tag == "title":
            self.title = data
        elif self.description_step == 3:
            self.description = self.description + data


# Generate the RSS feed structure
print("Generating RSS feed structure...")
fg = FeedGenerator()
fg.link(href="https://azure.github.io/PyRIT/blog/rss.xml", rel="self")
fg.title("PyRIT Blog")
fg.description("PyRIT Blog")
fg.logo("https://azure.github.io/PyRIT/_static/roakey.png")
fg.language("en")

# Iterate over the blog files and sort them
print("Pulling blog files...")
directory = Path("doc/_build/html/blog/")
files = [file for file in directory.iterdir() if file.is_file() and file.name.startswith("20")]
files.sort(key=lambda x: x.name)

# Add a feed entry for each file
for file in files:
    print(f"Parsing {file.name}...")
    fe = fg.add_entry()
    fe.link(href=f"https://azure.github.io/PyRIT/blog/{file.name}")
    fe.guid(f"https://azure.github.io/PyRIT/blog/{file.name}")

    # Extract title and description from HTML content
    with open(file, "r", encoding="utf-8") as f:
        parser = BlogEntryParser()
        parser.feed(f.read())
        fe.title(parser.title)
        fe.description(parser.description)

    # Extract publication date from file name
    fe.pubDate(f"{file.name[:10].replace('_', '-')}T10:00:00Z")

# Export the RSS feed
print("Exporting RSS feed...")
fg.rss_file("doc/_build/html/blog/rss.xml", pretty=True)

print("Done.")
