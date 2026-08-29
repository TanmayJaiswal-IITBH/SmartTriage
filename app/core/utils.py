import re

def clean_text(text: str) -> str:
    """
    Cleans raw Markdown and HTML from GitHub text payloads.
    Prepares the text for accurate ML semantic embedding by removing noise.
    """
    if not text:
        return ""

    # 1. Remove HTML comments (e.g., <!-- Write your description here -->)
    # GitHub issue templates are full of these hidden instructions.
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # 2. Remove multi-line code blocks (```python ... ```)
    # Code syntax heavily skews NLP models.
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # 3. Remove inline code (`var = True`)
    text = re.sub(r'`[^`]*`', '', text)

    # 4. Remove URLs (http/https links)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # 5. Extract text from Markdown links/images: [visible text](url) -> visible text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 6. Remove Markdown headings (# Heading)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # 7. Remove blockquotes (> quote)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

    # 8. Remove bold/italics formatting (**bold**, *italics*, etc.)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)

    # 9. Normalize whitespace (Convert multiple spaces/newlines into a single space)
    # This leaves the model with a flat, clean string of words.
    text = re.sub(r'\s+', ' ', text).strip()

    return text