import re

def extract_tag(text: str, tag: str) -> str | None:
  pattern = re.compile(
    fr"^<{tag}>(.*?)</{tag}>",
    re.MULTILINE | re.DOTALL
  )
  match = re.findall(pattern, text)
  return match[0].strip() if match else None
