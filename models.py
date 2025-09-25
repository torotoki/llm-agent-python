from argparse import ArgumentParser
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
import urllib.request
import urllib.error

import jinja2

DEBUG = 1

@dataclass
class AbstractConfig(ABC):
  pass

class AbstractPromptBuilder(ABC):
   @abstractmethod
   def build_prompt():
      pass

class AbstractModel(ABC):
  builder: AbstractPromptBuilder

  @abstractmethod
  def completion(self, chat: dict) -> dict:
    pass

class AbstractAgentModel(AbstractModel, ABC):
  impl: AbstractModel


@dataclass
class AzureOpenAIAPIConfig(AbstractConfig):
  AZURE_ALIASES = {
      "gpt-5": "gpt-5",
      "gpt-4": "gpt-4",
      "gpt-4o": "gpt-4o",
      "gpt-4.1": "gpt-4.1",
      "gpt-4.1-mini": "gpt-4.1-mini",
      "gpt-4o-mini": "gpt-4o-mini",
      "gpt-35-turbo": "gpt-35-turbo",
  }
  endpoint: str = os.getenv("AZURE_OPENAI_API_ENDPOINT", "")
  api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
  api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
  deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")

  def add_arguments(parser: ArgumentParser):
      # TODO: remove duplication
      parser.add_argument("--azure-endpoint", type=str, default=os.getenv("AZURE_OPENAI_API_ENDPOINT"))
      parser.add_argument("--azure-api-key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY", ""))
      parser.add_argument("--azure-api-version", type=str, default=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"))
      parser.add_argument("--azure-deployment", type=str, default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5"))


@dataclass
class BedrockAPIConfig(AbstractConfig):
  BEDROCK_ALIASES = {
    "claude-opus-4.1": "apac.anthropic.claude-opus-4-1-20250805-v1:0",
    "claude-opus-4": "apac.anthropic.claude-opus-4-20250514-v1:0",
    "claude-sonnet-4": "global.anthropic.claude-sonnet-4-20250514-v1:0",
  }
  api_token: str


class SimplePromptBuilder(AbstractPromptBuilder):
   def build_prompt(message: str):
      messages = [
        {"role": "user", "content": message}
      ]
      return messages

class FilePromptBuilder(AbstractPromptBuilder):
  def __init__(self):
     self.env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("./", encoding="utf-8"),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
     )

  def build_prompt(self, filepath: str, context: dict):
     template = self.env.get_template(filepath)
     return template.render(**context)

class AzureOpenAIModel(AbstractModel):
  def __init__(self, config: AzureOpenAIAPIConfig, builder: AbstractPromptBuilder = SimplePromptBuilder):
    self.config = config
    self.builder = builder
  
  def completion(self, example, **kwargs) -> dict:
    url = f"{self.config.endpoint.rstrip('/')}/openai/deployments/{self.config.deployment}/chat/completions?api-version={self.config.api_version}"
    payload = {
        "messages": self.builder.build_prompt(example),
    }
    if DEBUG:
      print("="*80)
      print(payload)
    payload.update(kwargs)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "api-key": self.config.api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Azure OpenAI HTTPError {e.code}: {err}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Azure OpenAI URLError: {e.reason}") from None

    obj = json.loads(body.decode("utf-8"))
    try:
        sys_out = obj["choices"][0]["message"]["content"]
        if DEBUG:
           print(sys_out)
           print("="*80)
        return sys_out
    except Exception:
        return json.dumps(obj, ensure_ascii=False)


