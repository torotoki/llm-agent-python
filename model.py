from argparse import ArgumentParser
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
import urllib

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

class AbstractAgentModel(ABC, AbstractModel):
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
        {"role": "system", "content": message}
      ]

class AzureOpenAIModel(AbstractModel):
  def __init__(self, config: AzureOpenAIAPIConfig, builder: AbstractPromptBuilder = SimplePromptBuilder):
    self.config = config
    self.builder = builder
  
  def completion(self, example, max_new_tokens = 1024, do_sample = False) -> dict:
    url = f"{self.config.endpoint.rstrip('/')}/openai/deployments/{self.config.deployment}/chat/completions?api-version={self.config.api_version}"
    payload = {
        "messages": self.builder(example),
        "max_completion_tokens": max_new_tokens,
        "temperature": 1.0 if do_sample else 0.0,
        "stream": False,
    }
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Azure OpenAI HTTPError {e.code}: {err}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Azure OpenAI URLError: {e.reason}") from None

    obj = json.loads(body.decode("utf-8"))
    try:
        return obj["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(obj, ensure_ascii=False)
