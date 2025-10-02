from argparse import ArgumentParser
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
import urllib.request
import urllib.error
import asyncio

import boto3
import jinja2

from botocore.exceptions import ClientError
from anthropic import AnthropicBedrock
from anthropic.types import Message

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

DEBUG = 1

@dataclass
class AbstractConfig(ABC):
  pass

class AbstractPromptBuilder(ABC):
  @abstractmethod
  def build_text(self, text: str | list):
    """ Building the text prompts. """
    raise NotImplementedError

  def build_tools(self):
    """ Optional method for formatting tool calling prompts. """
    pass

  def use_tools(self):
    """ Optional method for actually calling tools. """
    pass

class AbstractModel(ABC):
  config: AbstractConfig
  builder: AbstractPromptBuilder

  @abstractmethod
  def completion(self, chat: dict) -> str:
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

  @staticmethod
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
  api_token: str = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
  deployment: str = "claude-sonnet-4"
  region: str = "us-west-2"


class SimplePromptBuilder(AbstractPromptBuilder):
  def build_text(self, message: str):
    messages = [
      {"role": "user", "content": message}
    ]
    return messages

@dataclass
class AbstractMCPConfig(ABC):
  url: str
  headers: dict
  tools: list

  @abstractmethod
  def get_tools(self) -> list:
    pass

@dataclass
class GithubMCPConfig(AbstractConfig):
  url = "https://api.githubcopilot.com/mcp/"
  token = os.getenv("GITHUB_MCP_TOKEN")
  if not token:
    raise RuntimeError("Set GITHUB_MCP_TOKEN")
  headers = {"Authorization": f"Bearer {token}"}
  tools = []
  
  async def cache_tool_list(self):
    async with streamablehttp_client(
        url=self.url,
        headers=self.headers
    ) as (read, write, _sid):
      async with ClientSession(read, write) as mcp:
        await mcp.initialize()
        return (await mcp.list_tools()).tools
    
  def get_tools(self) -> list:
    if len(self.tools) == 0:
      if DEBUG:
        print("Load tools from the MCP server...")
      self.tools = asyncio.run(self.cache_tool_list())
    return self.tools

class BedrockPromptBuilder(AbstractPromptBuilder):
  def build_text(self, message: str):
    messages = [
      {"role": "user", "content": [{"text": message}]},
    ]
    return messages

class AnthropicBedrockPromptBuilder(AbstractPromptBuilder):
  def __init__(self, mcp_config: AbstractMCPConfig):
    self.mcp_config = mcp_config
    #TODO: async をなくしてMCPConfig側からツールのリストを引っ張る
    #TODO: responseからuse_toolsを行ってmsgに増やす部分を作る
    #TODO: ライブラリとして独立させてgithubに戻す
    #このコードを使ったサンプルコードを作る
    
  def build_text(self, message: str):
    messages = [
      {"role": "user", "content": message}
    ]
    return messages
  
  def build_tools(self):
    tools = self.mcp_config.get_tools()
    anthropic_tools = [{
      "name": t.name,
      "description": getattr(t, "description", "") or "",
      "input_schema": getattr(t, "inputSchema", {"type": "object"}),
    } for t in tools]

    return anthropic_tools

  async def use_tools(self, response) -> list:
    tool_uses = [b for b in response.content
                 if getattr(b, "type", None) == "tool_use"]

    results = []
    async with streamablehttp_client(
        url=self.mcp_config.url,
        headers=self.mcp_config.headers,
    ) as (read, write, _sid):
      async with ClientSession(read, write) as mcp:
        await mcp.initialize()

        for tu in tool_uses:
          res = await mcp.call_tool(tu.name, tu.input)
          out = []
          for c in res.content:
            typ = getattr(c, "type", None)
            if typ == "text":
              out.append(getattr(c, "text", ""))
            elif typ == "resource":
              out.append(f"[resource] {getattr(c, 'uri', '')}")
            else:
              try:
                out.append(json.dumps(c if isinstance(c, dict) else c.model_dump()))
              except Exception:
                out.append(str(c))
          results.append({"type": "tool_result", "tool_use_id": tu.id,
                        "content": "\n".join(out) or "(empty)"})

    return results   


class FilePromptBuilder(AbstractPromptBuilder):
  def __init__(self):
    self.env = jinja2.Environment(
      loader=jinja2.FileSystemLoader("./", encoding="utf-8"),
      trim_blocks=True,
      lstrip_blocks=True,
      undefined=jinja2.StrictUndefined,
    )

  def build_text(self, filepath: str, context: dict):
    template = self.env.get_template(filepath)
    return template.render(**context)

class AnthropicBedrockModel(AbstractModel):
  def __init__(
      self,
      config: BedrockAPIConfig,
      builder: AbstractPromptBuilder = SimplePromptBuilder()
  ):
    self.config = config
    self.builder = builder
    self.client = AnthropicBedrock(aws_region=config.region)
    self.retry = 10
    self.model_id = config.BEDROCK_ALIASES[config.deployment]
  
  def completion(self, text: str, **kwargs) -> tuple[list, str]:
    msgs = []
    msgs.append(self.builder.build_text(text))
    tools = self.builder.build_tools()
    resp = self.client.messages.create(
      model=self.model_id,
      messages=msgs,
      tools=tools,
      **kwargs,
    )
    results = asyncio.run(self.builder.use_tools(resp)) # type: ignore
    if results:
      msgs.append(self.builder.build_text(results))
    sys_out = resp.content[0].text
    return (msgs, sys_out)
  
class BedrockModel(AbstractModel):
  def __init__(self, config: BedrockAPIConfig, builder: AbstractPromptBuilder = BedrockPromptBuilder()):
    self.config = config
    self.builder = builder
    self.client = boto3.client('bedrock-runtime', region_name=config.region)
    self.retry = 10
    self.model_id = config.BEDROCK_ALIASES[config.deployment]

  def completion(self, prompt: str, **kwargs) -> str:
    if DEBUG:
      print("="*80)
      print(prompt)
    messages = self.builder.build_text(prompt) # type: ignore
    retry = self.retry
    while retry > 0:
      try:
        response = self.client.converse(
          modelId=self.model_id,
          messages=messages,
          **kwargs,
        )
      except self.client.exceptions.ServiceUnavailableException as e:
        print("Error:", e)
        time.sleep(60)
        retry -= 1
        continue
      except ClientError as e:
        if e.response['Error']['Code'] == 'ThrottlingException':
          print(f"ThrottlingException: {e}, wait 60 seconds...")
          time.sleep(60)
          continue
        gen = response['output']['message']['content'][0]['text']
        break
    if DEBUG:
      print(gen)
      print("="*80)
    return gen

class AzureOpenAIModel(AbstractModel):
  def __init__(self, config: AzureOpenAIAPIConfig, builder: AbstractPromptBuilder = SimplePromptBuilder()):
    self.config = config
    self.builder = builder
  
  def completion(self, example, **kwargs) -> str:
    url = f"{self.config.endpoint.rstrip('/')}/openai/deployments/{self.config.deployment}/chat/completions?api-version={self.config.api_version}"
    payload = {
        "messages": self.builder.build_text(example),
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
    count = 0
    while count < 10:
      try:
        with urllib.request.urlopen(req, timeout=300) as resp:
          body = resp.read()
      except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Azure OpenAI HTTPError {e.code}: {err}") from None
      except urllib.error.URLError as e:
        raise RuntimeError(f"Azure OpenAI URLError: {e.reason}") from None
      except TimeoutError:
        count += 1
        print("Request timed out... retry count:", count)
        continue

    obj = json.loads(body.decode("utf-8"))
    try:
      sys_out = obj["choices"][0]["message"]["content"]
      if DEBUG:
        print(sys_out)
        print("="*80)
        return sys_out
    except Exception:
      return json.dumps(obj, ensure_ascii=False)


