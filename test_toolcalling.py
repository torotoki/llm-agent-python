from pprint import pprint
from models import (
    BedrockAPIConfig,
    AnthropicBedrockModel,
    AnthropicBedrockPromptBuilder,
    GithubMCPConfig,
)


def test_anthropic_model():
  config = BedrockAPIConfig(deployment="claude-sonnet-4")
  mcp_config = GithubMCPConfig()
  builder = AnthropicBedrockPromptBuilder(mcp_config)
  model = AnthropicBedrockModel(config, builder)
  state, gen = model.completion(
      "Find the Linux project and display first 10 lines of one random "
      "file from the repository.",
      output_state=True
  )
  pprint(state)
  print(gen)
  print("="*80)
  for i in range(10):
    state, gen = model.completion(state=state, output_state=True)
    #print(state)
    print(i, gen)
    print("="*80)


def main():
  test_anthropic_model()


if __name__ == "__main__":
  main()