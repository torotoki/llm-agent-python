from models import AzureOpenAIModel, AzureOpenAIAPIConfig
from cwe_model import 

def test_openai_model():
    config = AzureOpenAIAPIConfig(deployment="gpt-5")
    model = AzureOpenAIModel(config)
    gen = model.completion("Describe the P not equal NP problem.")
    print(gen)
    gen = model.completion("Output something.")
    print(gen)



def main():
    test_openai_model()


if __name__ == "__main__":
    main()