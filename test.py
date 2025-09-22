from models import OpenAIModel

def test_openai_model():
    model = OpenAIModel()
    messages = [
        {"role": "system", "content": "You are a professional scientist."},
        {"role": "user", "content": "Describe the P not equal NP problem."},
    ]
    gen = model.completition(messages)
    print(gen)



def main():
    test_openai_model()


if __name__ == "__main__":
    main()
