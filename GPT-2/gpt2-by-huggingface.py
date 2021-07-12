from transformers import GPT2Tokenizer, GPT2Model
from transformers import pipeline, set_seed


def make_data():
    sentences = []
    with open("../shakespeare") as f:
        while 1:
            line = f.readline()
            if not line:
                break
            if line == '\n':
                line = "[CLS]"
            else:
                line = line.replace("\n", "[SEP]")
            sentences.append(line)
    return sentences


if __name__ == "__main__":
    sentences = make_data()
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
