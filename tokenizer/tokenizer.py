import re

def tokenize_code(code):
    if not isinstance(code, str):
        return []
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|==|!=|<=|>=|[\(\){}\[\];=+\-*/<>.,\"]", code)

def build_vocab(data):
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    idx = 4
    for item in data:
        for key in ["input_code", "output_code"]:
            tokens = tokenize_code(item[key])
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
    rev_vocab = {int(v): k for k, v in vocab.items()}
    return vocab, rev_vocab
