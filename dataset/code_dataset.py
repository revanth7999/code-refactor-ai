
import torch
from torch.utils.data import Dataset
from tokenizer.tokenizer import tokenize_code

class CodeRefactorDataset(Dataset):
    def __init__(self, data, vocab, max_len=64):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode(self, tokens):
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        ids = [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]
        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def __getitem__(self, idx):
        item = self.data[idx]
        inp = tokenize_code(item["input_code"])
        out = tokenize_code(item["output_code"])
        return {
            "input_ids": torch.tensor(self.encode(inp)),
            "target_ids": torch.tensor(self.encode(out))
        }
