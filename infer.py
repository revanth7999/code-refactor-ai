import torch
import pandas as pd
from model.transformer_model import TransformerCodeGen
from tokenizer.tokenizer import tokenize_code, build_vocab

# Load data and vocab
df = pd.read_csv("data/code_pairs.csv").dropna()
data = df.to_dict(orient="records")
vocab, rev_vocab = build_vocab(data)
rev_vocab = {int(v): k for k, v in vocab.items()}  # Ensure keys are integers

# Load model
model = TransformerCodeGen(vocab_size=len(vocab))
model.load_state_dict(torch.load("refactor_model.pth"))
print("Model loaded successfully.")
model.eval()

def generate(raw_code):
    tokens = tokenize_code(raw_code)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    ids = [vocab["<bos>"]] + ids + [vocab["<eos>"]]
    ids += [vocab["<pad>"]] * (64 - len(ids))
    input_tensor = torch.tensor([ids])
    tgt_input = torch.tensor([[vocab["<bos>"]]])

    print("Input Tokens:", tokens)
    print("Token IDs:", ids[:10])
    print("Vocab:", list(vocab.keys())[:10])

    for _ in range(64):
        with torch.no_grad():
            output = model(input_tensor, tgt_input)
        next_token_logits = output[0, -1]
        next_token_id = torch.argmax(next_token_logits).item()
        print("Next token ID:", next_token_id, "-", rev_vocab.get(next_token_id, "<unk>"))

        if next_token_id == vocab["<eos>"]:
            break

        tgt_input = torch.cat([tgt_input, torch.tensor([[next_token_id]])], dim=1)

    # Decode the generated token IDs (excluding <bos>)
    decoded_ids = tgt_input[0][1:].tolist()
    decoded = [rev_vocab.get(i, "<unk>") for i in decoded_ids]

    print("Generated token IDs:", decoded_ids)
    print("Decoded tokens:", decoded)

    return " ".join(decoded)

# Run
print("Refactored:")
print(generate("int x=true;"))
