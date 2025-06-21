
import torch
from torch.utils.data import DataLoader
import pandas as pd
from model.transformer_model import TransformerCodeGen
from tokenizer.tokenizer import tokenize_code, build_vocab
from dataset.code_dataset import CodeRefactorDataset

df = pd.read_csv("data/code_pairs.csv")
df = df.dropna()
data = df.to_dict(orient="records")

vocab, rev_vocab = build_vocab(data)
dataset = CodeRefactorDataset(data, vocab)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = TransformerCodeGen(vocab_size=len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

model.train()
for epoch in range(1500):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]

        output = model(input_ids, target_ids[:, :-1])
        loss = loss_fn(output.reshape(-1, output.shape[-1]), target_ids[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "refactor_model.pth")
