import torch
from torch import nn
import pickle
import time
from Transformer import LanguageModel
from BPE import BytePairEncoding



def create_objects():
    with open("FineTuneDS.txt", "r") as f:
        # Take a look at the .txt file. Here I am replacing \n with an unknown character that tokenizer hasn't seen before
        # The goal is to later replace <UNK> token with <EOS> when encoding the text
        data = f.read().replace("\n", "一")

    with open("./Datasets/tokenizer_4096.pkl", "rb") as f:
        tokenizer: BytePairEncoding = pickle.load(f)

    model_path = "./Datasets/SavedModels/cp_TL-1.7261_VL-1.7724_TI-16500.pth"
    config: dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)

    dropout = 0.1
    model = LanguageModel(vocab_size=tokenizer.get_vocab_size(), seq_len=config["seq_len"], n_embd=config["n_embd"],
                          n_heads=config["n_heads"], n_layers=config["n_layers"], dropout=dropout, device=device,
                          tokenizer=tokenizer).to(device)

    model.load_state_dict(config["model_state_dict"])

    return model, config, tokenizer, data



device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 1
seq_len = 128


model, config, tokenizer, data = create_objects()
model.train()

# Replace UNK with EOS to separate samples
data = tokenizer.encode(data)
for i in range(len(data)):
    if data[i] == tokenizer.UNK_token:
        data[i] = tokenizer.EOS_token


optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-5, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_token)


# Before Training
# while True:
#     user = input("Enter: ")
#     if user == "q":
#         break
#     print(model.generate(user, max_tokens=100, k=10))
#     print("\n-----------------------------\n")


# Now tuning the model. Basic training loop
start = time.time()
iterations = 0
for epoch in range(epochs):
    print("\n----------------------")
    print(f"Now at Epoch={epoch+1}")
    print("----------------------\n")

    for i in range(0, len(data)-seq_len, seq_len):
        tokenized_q, tokenized_a = [data[i:i+seq_len]], [data[i+1:i+1+seq_len]]

        tokenized_q = torch.tensor(tokenized_q, dtype=torch.long, device=device)
        tokenized_a = torch.tensor(tokenized_a, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(tokenized_q)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = tokenized_a.view(B * T)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iterations == 50:  # Give updates every 50 iterations
            print(f"progress={(i / len(data)) * 100:.2f}%, loss={loss.item():.4f}, Took {time.time() - start:.1f}s")
            start = time.time()
            iterations = 0

        iterations += 1


model.eval()
print("\n-------------------------\n")
while True:
    user = input("Enter: ")
    if user.lower() == "q":
        break
    print(model.generate(user, max_tokens=100, k=10))
    print("\n-----------------------------\n")


# Saving model
config["model_state_dict"] = model.state_dict()
torch.save(config, "./Datasets/SavedModels/cp_TL-1.7261_VL-1.7724_TI-16500_ft.pth")
