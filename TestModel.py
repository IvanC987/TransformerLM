import torch
import pickle
from Transformer import LanguageModel
from BPE import BytePairEncoding


device = "cuda" if torch.cuda.is_available() else "cpu"
with open("./Datasets/tokenizer_4096.pkl", "rb") as f:
    tokenizer: BytePairEncoding = pickle.load(f)


# Adjust filepath as needed, this is using the fine-tuned model
model_path = "./Datasets/SavedModels/cp_TL-1.7261_VL-1.7724_TI-16500_ft.pth"
config: dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)


dropout = 0  # Dropout can be 0 here since we're evaluating the model rather than training
model = LanguageModel(vocab_size=tokenizer.get_vocab_size(), seq_len=config["seq_len"], n_embd=config["n_embd"], n_heads=config["n_heads"],
                      n_layers=config["n_layers"], dropout=dropout, device=device, tokenizer=tokenizer).to(device)

model.load_state_dict(config["model_state_dict"])


while True:
    user = input("Enter: ")
    print(model.generate(user, max_tokens=128, k=10))
    print("\n----------------------------------------\n")
