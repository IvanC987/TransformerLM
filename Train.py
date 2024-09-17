import torch
from torch import nn
import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt
from Transformer import LanguageModel
from BPE import BytePairEncoding


class DataLoader:
    def __init__(self, directory, num_val_files):
        # _files holds the file paths
        # _index holds the index of next file
        file_names = [directory + f for f in os.listdir(directory)]
        self.training_files = file_names[:-num_val_files]
        self.train_index = 0
        self.val_files = file_names[-num_val_files:]
        self.val_index = 0

        # _pos is the index of current token
        # _data holds the numpy array of tokens
        self.train_pos = 0
        self.val_pos = 0
        self.training_data = []
        self._load_next_file(train=True)  # Load in the file and training data
        self.val_data = []
        self._load_next_file(train=False)  # Load in the file and validation data

        batches_per_file = int(len(self.training_data)) // (batch_size * seq_len)
        assert batches_per_file > 0, "Number of batches per file is less than 1!"
        print(f"Batches per file: {batches_per_file / 1000:.1f}K")
        print(f"Batches per epoch (training): {batches_per_file * len(self.training_files) / 1000:.1f}K")
        print(f"Batches per epoch (val): {batches_per_file * len(self.val_files) / 1000:.1f}K")

    def _load_next_file(self, train: bool):
        # Loads the numpy array, reset token index to 0 and increment file index
        if train:
            if self.train_index >= len(self.training_files):  # >= rather than == since checkpointing increment by 1
                self.train_index = 0

            self.training_data = np.load(self.training_files[self.train_index])
            self.train_pos = 0
            self.train_index += 1
        else:
            if self.val_index >= len(self.val_files):
                self.val_index = 0

            self.val_data = np.load(self.val_files[self.val_index])
            self.val_pos = 0
            self.val_index += 1

    def load_batch(self, train: bool):
        # Returns the next batch of tokens and loading in next file as needed
        if train:
            batch = self.training_data[self.train_pos: self.train_pos + (batch_size * seq_len) + 1]
            x, y = torch.tensor(batch[:-1], dtype=torch.long), torch.tensor(batch[1:], dtype=torch.long)
            x, y = x.reshape(batch_size, seq_len).to(device), y.reshape(batch_size, seq_len).to(device)

            self.train_pos += batch_size * seq_len
            if self.train_pos + (batch_size * seq_len + 1) >= len(self.training_data):
                self._load_next_file(train=True)

            return x, y
        else:
            batch = self.val_data[self.val_pos: self.val_pos + (batch_size * seq_len) + 1]
            x, y = torch.tensor(batch[:-1], dtype=torch.long), torch.tensor(batch[1:], dtype=torch.long)
            x, y = x.reshape(batch_size, seq_len).to(device), y.reshape(batch_size, seq_len).to(device)

            self.val_pos += batch_size * seq_len
            if self.val_pos + (batch_size * seq_len + 1) >= len(self.val_data):
                self._load_next_file(train=False)

            return x, y


@torch.no_grad()
def estimate_loss():
    # Evaluates the current training/validation loss
    out = {}
    model.eval()
    for split in ["train", "val"]:
        all_losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            x, y = data_loader.load_batch(train=True if split == "train" else False)
            logits = model(x)

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = y.view(B * T)
            loss = criterion(logits, targets)
            all_losses[k] = loss.item()

        out[split] = all_losses.mean()
    model.train()
    return out


def update_lr(step: int):
    # Updates the learning rate for optimizer
    if step == 0:
        lr = 3e-4
    else:
        lr = (n_embd ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_model():
    # Returns an instantiated model object
    model = LanguageModel(vocab_size=tokenizer.get_vocab_size(), seq_len=seq_len, n_embd=n_embd, n_heads=n_heads,
                          n_layers=n_layers, dropout=dropout, device=device, tokenizer=tokenizer).to(device)

    # Simple weight initialization using the Kaiming init
    for params in model.parameters():
        if params.dim() > 1:
            nn.init.kaiming_normal_(params)

    # Print the number of parameters in the model
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

    return model


# ***********************************
# Now loading everything
# ***********************************


if __name__ == "__main__":
    torch.manual_seed(89)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Currently using device={device}")

    # Hyper Parameters
    # ------------------------------------
    # Parameters for model
    batch_size = 32
    seq_len = 512
    n_embd = 512
    n_heads = 8
    n_layers = 4
    dropout = 0.1

    # Values for training loop
    training_iterations = 16500  # This is equivalent to one epoch for my dataset, adjust as needed
    grad_accum_steps = 8  # So 8 grad_accum_steps * 32 batches * seq_len=512 ~131K tokens. GPT Paper mentioned using 0.5M, but for our case, 131K works
    eval_iterations = 50
    eval_interval = 100  # 100 might seem short, however since grad_accum_steps is 8 we technically evaluate every 800 iterations
    warmup_steps = 4000

    # Values for checkpointing
    previously_trained_iters = 0
    save_cp_iters = 1000
    save_cp = True
    # ------------------------------------


    with open(r".\Datasets\tokenizer_4096.pkl", 'rb') as f:
        tokenizer: BytePairEncoding = pickle.load(f)

    directory = './Datasets/TrainingDataset_Tokens/'
    data_loader = DataLoader(directory=directory, num_val_files=3)
    model = create_model()
    optimizer = torch.optim.AdamW(params=model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_token)


    load_checkpoint = input("Load checkpoint? [Y/N]: ")
    while load_checkpoint.lower() != "y" and load_checkpoint.lower() != "n":
        print("Invalid input")
        load_checkpoint = input("Load checkpoint? [Y/N]: ")

    if load_checkpoint.lower() == "y":  # Update the current model and variables if loading in a checkpointed model
        name = input("Enter checkpoint filepath: ")
        checkpoint = torch.load(name)

        condition = checkpoint["seq_len"] == seq_len and checkpoint["n_embd"] == n_embd and checkpoint["n_heads"] == n_heads and checkpoint["n_layers"] == n_layers
        assert condition, "Given model's hyperparameter does not match!"

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        previously_trained_iters = checkpoint["step"]

        data_loader.train_index = checkpoint["train_index"]
        data_loader.load_batch(train=True)
        data_loader.val_index = checkpoint["val_index"]
        data_loader.load_batch(train=False)
        train_idx = data_loader.train_index if data_loader.train_index < len(data_loader.training_files) else 0
        val_idx = data_loader.val_index if data_loader.val_index < len(data_loader.val_files) else 0
        print(f"Currently using train_file={data_loader.training_files[train_idx]}, val_file={data_loader.val_files[val_idx]}")


    train_losses = []
    eval_losses = []
    start = time.time()
    for step in range(training_iterations):
        optimizer.zero_grad(set_to_none=True)

        # Applying gradient accumulation while keeping track of average loss
        total_loss = 0
        for mini_step in range(grad_accum_steps):
          x, y = data_loader.load_batch(train=True)
          logits = model(x)
          B, T, C = logits.shape
          logits = logits.view(B * T, C)
          targets = y.view(B * T)
          loss = criterion(logits, targets) / grad_accum_steps
          loss.backward()
          total_loss += loss.item()

        # Update learning rate and making sure to include previously trained iterations
        update_lr(step=step + previously_trained_iters)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevents unstable learning
        optimizer.step()

        if step % eval_interval == 0 or step == training_iterations - 1:
            out = estimate_loss()
            eval_losses.append([round(float(out["train"]), 4), round(float(out["val"]), 4)])
            print(f"At step={step}, Train Loss={out['train']:.4f}, Val Loss={out['val']:.4f}, Took {int(time.time() - start)}s")
            start = time.time()

        if save_cp and step != 0 and (step + previously_trained_iters) % save_cp_iters == 0:
            save_values = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                           "train_index": data_loader.train_index + 1, "val_index": data_loader.val_index + 1,
                           "step": step + previously_trained_iters, "seq_len": seq_len, "n_embd": n_embd,
                           "n_heads": n_heads, "n_layers": n_layers}

            print(f"Saving Model at step={step}")
            torch.save(save_values, f'cp_TL-{eval_losses[-1][0]}_VL-{eval_losses[-1][1]}_TI-{step + previously_trained_iters}.pth')

        train_losses.append(total_loss)


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(train_losses, c="red")
    ax1.set_title('Figure 1: Training Losses')
    if previously_trained_iters == 0:
        ax1.set_ylim(bottom=None, top=4)  # Adjust these values as needed

    plt.show()


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot([loss[0] for loss in eval_losses], c="red", label="Eval Train Loss")
    ax2.plot([loss[1] for loss in eval_losses], c="blue", label="Eval Val Loss")
    ax2.legend()
    ax2.set_title('Figure 2: Eval Losses')
    if previously_trained_iters == 0:
        ax2.set_ylim(bottom=None, top=4)  # Adjust these values as needed

    plt.show()


    model.eval()
    user = input("Enter: ")
    while user != "q":
        print(model.generate(user, max_tokens=128))
        user = input("Enter: ")
        print("\n-----------------------------\n")


    save_checkpoint = input("Save checkpoint? [Y/N]: ")
    while save_checkpoint.lower() != "y" and save_checkpoint.lower() != "n":
        print("Invalid input")
        save_checkpoint = input("Save checkpoint? [Y/N]: ")

    save_values = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                   "train_index": data_loader.train_index + 1, "val_index": data_loader.val_index + 1,
                   "step": training_iterations + previously_trained_iters, "seq_len": seq_len,
                   "n_embd": n_embd, "n_heads": n_heads, "n_layers": n_layers}

    if save_checkpoint.lower() == "y":
        print("Saving Model")
        torch.save(save_values, f'cp_TL-{eval_losses[-1][0]}_VL-{eval_losses[-1][1]}_TI-{training_iterations + previously_trained_iters}.pth')
