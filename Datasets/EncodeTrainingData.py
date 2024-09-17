import pickle
import numpy as np
from BPE import BytePairEncoding
import time


# My dataset has 45 files, adjust this range as needed
files = [f"./TextDS_100MB/sharded_text_{i:0{3}d}.txt" for i in range(1, 45)]
with open("tokenizer_4096.pkl", "rb") as f:
    tokenizer_4k: BytePairEncoding = pickle.load(f)


for i in range(len(files)):
    # Just reading and encoding each file.
    # The encoded list will then be converted into np array where each token is unit16 dtype to save space
    # One thing to note is the replacement of UNK token to EOS.
    # I separated each sample text with unknown character so the tokenizer would place an UNK token. However, this method isn't quite recommended
    start = time.time()
    with open(files[i], "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer_4k.encode(text)

    for j in range(len(tokens)):
        if tokens[j] == tokenizer_4k.UNK_token:
            tokens[j] = tokenizer_4k.EOS_token

    converted = np.array(tokens, dtype=np.uint16)
    np.save(f"./TrainingDataset_Tokens/sharded_{i+1:0{3}d}.npy", converted)

    print(f"File {files[i]} has been encoded")
    print(f"Took {time.time() - start:.2f}s")
    print("\n------------------------------\n")
