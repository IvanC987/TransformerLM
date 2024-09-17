from datasets import load_dataset
from collections import Counter
import time
import textstat


# I'm using 100MB for each file, but scale it to whatever size suits best
FILE_SIZE_BYTES = 100 * 1000 * 1000  # Each file to be ~100MB
MAX_NUM_FILES = 40  # Dataset would be ~40 * ~100MB = ~4GB
shard_index = 0  # File/Shard index
current_shard_size = 0  # Size of current list in bytes
shard_text = []

# For this model, I'm only using a tokenizer with ASCII-value-only training data for simplicity
allowed_chars = {'\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2',
                 '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']',
                 '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '~'}
special_chars = ['!', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', '|', '~', '[', ']', '_', '`']


def pass_requirements(example: dict):
    count = Counter(example["text"])

    if len(example["text"]) < 30 or len(example["text"]) > 1500:  # Setting lower and upper limit for the examples
        return False

    # Check if there are "too many" special characters. Those can be harder to work with.
    count_sc = 0
    for spec_char in special_chars:
        count_sc += count.get(spec_char, 0)
    if count_sc >= max(10, int(len(example["text"]) * 0.1)):
        return False

    # This requirement can be quite rigorous. However since this is a massive dataset, it's fine for our small-sized model
    for char in count.keys():
        if char not in allowed_chars:
            return False

    # Going to ignore site links as they can be irrelevant and steer model in an undesirable direction
    if "www." in example["text"] or "http" in example["text"]:
        return False

    # Checks the complexity of sentence
    if textstat.flesch_kincaid_grade(example["text"]) > 8:
        return False


    return True


# Used a variety of subsets from the list name=['default', 'sample-10BT', 'sample-100BT', 'sample-350BT', 'CC-MAIN-2024-10', 'CC-MAIN-2023-50', 'CC-MAIN-2023-40', 'CC-MAIN-2023-23', 'CC-MAIN-2023-14', 'CC-MAIN-2023-06', 'CC-MAIN-2022-49', 'CC-MAIN-2022-40', 'CC-MAIN-2022-33', 'CC-MAIN-2022-27', 'CC-MAIN-2022-21', 'CC-MAIN-2022-05', 'CC-MAIN-2021-49', 'CC-MAIN-2021-43', 'CC-MAIN-2021-39', 'CC-MAIN-2021-31', 'CC-MAIN-2021-25', 'CC-MAIN-2021-21', 'CC-MAIN-2021-17', 'CC-MAIN-2021-10', 'CC-MAIN-2021-04', 'CC-MAIN-2020-50', 'CC-MAIN-2020-45', 'CC-MAIN-2020-40', 'CC-MAIN-2020-34', 'CC-MAIN-2020-29', 'CC-MAIN-2020-24', 'CC-MAIN-2020-16', 'CC-MAIN-2020-10', 'CC-MAIN-2020-05', 'CC-MAIN-2019-51', 'CC-MAIN-2019-47', 'CC-MAIN-2019-43', 'CC-MAIN-2019-39', 'CC-MAIN-2019-35', 'CC-MAIN-2019-30', 'CC-MAIN-2019-26', 'CC-MAIN-2019-22', 'CC-MAIN-2019-18', 'CC-MAIN-2019-13', 'CC-MAIN-2019-09', 'CC-MAIN-2019-04', 'CC-MAIN-2018-51', 'CC-MAIN-2018-47', 'CC-MAIN-2018-43', 'CC-MAIN-2018-39', 'CC-MAIN-2018-34', 'CC-MAIN-2018-30', 'CC-MAIN-2018-26', 'CC-MAIN-2018-22', 'CC-MAIN-2018-17', 'CC-MAIN-2018-13', 'CC-MAIN-2018-09', 'CC-MAIN-2018-05', 'CC-MAIN-2017-51', 'CC-MAIN-2017-47', 'CC-MAIN-2017-43', 'CC-MAIN-2017-39', 'CC-MAIN-2017-34', 'CC-MAIN-2017-30', 'CC-MAIN-2017-26', 'CC-MAIN-2017-22', 'CC-MAIN-2017-17', 'CC-MAIN-2017-13', 'CC-MAIN-2017-09', 'CC-MAIN-2017-04', 'CC-MAIN-2016-50', 'CC-MAIN-2016-44', 'CC-MAIN-2016-40', 'CC-MAIN-2016-36', 'CC-MAIN-2016-30', 'CC-MAIN-2016-26', 'CC-MAIN-2016-22', 'CC-MAIN-2016-18', 'CC-MAIN-2016-07', 'CC-MAIN-2015-48', 'CC-MAIN-2015-40', 'CC-MAIN-2015-35', 'CC-MAIN-2015-32', 'CC-MAIN-2015-27', 'CC-MAIN-2015-22', 'CC-MAIN-2015-18', 'CC-MAIN-2015-14', 'CC-MAIN-2015-11', 'CC-MAIN-2015-06', 'CC-MAIN-2014-52', 'CC-MAIN-2014-49', 'CC-MAIN-2014-42', 'CC-MAIN-2014-41', 'CC-MAIN-2014-35', 'CC-MAIN-2014-23', 'CC-MAIN-2014-15', 'CC-MAIN-2014-10', 'CC-MAIN-2013-48', 'CC-MAIN-2013-20']
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-350BT", split="train", streaming=True)

start = time.time()
used_ex, total_ex = 0, 0  # Used as a metric to seem the % of dataset used
print("\n\nNow starting\n\n")
for ex in dataset:
    if shard_index >= MAX_NUM_FILES:  # Stop if we reach MAX_NUM_FILES
        break

    if pass_requirements(ex):  # Checks the requirements
        shard_text.append(ex["text"])
        current_shard_size += len(ex["text"].encode("utf-8"))
        used_ex += 1

    if current_shard_size > FILE_SIZE_BYTES:  # If size of text is greater than FILE_SIZE_BYTES, write to txt file
        print(f"Used {used_ex}/{total_ex} examples! ({(used_ex/total_ex) * 100: .2f}%)", flush=True)
        print(f"Took {time.time() - start:.2f}s", flush=True)
        print(f"Now writing sharded file {shard_index+1}", flush=True)
        print()
        start = time.time()

        # Separating each example with unknown character to tokenizer. Will later replace it as <EOS> token
        text = "\n一\n".join(shard_text)
        filename = f"sharded_text_{shard_index+1:0{3}d}.txt"
        with open("./TrainingDataset_Text/" + filename, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()

        print(f"File writing complete. Took {time.time() - start:.2f}s", flush=True)
        shard_index += 1
        current_shard_size = 0
        shard_text = []
        start = time.time()
    total_ex += 1


print("Sharding Complete")
