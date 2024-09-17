from collections import defaultdict
from tqdm import tqdm
import regex
import pickle
import time


class BytePairEncoding:
    """
    This implementation of BPE slightly deviates from the original version.
    It's less efficient, but for the sake of this project at this scale, it would suffice.
    """
    def __init__(self, text: str, desired_vocab_size: int, minimum_frequency=1000):
        # The pattern used by GPT2 in their GitHub- https://github.com/openai/gpt-2/blob/master/src/encoder.py
        self.pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.unique_chars = set(text)  # Used to compare input text to find unseen characters

        self.merge_dict = self._train_bpe(text, desired_vocab_size, minimum_frequency)
        self.EOS_token = len(self.merge_dict) + 255 + 1
        self.PAD_token = len(self.merge_dict) + 255 + 2
        self.UNK_token = len(self.merge_dict) + 255 + 3

        print(f"EOS Token: {self.EOS_token}")
        print(f"PAD Token: {self.PAD_token}")
        print(f"UNK Token: {self.UNK_token}")

        # For later on when decoding from bytes to string
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (b1, b2), idx in self.merge_dict.items():
            self.vocab[idx] = self.vocab[b1] + self.vocab[b2]

    @staticmethod
    def _count_frequency(bytes_list: list):
        # Returns the frequency of byte-pairs

        freq = defaultdict(int)
        for b1, b2 in zip(bytes_list[:-1], bytes_list[1:]):
            freq[(b1, b2)] += 1

        return freq

    @staticmethod
    def _merge_pairs(bytes_list: list, pair: tuple, replacement: int):
        # Merges the given pairs and returns the new list
        if len(bytes_list) == 0:
            raise ValueError(f"Bytes List is empty!")

        result = []
        i = 0
        while i < len(bytes_list) - 1:
            if bytes_list[i] == pair[0] and bytes_list[i + 1] == pair[1]:
                result.append(replacement)
                i += 2  # Skip the next byte as it's part of the pair
            else:
                result.append(bytes_list[i])
                i += 1

        # Append the last byte if it wasn't part of a pair
        if i < len(bytes_list):
            result.append(bytes_list[i])

        return result

    @staticmethod
    def _unmerge_pairs(bytes_list: list, idx: int, pair: tuple):
        # As the name suggests, it undoes the merges
        if len(bytes_list) == 0:
            raise ValueError(f"Bytes List is empty!")

        result = []
        for b in bytes_list:
            if b == idx:
                result.extend([pair[0], pair[1]])
            else:
                result.append(b)

        return result

    def _train_bpe(self, text: str, desired_vocab_size: int, minimum_frequency: int):
        # Trains the model by iteratively merging most frequency pairs

        if desired_vocab_size <= 256 + 3:  # First 256 for byte-representation. Remaining 3 for special tokens
            print(f"Warning, Desired Vocab Size should be greater than {256 + 3}")
            return {}

        bytes_list = list(text.encode("utf-8"))
        original_length = len(bytes_list)
        # Need to account for the initial 256 are used for encoding and 3 that are used for special tokens
        num_merges = desired_vocab_size - (256 + 3)
        next_idx = 256  # Start at 256, 0 to 255 are taken

        merges = {}
        count = 0
        start = time.time()
        while num_merges > 0:
            freq = self._count_frequency(bytes_list=bytes_list)
            most_common_pair = max(freq, key=lambda k: freq[k])
            pair_frequency = freq[most_common_pair]  # Identify the most common pair

            if pair_frequency < minimum_frequency:  # Exit early if max(pair frequency) < min specified frequency
                print("***********************************************************************************")
                print(f"Current pair \"{most_common_pair}\" occurred {pair_frequency} times")
                print(f"It is below the minimum frequency={minimum_frequency}, stopping merges...")
                print("***********************************************************************************")
                print()
                break

            print(f"Pair \"{most_common_pair}\" occurred {pair_frequency} times")
            bytes_list = self._merge_pairs(bytes_list, most_common_pair, next_idx)  # Merge and update
            print(f"Took {time.time() - start:.2f}s to merge \"{most_common_pair}\" with idx {next_idx}")
            start = time.time()
            print()

            merges[most_common_pair] = next_idx
            next_idx += 1
            count += 1
            num_merges -= 1

        # Prints additional information
        print(f"Total number of performed merges: {count}")
        print(f"Original byte-list length: {original_length}")
        print(f"New byte-list length: {len(bytes_list)}")
        print(f"Compression Ratio: {round(original_length/len(bytes_list), 3)}x")
        return merges

    def _replace_unknown(self, bytes_list: list, char: str):
        # This method takes in byte list and replaces all unknown chars with UNK token
        assert len(bytes_list) > 0, "Length Bytes_List cannot be empty!"

        byte_repr = list(char.encode("utf-8"))
        result = []

        i = 0
        while i < len(bytes_list) - len(byte_repr) + 1:
            if bytes_list[i: i + len(byte_repr)] == byte_repr:
                result.append(self.UNK_token)
                i += len(byte_repr)
            else:
                result.append(bytes_list[i])
                i += 1

        result.extend(bytes_list[i:])  # Add in any remaining bytes left
        return result


    def encode(self, text: str, seq_len=None, add_eos=False, add_pad=False):
        # Used to encode text after training

        assert seq_len is None or isinstance(seq_len, int), "Seq_Len must be None or of type int!"
        if add_pad and seq_len is None:
            raise ValueError("Seq_len must be specified if padding tokens needed")

        unknown = set()  # Checks for any characters that tokenizer has not encountered before
        for c in set(text):
            if c not in self.unique_chars:
                unknown.add(c)

        if len(unknown) > 0:
            print(f"Unknown characters found:\n{unknown}\n")

        split_text = self.pattern.findall(" " + text)  # Adding whitespace for consistency
        result = []

        # Adjust the value as needed. Purpose is to not display progress bar for user input, rather encoding of dataset
        use_tqdm = len(text) > 2048
        iterable = tqdm(split_text) if use_tqdm else split_text

        for txt in iterable:
            bytes_list = list(txt.encode("utf-8"))
            for c in unknown:
                bytes_list = self._replace_unknown(bytes_list, c)

            while len(bytes_list) >= 2:  # Base Case
                freq = self._count_frequency(bytes_list)
                pair = min(freq, key=lambda p: self.merge_dict.get(p, float("inf")))
                if pair not in self.merge_dict:
                    break
                idx = self.merge_dict[pair]
                bytes_list = self._merge_pairs(bytes_list, pair, idx)

            result.extend(bytes_list)

            if seq_len is not None and len(result) >= seq_len:  # Break early if met seq_len
                break

        # Returning conditions
        if seq_len is not None:
            if len(result) >= seq_len:
                return result[:seq_len-1] + [self.EOS_token] if add_eos else result[:seq_len]
            if add_eos:
                return result + [self.EOS_token] + [self.PAD_token for _ in range(seq_len - len(result) - 1)]
            return result + [self.PAD_token for _ in range(seq_len - len(result) - 1)]
        else:
            return result + [self.EOS_token] if add_eos else result

    def decode(self, bytes_list):
        # Returns the decoded string
        tokens = [self.vocab[idx] for idx in bytes_list]
        tokens = b"".join(tokens[1:])  # Remove the added whitespace used in .encode() method
        return tokens.decode("utf-8", errors="replace")

    def save_tokenizer(self, file: str):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    def get_vocab_size(self):
        return self.UNK_token + 1


if __name__ == "__main__":
    # I used a 100MB text file to train my tokenizer
    with open(fr".\TrainingDataset\sharded_text_001.txt", "r", encoding="utf-8") as f:
        txt = f.read()

    print(f"There are {len(txt)} characters in the dataset")
    print("Now training tokenizer")
    print()

    vocab_size = 4096
    tokenizer = BytePairEncoding(text=txt, desired_vocab_size=vocab_size, minimum_frequency=1000)
    tokenizer.save_tokenizer(f"tokenizer_{vocab_size}.pkl")
