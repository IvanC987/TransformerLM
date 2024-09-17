# Dataset Preparation and Encoding

This folder contains scripts and resources for preparing and encoding text data for model training. Below is an overview of the key files and their purposes.

## Scripts

### `GatherTrainingData.py`

This script preprocesses and shards a large text dataset into multiple files suitable for model training.

#### Features

- **Data Filtering**:
  - Filters texts based on length, character types, and complexity.
  - Excludes texts containing URLs and those with excessive special characters.
- **Sharding**:
  - Splits the dataset into files of approximately 100MB each.
  - Stores each shard in a text file with texts separated by a special token.


### `EncodeTrainingData.py`

This script encodes the text data from sharded files using a Byte Pair Encoding (BPE) tokenizer and saves the encoded data as NumPy arrays.

#### Features

- **Encoding**:
  - Reads text from sharded files and encodes it using a BPE tokenizer.
  - Adds <EOS> tokens as needed.
  - Saves encoded data as NumPy arrays for efficient storage and processing.


## Resources

### `SavedModels`

The `SavedModels` folder contains:

1. **cp_TL-1.7261_VL-1.7724_TI-16500.pth**: The base model before fine-tuning.
2. **cp_TL-1.7261_VL-1.7724_TI-16500_ft.pth**: The model after fine-tuning with additional training.

The `.pth` files are named using the format `"cp_TL-{TrainingLoss}_VL-{ValidationLoss}_TI-{TotalTrainingIterations}.pth"`, with the `_ft` suffix indicating the fine-tuned model.

### `tokenizer_4096.pkl`

- A pickle file containing the Byte Pair Encoding (BPE) tokenizer used for encoding the text data.

### Notes

1. The dataset, including text and tokenized files, totals around 90 files and 9GB. These will not be uploaded to the GitHub repository.
2. After training the model, it should be noted that `<PAD>` and `<UNK>` tokens were not used. 
Only `<EOS>` tokens were used to separate training samples. 
All unknown characters were pre-filtered, so `<UNK>` tokens were not included, and `<PAD>` tokens were not employed, which in hindsight it's my mistake. 
