from torchvision import datasets, transforms
from base import BaseDataLoader
from itertools import chain
from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizerFast
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from collections.abc import Mapping

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    
class MSMacroDataLoader(BaseDataLoader):
    """
    MS Macro data loading using Dataset
    """

    def __init__(self, file_path, max_seq_length, mlm_probability, line_by_line, pad_to_max_length, tokenizer,
                 batch_size=32, shuffle=True):
        self.raw_datasets = load_dataset('json', data_files=file_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.pad_to_max_length = pad_to_max_length
        
        self.tokinzed_datasets = self._tokenize_and_prepare_data(self.raw_datasets)
        self.tokinzed_datasets = 
        super(MSMacroDataLoader, self).__init__(
            dataset=self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle
        )

    def __len__(self) -> int:
        return len(self.raw_datasets)

    def __getitem__(self, index: int):
        # Tokenize the data and perform other preprocessing
        return self._tokenize_and_prepare_data(self.raw_datasets[index])

    def _tokenize_function(self, examples):
        return self.tokenizer(examples['passage'], return_special_tokens_mask=True)

    def _group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Logic to generate chunks of max_seq_length
        total_length = (total_length // self.max_seq_length) * self.max_seq_length
        result = {
            k: [t[i: i + self.max_seq_length] for i in range(0, total_length, self.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_prepare_data(self, examples):
        processed_examples = self._tokenize_function(examples)
        processed_examples = self._group_texts(processed_examples)
        return processed_examples
    
    def collate_fn(self, batch):
        # Implement logic here for padding sequences and generating MLM labels
        # This function should return processed batch data
        # Use the tokenizer and MLM logic from DataCollatorForLanguageModeling here
        pass
    def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        import torch

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": self._torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels