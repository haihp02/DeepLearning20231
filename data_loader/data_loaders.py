import tarfile
from tqdm import tqdm
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
from itertools import chain
import random
import logging
import os

from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, AutoTokenizer
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader, default_collate

from base.base_data_loader import BaseDataLoader
from data_loader.collate_fn import collate_fn_for_msmarco_mlm

class MSMarcoMLMDataset(Dataset):
    """
    MS Macro data loading using Dataset
    """

    def __init__(self, file_path : str,
                 tokenizer,
                 max_seq_length : int = 512,
                 mlm_probability = 0.15 , 
                 pad_to_max_length : bool = False,
                 line_by_line : bool = False,):
        import pandas as pd
        self.raw_data = pd.read_csv(file_path, sep='\t',index_col=0)
        self.raw_data = self.raw_data.iloc[:,0].tolist()
        self.dataset = Dataset.from_pandas(pd.DataFrame(self.raw_data, columns=["text"]))
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.pad_to_max_length = pad_to_max_length
        
        #process data 
        self.dataset = self.dataset.map(self.tokenize_function, remove_columns=["text"])
        if not line_by_line:
            self.dataset = self.dataset.map(self.group_texts, batched=True, num_proc=1)
        self.dataset = self.dataset.map(self.torch_call)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        # Tokenize the data and perform other preprocessing
        
        return self.dataset[int(index)]

    def tokenize_function(self,examples):
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=True,
        )
    
    def group_texts(self, examples):
        # examples is a dict of lists. We need to concatenate all lists in a single list.
        # an examples like {'input_ids': [[101, 102], [101,]], 'token_type_ids': [[0, 0], [0, 0, 0, 00]], 'attention_mask': [[1], [1, 1, 1]], 'special_tokens_mask': [[1,, 0, 1], [1, 0,1]]}
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
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=8 if not self.pad_to_max_length else None)


        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"], special_tokens_mask=special_tokens_mask)
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
    

class MLMDataLoader(BaseDataLoader):
    """
    BERT dataloader for MLM task
    A batch of data has type dict with keys:
    'input_ids', 'attention_mask', 'token_type_ids', 'special_token_mask', 'labels'
    """
    def __init__(self, tokenizer, data_path, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn_for_msmarco_mlm):
        self.dataset = MSMarcoMLMDataset(
            file_path=data_path,
            tokenizer=tokenizer,
        )
        super(MLMDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=validation_split,
            num_workers=num_workers,
            collate_fn=collate_fn
        )


# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased")
#     msmarco = MSMarcoMLMDataset(r'C:\Users\thanh\OneDrive\Desktop\20231\Deep Learing\DeepLearning20231\data_loader\sample.tsv',tokenizer = tokenizer)


