from datasets import Dataset
from itertools import chain
from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizerFast
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from collections.abc import Mapping
import pandas as pd

from datasets import Dataset
from itertools import chain
from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizerFast
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from collections.abc import Mapping
import pandas as pd
from transformers import AutoTokenizer
class MSMarcoDataset(Dataset):
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
        raise len(self.dataset)

    def __getitem__(self, index: int):
        # Tokenize the data and perform other preprocessing
        return self.dataset[index]

    def tokenize_function(self,examples):
        return self.tokenizer(
            examples['text'],
            truncation='only_second',
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




# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMacroTriplet(Dataset):
    def __init__(self, queries, corpus, ce_scores):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']
        qid = query['qid']

        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)    #Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)
        else:   #We only have negatives, use two negs
            pos_id = query['neg'].pop(0)    #Pop negative and add at end
            pos_text = self.corpus[pos_id]
            query['neg'].append(pos_id)

        #Get a negative passage
        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        pos_score = self.ce_scores[qid][pos_id]
        neg_score = self.ce_scores[qid][neg_id]

        return InputExample(texts=[query_text, pos_text, neg_text], label=pos_score-neg_score)

    def __len__(self):
        return len(self.queries)
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased")
    msmarco = MSMarcoDataset(r'C:\Users\thanh\OneDrive\Desktop\20231\Deep Learing\DeepLearning20231\data_loader\sample.tsv',tokenizer = tokenizer)
    print(msmarco[0])
    
    data_folder = 'msmarco-data'

    #### Read the corpus files, that contain all the passages. Store them in the corpus dict
    corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download collection.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    logging.info("Read corpus: collection.tsv")
    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            pid = int(pid)
            corpus[pid] = passage


    ### Read the train queries, store in queries dict
    queries = {}        #dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download queries.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)


    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query


    # Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
    # to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
    ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
    if not os.path.exists(ce_scores_file):
        logging.info("Download cross-encoder scores file")
        util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)

    logging.info("Load CrossEncoder scores dict")
    with gzip.open(ce_scores_file, 'rb') as fIn:
        ce_scores = pickle.load(fIn)

    # As training data we use hard-negatives that have been mined using various systems
    hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
    if not os.path.exists(hard_negatives_filepath):
        logging.info("Download cross-encoder scores file")
        util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', hard_negatives_filepath)


    logging.info("Read hard negatives train file")
    train_queries = {}
    negs_to_use = None
    with gzip.open(hard_negatives_filepath, 'rt') as fIn:
        for line in tqdm.tqdm(fIn):
            if max_passages > 0 and len(train_queries) >= max_passages:
                break
            data = json.loads(line)

            #Get the positive passage ids
            pos_pids = data['pos']

            #Get the hard negatives
            neg_pids = set()
            if negs_to_use is None:
                if args.negs_to_use is not None:    #Use specific system for negatives
                    negs_to_use = args.negs_to_use.split(",")
                else:   #Use all systems
                    negs_to_use = list(data['neg'].keys())
                logging.info("Using negatives from the following systems:", negs_to_use)

            for system_name in negs_to_use:
                if system_name not in data['neg']:
                    continue

                system_negs = data['neg'][system_name]
                negs_added = 0
                for pid in system_negs:
                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added += 1
                        if negs_added >= num_negs_per_system:
                            break

            if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
                train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}

    logging.info("Train queries: {}".format(len(train_queries)))
    
    train_biencoder_dataset = MSMARCODataset(queries=train_queries, corpus=corpus, ce_scores=ce_scores)
    print(train_biencoder_dataset[0])
    