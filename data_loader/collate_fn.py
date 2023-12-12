import torch

def collate_fn_for_msmarco_mlm(batch):
    '''
    Custom collate function to transform multiple dictionarys with list as values
    to single dictionary with batch of lists in form of tensor as values.
    '''
    collected_batch = {}
    for key in batch[0].keys():
        collected_batch[key] = torch.stack([torch.tensor(sample[key]) for sample in batch])
    return collected_batch