import torch

from model.model import SentenceTransformersWrapperForLM

class Config:
    hidden_size = 512
    vocab_size = 32000
    model_name = 'multi-qa-MiniLM-L6-cos-v1'
    dropout = 0.1

config = Config()

model = SentenceTransformersWrapperForLM(config)
model._extend_vocab_size()

input = {'input_ids': torch.tensor([[1,2,3,4], [2,3,4,5]]),
         'attention_mask': torch.tensor([[1,1,1,1], [1,1,1,0]])}

output = model(input)
print(model(input))
print('shape:')
print(output['token_embeddings'].shape)
print(output['sentence_embedding'].shape)
print(output['language_model_logits'].shape)