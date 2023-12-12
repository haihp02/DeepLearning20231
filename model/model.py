import torch.nn as nn
import torch.nn.functional as F
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

from base import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JointEmbedding(nn.Module):

    def __init__(self, vocab_size, size):
        super(JointEmbedding, self).__init__()

        self.size = size

        self.token_emb = nn.Embedding(vocab_size, size)
        self.segment_emb = nn.Embedding(vocab_size, size)

        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor):
        sentence_size = input_tensor.size(-1)
        pos_tensor = self.attention_position(self.size, input_tensor)

        segment_tensor = torch.zeros_like(input_tensor).to(device)
        segment_tensor[:, sentence_size // 2 + 1:] = 1

        output = self.token_emb(input_tensor) + self.segment_emb(segment_tensor) + pos_tensor
        return self.norm(output)

    def attention_position(self, dim, input_tensor):
        batch_size = input_tensor.size(0)
        sentence_size = input_tensor.size(-1)

        pos = torch.arange(sentence_size, dtype=torch.long).to(device)
        d = torch.arange(dim, dtype=torch.long).to(device)
        d = (2 * d / dim)

        pos = pos.unsqueeze(1)
        pos = pos / (1e4 ** d)

        pos[:, ::2] = torch.sin(pos[:, ::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        return pos.expand(batch_size, *pos.size())

    def numeric_position(self, dim, input_tensor):
        pos_tensor = torch.arange(dim, dtype=torch.long).to(device)
        return pos_tensor.expand_as(input_tensor)


class AttentionHead(nn.Module):

    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()

        self.dim_inp = dim_inp

        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)

        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale

        scores = scores.masked_fill_(attention_mask, -1e9)
        attn = f.softmax(scores, dim=-1)
        context = torch.bmm(attn, value)

        return context


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([
            AttentionHead(dim_inp, dim_out) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)


class Encoder(nn.Module):

    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1):
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out)  # batch_size x sentence size x dim_inp
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        context = self.attention(input_tensor, attention_mask)
        res = self.feed_forward(context)
        return self.norm(res)


class BERT(BaseModel):

    def __init__(self, config):
        super(BERT, self).__init__()
        raise NotImplementedError   
        

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        raise NotImplementedError  
    
    def load_weights_from_huggingface(self, model_name):
        # Load weights manually from Hugging Face checkpoint
        # This example assumes 'model_name' is the path to the checkpoint
        # You'll need to load individual weights and set them to corresponding layers

        # Example: Loading weights for the JointEmbedding layer
        # Replace 'self.embedding.token_emb.weight' and 'self.embedding.segment_emb.weight' 
        # with the weights from the Hugging Face checkpoint
        self.embedding.token_emb.weight = nn.Parameter(torch.tensor(...))  # Load token_emb weights
        self.embedding.segment_emb.weight = nn.Parameter(torch.tensor(...))  # Load segment_emb weights

        # Load other weights for Encoder, token_prediction_layer, classification_layer, etc.


class SentenceTransformersWrapperForLM(BaseModel):

    def __init__(self, model_name, model_path, hidden_size, dropout, vocab_size, load_path=None):
        super(SentenceTransformersWrapperForLM, self).__init__()
        
        self.model_name = model_name
        self.model_path = model_path
        self.vocal_size = vocab_size

        if model_name:
            self.model = SentenceTransformer(model_name)
        elif model_path:
            self.model = SentenceTransformer(model_path)
        self._extend_vocab_size(vocab_size)
        self.lm_output_layer = nn.Sequential(
            nn.LazyLinear(out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=vocab_size)
        )

        if load_path is not None:
            torch_model_checkpoint = torch.load(load_path, map_location='cpu')
            self.load_state_dict(torch_model_checkpoint['state_dict'])
            print('Loaded model from', load_path)

    def forward(self, input_features):
        '''
        input_features: dict, include:
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
        - token_type_ids: optional (batch_size, seq_len)

        return: 
        - type: dict with 'token_embddings', 'sentence_embeddings' and 'language_model_logits'
        - 'token_embeddings': last layer embeddings for all tokens in all sentences in batch (batch_size, seq_len, embeddings_size)
        - 'sentence_embedding': sentence embeddings for all sentences in batch (batch_size, seq_len)
        - 'language_model_logits': logits for language modelling (batch_size, vocab_size, seq_len)
        '''
        out_features = self.model.forward(input_features)

        embeddings = []
        for sent_idx in range(len(out_features['sentence_embedding'])):
            row = {name: out_features[name][sent_idx] for name in out_features}
            embeddings.append(row)
        
        batch_output = {}
        batch_output['token_embeddings'] = torch.stack([embedding['token_embeddings'] for embedding in embeddings])
        batch_output['sentence_embedding'] = torch.stack([embedding['sentence_embedding'] for embedding in embeddings])
        batch_output['language_model_logits'] = torch.stack([self.lm_output_layer(embedding['token_embeddings']) for embedding in embeddings]).swapaxes(1, 2)
        return batch_output
        
    def _extend_vocab_size(self, new_vocab_size=None):
        '''
        Update model embedding layer for larger vocabulary, keeps all the trained embeddings
        '''
        if not new_vocab_size:
            new_vocab_size = self.vocab_size
        old_embedding_layer = self.model._first_module().auto_model.embeddings.word_embeddings
        old_vocab_size = old_embedding_layer.weight.shape[0]
        embedding_size = old_embedding_layer.weight.shape[1]
        new_embedding_layer = nn.Embedding(num_embeddings=new_vocab_size, embedding_dim=embedding_size)
        new_embedding_layer.weight.data[:old_vocab_size] = old_embedding_layer.weight.data
        self.model._first_module().auto_model.embeddings.word_embeddings = new_embedding_layer

class BERTBiEncoder(nn.Module):
    def __init__(
        self,
        query_encoder_name,
        query_encoder_path,
        passage_encoder_name,
        passage_encoder_path,
        vocab_size,
        load_path=None,
        query_encoder_load_path=None,
        passage_encoder_load_path=None,
    ):
        super(BERTBiEncoder, self).__init__()
        
        self.query_encoder_name = query_encoder_name
        self.query_encoder_path = query_encoder_path
        self.passage_encoder_name = passage_encoder_name
        self.passage_encoder_path = passage_encoder_path
        self.vocab_size = vocab_size

        if query_encoder_name:
            self.query_encoder = SentenceTransformer(query_encoder_name)
        elif query_encoder_path:
            self.query_encoder = SentenceTransformer(query_encoder_path)
        self._extend_encoder_vocab_size(encoder=self.query_encoder, new_vocab_size=vocab_size)
        
        if passage_encoder_name:
            self.passage_encoder = SentenceTransformer(passage_encoder_name)
        elif passage_encoder_path:
            self.passage_encoder = SentenceTransformer(passage_encoder_path)
        self._extend_encoder_vocab_size(encoder=self.passage_encoder, new_vocab_size=vocab_size)
        
        if load_path is not None:
            torch_model_checkpoint = torch.load(load_path, map_location='cpu')
            self.load_state_dict(torch_model_checkpoint['state_dict'])
            print('Loaded model from', load_path)
        if query_encoder_load_path is not None:
            torch_model_checkpoint = torch.load(query_encoder_load_path, map_location='cpu')
            self.query_encoder.load_state_dict(torch_model_checkpoint['state_dict'])
            print('Loaded query encoder from', query_encoder_load_path)
        if passage_encoder_load_path is not None:
            torch_model_checkpoint = torch.load(passage_encoder_load_path, map_location='cpu')
            self.passage_encoder.load_state_dict(torch_model_checkpoint['state_dict'])
            print('Loaded model from', passage_encoder_load_path)

    def forward(self, query_features, passage_features):
        query_embeddings = self.query_encoder.forward(query_features)['sentence_embedding']
        passage_embeddings = self.passage_encoder.forward(passage_features)['sentence_embedding']
        scores = torch.stack([F.cosine_similarity(q_emb, passage_embeddings) for q_emb in query_embeddings])
        outputs = {
            'scores': scores,
            'query_embeddings': query_embeddings,
            'passage_embeddings': passage_embeddings
        }
        return outputs

    def _extend_encoder_vocab_size(self, encoder: SentenceTransformer, new_vocab_size=None):
        '''
        Update model embedding layer for larger vocabulary, keeps all the trained embeddings
        '''
        if not new_vocab_size:
            new_vocab_size = self.vocab_size
        old_embedding_layer = encoder._first_module().auto_model.embeddings.word_embeddings
        old_vocab_size = old_embedding_layer.weight.shape[0]
        embedding_size = old_embedding_layer.weight.shape[1]
        new_embedding_layer = nn.Embedding(num_embeddings=new_vocab_size, embedding_dim=embedding_size)
        new_embedding_layer.weight.data[:old_vocab_size] = old_embedding_layer.weight.data
        encoder._first_module().auto_model.embeddings.word_embeddings = new_embedding_layer        