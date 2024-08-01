import torch.nn as nn

from transformers import GPT2LMHeadModel


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(config.model_path, cache_dir=config.cache_dir)
        self.model.resize_token_embeddings(config.vocab_size)
        self.config = self.model.config


    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output
    


