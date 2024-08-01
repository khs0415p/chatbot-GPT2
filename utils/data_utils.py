import torch
import pandas as pd

from typing import List
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.data = pd.read_csv(config.data_path)
        self.max_length = config.max_length

        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens(
            {
                'pad_token': '[PAD]',
                'sep_token': '[SEP]',
                'cls_token': '[CLS]',
            }
        )


    def make_data(self, sentence_list: List[str]):
        """
        Args:
            sentence_list (List[str]) : Conversation list

        Returns:
            (List[str]) : [[CLS] sentence [SEP] sentence [SEP] ... [SEP]]
        """
        line = [self.tokenizer.cls_token_id]
        for sentence in sentence_list:
            temp = self.tokenizer.encode(sentence) + [self.tokenizer.sep_token_id]
            if len(line) + len(temp) > self.max_length: break
            line += temp

        return line 

    def __getitem__(self, index):
        row = self.data.iloc[index]

        sentence_list = row['발화']
        sentence_list = eval(sentence_list)

        sentence = self.make_data(sentence_list)

        return {
            "sentence" : torch.LongTensor(sentence)
        }


    def __len__(self):
        return len(self.data)