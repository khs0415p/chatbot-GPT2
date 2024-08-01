import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTrainer
from utils.train_utils import get_dataloader



class Trainer(BaseTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)

        self.bleu_score_list = []

        # dataloaders
        if config.mode == "train":
            self.dataloader, self.tokenizer = get_dataloader(config) # {'train': dataloader, 'valid': dataloader}
            self.config.vocab_size = len(self.tokenizer) 
            self.config.pad_token_id = self.tokenizer.pad_token_id

        # main process
        self.rank_zero = True if not self.ddp or (self.ddp and device == 0) else False

        # initialize trainer
        self._init_trainer()

        # criterion
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)


    def _training_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """        
        
        output = self.model(
            input_ids=model_inputs['input_ids'],    # batch, seq
            attention_mask=model_inputs['attention_mask'],
        )
        
        logits = output.logits # batch, seq, vocab_size
        loss = self.cross_entropy(logits[:, :-1, :].reshape(-1, self.config.vocab_size), model_inputs['input_ids'][:, 1:].reshape(-1))

        self._backward_step(loss)

        return loss.item()


    @torch.no_grad()
    def _validation_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """
        output = self.model(
            input_ids=model_inputs['input_ids'],    # batch, seq
            attention_mask=model_inputs['attention_mask'],
        )
        
        logits = output.logits # batch, seq, vocab_size
        loss = self.cross_entropy(logits[:, :-1, :].reshape(-1, self.config.vocab_size), model_inputs['input_ids'][:, 1:].reshape(-1))

        return loss.item(), logits

    @torch.no_grad()
    def test(self):
        print()

