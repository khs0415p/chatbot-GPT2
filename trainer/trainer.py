import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTrainer
from utils.train_utils import get_dataloader
from transformers import AutoTokenizer



class Trainer(BaseTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)

        self.bleu_score_list = []

        # dataloaders
        if config.mode == "train":
            self.dataloader, self.tokenizer = get_dataloader(config) # {'train': dataloader, 'valid': dataloader}
            self.config.vocab_size = len(self.tokenizer) 
            self.config.pad_token_id = self.tokenizer.pad_token_id

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)

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
        self.model.eval()

        chat_history = [self.tokenizer.cls_token_id]
        while True:
            user_text = input("Enter chat (if you want to end, enter the exit) > ")

            if user_text in ['exit', 'exit()']: break

            chat_history += self.tokenizer.encode(user_text) + [self.tokenizer.sep_token_id]

            token_length = len(chat_history)

            if token_length >= self.config.max_length:
                print("Model maximum length exceeded.")
                break

            model_input = torch.LongTensor(chat_history).unsqueeze(0).to(self.device) # (1, len)

            for _ in range(self.config.max_length):
                output = self.model(model_input)
                new_token = torch.argmax(output.logits[:, -1], dim=-1) # (1)

                model_input = torch.cat((model_input, new_token.unsqueeze(1)), dim=1)       # (1, len) (1, 1)

                if new_token.item() == self.tokenizer.sep_token_id:
                    break

            model_answer = model_input[0, token_length:].detach().cpu().tolist()
            if model_answer[-1] == self.tokenizer.sep_token_id:
                chat_history += model_answer
                model_answer = model_answer[:-1]
            else:
                chat_history += model_answer + [self.tokenizer.sep_token_id]

            model_answer = self.tokenizer.decode(model_answer)
            
            print(f"Answer > {model_answer}")