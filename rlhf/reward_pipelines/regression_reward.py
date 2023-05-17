import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import gc

class Reward_pipeline:
    def __init__(self, model_name, accelerator):
        self.model_name = model_name
        self.accelerator = accelerator

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token

        self.reward_model = accelerator.prepare(self.reward_model)
        self.reward_tokenizer = accelerator.prepare(self.reward_tokenizer)

        self.data_collator = DataCollatorWithPadding(self.reward_tokenizer, padding='longest')
    
    def __call__(self, input_texts, batch_size):
      return self.get_rewards(input_texts, batch_size)

    @torch.no_grad()
    def get_rewards(self, input_texts, batch_size):
        tokenized_input = self.reward_tokenizer(input_texts)
        input_data = [[input_ids, attnetion_mask] for input_ids, attnetion_mask in zip(tokenized_input['input_ids'], tokenized_input['attention_mask'])]

        del tokenized_input
        gc.collect()

        dataloader = DataLoader(input_data, batch_size,
                                collate_fn=self.collate_fn)
        
        dataloader = self.accelerator.prepare(dataloader)
        predictions = []
        
        for batch in dataloader:            
            outputs = self.reward_model(**batch).logits
            
            # batch_predictions = outputs.detach().cpu()
            # predictions.extend(batch_predictions)

            batch_predictions = outputs.detach()
            batch_predictions = self.accelerator.gather(batch_predictions)
            predictions.append(batch_predictions.cpu())
        
        predictions = torch.cat(predictions, dim=0)
        
        return predictions.tolist()
    
    def collate_fn(self, batch):
        input_ids = [torch.tensor(e[0], dtype=torch.long) for e in batch]
        attention_masks = [torch.tensor(e[1], dtype=torch.long) for e in batch]
        
        # input_ids = torch.stack(input_ids, dim=0)
        # attention_masks = torch.stack(attention_masks, dim=0)

        return self.data_collator({'input_ids': input_ids, 'attention_mask': attention_masks})
