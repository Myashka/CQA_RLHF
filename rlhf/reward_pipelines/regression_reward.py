import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Reward_pipeline:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token

        self.reward_model = self.reward_model.to(device)
    
    def __call__(self, input_texts, batch_size):
      return self.get_rewards(input_texts, batch_size)

    @torch.no_grad()
    def get_rewards(self, input_texts, batch_size):
        predictions = []
        
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i+batch_size]
            
            inputs = self.reward_tokenizer(batch, padding='longest', return_tensors='pt').to(self.device)
            outputs = self.reward_model(**inputs).logits
            
            batch_predictions = outputs.detach()
            predictions.append(batch_predictions)
        
        predictions = torch.cat(predictions, dim=0)
        
        return list(predictions)
