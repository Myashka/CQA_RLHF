from datasets import load_dataset
from transformers import AutoTokenizer


def build_dataset(
    config,
    data_config,
    splits,
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def promt_tokenize(examples):
        if data_config['truncate_promt']:
            q_toks = tokenizer.encode(examples['Question'])
            q_toks = q_toks[:data_config['max_length_promt']-7]
            tmp = tokenizer.decode(q_toks).strip()
        else:
            tmp = examples['Question']

        sample = 'Question: ' + tmp + "\nAnswer:"

        tokenized_dict = tokenizer(
            sample, padding=data_config['padding'], max_length=data_config['max_length_promt'], truncation=True)
        
        tokenized_dict['query'] = sample

        return tokenized_dict

    datasets = []
    for split in splits:
        dataset = load_dataset(
            "json", data_files=f"{data_config['data_file_path']}", field=f'{split}')['train']
        dataset = dataset.map(promt_tokenize)
        dataset.set_format(type="torch", columns=["input_ids", "query"])
        datasets.append(dataset)
    return datasets

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])