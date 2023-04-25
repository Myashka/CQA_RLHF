def freeze_model(model, config):
    for name, p in model.named_parameters():
        name = name.lower()
        if 'transformer.h' in name and int(name.split('.')[3]) in config['layers_not_to_freeze']:
            p.requires_grad = True
            continue
        if 'ln' in name or 'norm' in name:
            p.requires_grad = not config['freeze_ln']
        elif 'wte' in name or 'wpe' in name:
            p.requires_grad = not config['freeze_emb']
        elif 'mlp' in name:
            p.requires_grad = not config['freeze_ff']
        elif 'attn' in name:
            p.requires_grad = not config['freeze_attn']
        else:
            p.requires_grad = not config['freeze_other']
        
    print('Model freezed')
    return model