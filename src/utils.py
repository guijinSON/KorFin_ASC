from transformers import AdamW

def get_optimizer_T5(model, 
                     lr = 3e-4,
                     no_decay=['bias','layerNorm.weight']):
    
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    return optimizer
