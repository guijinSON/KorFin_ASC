import tqdm
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score


def single_epoch_train_T5(model,optimizer,train_loader,device):
    model.train()
    loader = tqdm.tqdm(train_loader)

    for idx,batch in enumerate(loader):

        src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask = (
            batch['src_input_ids'].to(device),
            batch['src_attention_mask'].to(device),
            batch['tgt_input_ids'].to(device),
            batch['tgt_attention_mask'].to(device)
        )

        outputs = model(
            input_ids = src_input_ids,
            attention_mask = src_attention_mask,
            labels = tgt_input_ids,
            decoder_attention_mask = tgt_attention_mask
        )

        loss = outputs[0]

        wandb.log({"Training Loss":loss.item()})
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

@torch.no_grad()
def single_epoch_test_T5(model,
                           loader,
                           tgt_idx=-3,
                           device='cuda:0'
                           ):
    
    model.eval()
    acc_ = 0.0
    f1_ = 0.0

    for batch in tqdm.tqdm(loader):
        src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask = (
                        batch['src_input_ids'].to(device),
                        batch['src_attention_mask'].to(device),
                        batch['tgt_input_ids'],
                        batch['tgt_attention_mask']
                    )

        outputs = model.generate(
                input_ids=src_input_ids, attention_mask=src_attention_mask,
                max_length=32,
                num_beams = 3,
                early_stopping=True,
            )
    
        output_array = outputs[:,tgt_idx].detach().cpu()
        tgt_array = tgt_input_ids[:,tgt_idx]
        acc_ +=  accuracy_score(tgt_array, output_array)
        f1_ += f1_score(tgt_array, output_array, average='macro')

    acc_ /= len(loader)
    f1_ /= len(loader)
    
    wandb.log({
        "Test Accuracy":acc_,
        "Test F1 Score":f1_
    })
