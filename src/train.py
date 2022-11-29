import tqdm
import torch
import wandb

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
