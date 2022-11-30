from tqdm import tqdm
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

@torch.no_grad()
def single_epoch_test_T5(model,
                           loader,
                           tgt_idx=-3,
                           device='cuda:0'
                           ):
    
    model.eval()
    acc_score = 0.0
    f1_score = 0.0

    for batch in tqdm(loader):
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
        acc_score +=  accuracy_score(tgt_array, output_array)
        f1_score += f1_score(tgt_array, output_array, average='macro')

    acc_score /= len(loader)
    f1_score /= len(loader)
    
    wandb.log({
        "Test Accuracy":acc_score,
        "Test F1 Score":f1_score
    })
