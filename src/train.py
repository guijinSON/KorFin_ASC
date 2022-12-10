import tqdm
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score


def single_epoch_train_T5(model,
                          optimizer,
                          train_loader,
                          gradient_accumulation_steps=1,
                          device='cuda:0'):
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
        
        if idx % gradient_accumulation_steps==0:
            optimizer.step()
            optimizer.zero_grad()

@torch.no_grad()
def single_epoch_test_T5(model,
                         test_loader,
                         tokenizer,
                         tgt_idx=-3,
                         device='cuda:0'):
    
    model.eval()
    acc_ = 0.0
    f1_ = 0.0
    loader = tqdm.tqdm(test_loader)
    
    for batch in loader:
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
            ).detach().cpu()


        pred = [sent.split()[-1].strip().replace('적이다','').replace('.','') for sent in tokenizer.batch_decode(outputs,skip_special_tokens=True)]
        target = batch['label']
        acc_ +=  accuracy_score(target, pred)
        f1_ += f1_score(target, pred, average='macro')

    acc_ /= len(loader)
    f1_ /= len(loader)
    
    wandb.log({
        "Test Accuracy":acc_,
        "Test F1 Score":f1_
    })


def single_epoch_train_BERT(model,
                            optimizer,
                            train_loader,
                            gradient_accumulation_steps=1,
                            device='cuda:0'):

    model.train()
    loader = tqdm.tqdm(train_loader)

    for idx,batch in enumerate(loader):

        src_input_ids, src_attention_mask, labels = (
                        batch['src_input_ids'].to(device),
                        batch['src_attention_mask'].to(device),
                        batch['label'].to(device)
                        )
        
        outputs = model(
            input_ids = src_input_ids,
            attention_mask = src_attention_mask,
            labels = labels
        )

        loss = outputs.loss

        wandb.log({"Training Loss":loss.item()})
        loss.backward()
        
        if idx % gradient_accumulation_steps==0:
            optimizer.step()
            optimizer.zero_grad()
            
@torch.no_grad()
def single_epoch_test_BERT(model,
                           test_loader,
                           device='cuda:0'):
    model.eval()
    loader = tqdm.tqdm(test_loader)
    acc_ = 0.0
    f1_ = 0.0

    for batch in loader:
        src_input_ids, src_attention_mask, target = (
                        batch['src_input_ids'].to(device),
                        batch['src_attention_mask'].to(device),
                        batch['label']
                        )
        
        outputs = model(
            input_ids = src_input_ids,
            attention_mask = src_attention_mask
        )

        pred = torch.argmax(outputs.logits,dim=1).detach().cpu()
        acc_ +=  accuracy_score(target, pred)
        f1_ += f1_score(target, pred, average='macro')

    acc_ /= len(loader)
    f1_ /= len(loader)
    
    wandb.log({
        "Test Accuracy":acc_,
        "Test F1 Score":f1_
    })
