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
                           test_loader_mult,
                           test_loader_sing,
                           device='cuda:0'):
    model.eval()
    mult_loader = tqdm.tqdm(test_loader_mult)
    sing_loader = tqdm.tqdm(test_loader_sing)

    mult_acc_ = 0.0
    mult_f1_ = 0.0
    sing_acc_ = 0.0
    sing_f1_ = 0.0

    for batch in mult_loader:
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
        mult_acc_ +=  accuracy_score(target, pred)
        mult_f1_ += f1_score(target, pred, average='macro')

    for batch in sing_loader:
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
        sing_acc_ +=  accuracy_score(target, pred)
        sing_f1_ += f1_score(target, pred, average='macro')

    mult_acc_ /= len(mult_loader)
    mult_f1_ /= len(mult_loader)
    sing_acc_ /= len(sing_loader)
    sing_f1_ /= len(sing_loader)
    
    wandb.log({
        "Test Accuracy(MULTIPLE_ENT)":mult_acc_,
        "Test F1 Score(MULTIPLE_ENT)":mult_f1_,
        "Test Accuracy(SINGLE_ENT)":sing_acc_,
        "Test F1 Score(SINGLE_ENT)":sing_f1_})
    
@torch.no_grad()
def single_epoch_test_T5(model,
                         test_loader_mult,
                         test_loader_sing,
                         tokenizer,
                         tgt_idx=-3,
                         device='cuda:0'):
    
    model.eval()
    mult_acc_ = 0.0
    mult_f1_ = 0.0
    sing_acc_ = 0.0
    sing_f1_ = 0.0
    
    mult_loader = tqdm.tqdm(test_loader_mult)
    sing_loader = tqdm.tqdm(test_loader_sing)
    
    for batch in mult_loader:
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
        mult_acc_ +=  accuracy_score(target, pred)
        mult_f1_ += f1_score(target, pred, average='macro')

    for batch in sing_loader:
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
        sing_acc_ +=  accuracy_score(target, pred)
        sing_f1_ += f1_score(target, pred, average='macro')

    mult_acc_ /= len(mult_loader)
    mult_f1_ /= len(mult_loader)
    sing_acc_ /= len(sing_loader)
    sing_f1_ /= len(sing_loader)
    
    wandb.log({
        "Test Accuracy(MULTIPLE_ENT)":mult_acc_,
        "Test F1 Score(MULTIPLE_ENT)":mult_f1_,
        "Test Accuracy(SINGLE_ENT)":sing_acc_,
        "Test F1 Score(SINGLE_ENT)":sing_f1_}) 
