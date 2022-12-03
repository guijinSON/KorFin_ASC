import torch 
from torch.utils.data import Dataset, DataLoader

class KorFin_ABSA_Dataset(Dataset):
    def __init__(self, 
                 src, 
                 tgt, 
                 sentiment,
                 model_type = 'T5',
                 data_type = 'absa',
                 sep_token = '</s>',
                 mask_aspect = True,
                 sentiment_classification = {"긍정":0, "부정":1,"중립":2},
                 template   =  'The sentiment for [TGT] in the given sentence is [SENTIMENT].'
                 ):
        
        if model_type not in ['BERT','T5']:
            raise Exception(f"Undefined model_type error. {model_type} is not a valid model. Please choose from BERT or T5.")

        if '[SENTIMENT]' not in template:
            raise Exception(f"Template error. Please make sure the template include both [SENTIMENT] and [TGT].")

        self.src = src
        self.tgt = tgt
        self.sentiment = sentiment

        self.sep_token = sep_token
        self.model_type = model_type
        self.data_type = data_type
        self.mask_aspect = mask_aspect
        self.template = template
        self.sentiment_classification = sentiment_classification

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx] 
        if self.date_type == 'absa':
            tgt = self.tgt[idx]
        else:
            tgt = ''
        sentiment = self.sentiment[idx]

        if self.model_type == 'BERT':
            if self.mask_aspect:
                src = src.replace(tgt,'[TGT]')
            else: 
                src = ' '.join([src, self.sep_token, tgt])
        
        elif self.model_type == 'T5':
            if self.mask_aspect:
                src = src.replace(tgt,'[TGT]')
                tgt = self.template.replace('[SENTIMENT]',sentiment.lower())
            else: 
                src = ' '.join([src, self.sep_token, tgt])
                tgt = self.template.replace('[TGT]',tgt).replace('[SENTIMENT]',sentiment.lower())

        return {
            'SRC':src,
            'TGT':tgt,
            'SENTIMENT': (sentiment,self.sentiment_classification[sentiment.upper()])
        }


class Seq2SeqBatchGenerator:
    def __init__(self, 
                 tokenizer,
                 max_length,
                 model_type='T5'
                 ):
        
        if model_type not in ['BERT','T5']:
            raise Exception(f"Undefined model_type error. {model_type} is not a valid model. Please choose from BERT or T5.")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        
    def __call__(self, batch):
        src   = [item['SRC'] for item in batch]
        tgt = [item['TGT'] for item in batch]
    
        src_tokenized = self.tokenize(src)

        if self.model_type == 'T5':
            sentiment = [item['SENTIMENT'][0] for item in batch]
            tgt_tokenized = self.tokenize(tgt)
            
            tgt_input_ids = tgt_tokenized.input_ids
            tgt_attention_mask = tgt_tokenized.attention_mask

            label = sentiment

        else: 
            sentiment = [item['SENTIMENT'][1] for item in batch]

            tgt_input_ids = None
            tgt_attention_mask = None
            label = torch.tensor(sentiment)

        

        return {
            'src_input_ids': src_tokenized.input_ids, 
            'src_attention_mask': src_tokenized.attention_mask,
            'tgt_input_ids':tgt_input_ids,
            'tgt_attention_mask':tgt_attention_mask,
            'label': label
            }

    def tokenize(self,input_str):
        return  self.tokenizer.batch_encode_plus(input_str, 
                                                    padding='longest', 
                                                    max_length=self.max_length,
                                                    truncation=True, 
                                                    return_tensors='pt')


def get_dataloader(dataset, batch_generator, batch_size=4, shuffle=True):
    data_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=shuffle, 
                              collate_fn=batch_generator,
                              drop_last=True,
                              num_workers=4)
    return data_loader
