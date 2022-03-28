import random
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,hamming_loss
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, BertTokenizerFast, get_linear_schedule_with_warmup,RobertaTokenizerFast,RobertaModel,RobertaConfig
import json
import time
import pandas
import pytorch_lightning as pl
from math import log
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# @title GlobalConfig
class GlobalConfig:
    def __init__(self):
        self.seed = 2020
        self.path = Path('./data/')
        self.max_length = 64
        self.roberta_path = 'cahya/bert-base-indonesian-1.5G'  # @param
        self.num_workers = os.cpu_count()
        self.batch_size = 8
        self.accum_steps = 1
        num_epochs = 15  # @param
        self.epochs = num_epochs
        self.warmup_steps = 0
        lr = 5e-6  # @param
        self.lr = lr  # modified from 3e-5
        run_id = "stage1"  # @param
        self.offline = True
        self.saved_model_path = run_id
        self.n_splits = 5

def get_callbacks(name):
    mc_cb = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(GCONF.saved_model_path,'models/{epoch}'),
        monitor=f'avg_jaccard_{name}',
        mode="max",
        save_top_k=1,
        prefix=f'{name}_',
        save_weights_only=False
    )
    return mc_cb

def get_best_model_fn(mc_cb, fold_i):
    for k, v in mc_cb.best_k_models.items():
        if (v == mc_cb.best) and Path(k).stem.startswith(str(fold_i)):
            return k

def move_to_device(x, device):
  if callable(getattr(x, 'to', None)): return x.to(device)
  if isinstance(x, (tuple, list)): return [move_to_device(o, device) for o in x]
  elif isinstance(x, dict): return {k: move_to_device(v, device) for k, v in x.items()}
  return x

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterF(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
        self.answers = []
        self.preds = []

    def update(self, answer, pred):
        self.answers.extend(answer)
        self.preds.extend(pred)
        self.avg = f1_score(self.answers,self.preds,average='macro')

class TweetBertDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_testing = is_testing

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        query = self.df.iloc[ix]['text_a']

        input_ids_query = tokenizer.encode(query)
        if not self.is_testing:
            sentiment = self.df.iloc[ix]['label']

        input_ids = input_ids_query
        input_ids = input_ids[:-1]
        input_ids = input_ids[:self.max_length-1] + [1]
        attn_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        input_ids += [0] * pad_len
        attn_mask += [0] * pad_len
        token_type_ids += [0] * pad_len

        input_ids, attn_mask, token_type_ids = map(torch.LongTensor, [input_ids, attn_mask, token_type_ids])

        encoded_dict = {
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'token_type_ids': token_type_ids,

        }

        if not self.is_testing:
            tmp = [0] * 8
            for _ in sentiment:
              tmp[all_labels.index(_)] = 1
            encoded_dict['sentiment'] = torch.tensor(tmp, dtype=torch.long)

        return encoded_dict

class TweetBertModel(nn.Module):
    def __init__(self, roberta_path):
        super().__init__()

        roberta_config = BertConfig.from_pretrained(roberta_path)
        roberta_config.output_hidden_states = True
        self.roberta = BertModel.from_pretrained(roberta_path, config=roberta_config)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(roberta_config.hidden_size, 8)
        torch.nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, input_ids, attn_mask, token_type_ids):
        # transformer == 3.3.0
        # pooled_output, a, b = self.roberta(
        #     input_ids=input_ids,
        #     attention_mask=attn_mask,
        #     token_type_ids=token_type_ids
        # )

        # transformer == 4.4.0
        pooled_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )['last_hidden_state']
        pooled_output = self.dropout(pooled_output[:,0,:])
        start_logits = self.classifier(pooled_output)
        return start_logits,pooled_output


class TweetBert(pl.LightningModule):
    def __init__(self, roberta_path, fold_i=None, train_len=None):
        super().__init__()

        self.fold_i = fold_i
        self.train_len = train_len

        self.model = TweetBertModel(roberta_path)

        self.am_tloss = AverageMeter()
        self.am_vloss = AverageMeter()
        self.am_jscore = AverageMeterF()

    def _compute_loss(self, start_logits, batch):

        loss_fn = nn.MultiLabelSoftMarginLoss()
        start_loss = loss_fn(start_logits, batch['sentiment'])

        return start_loss

    def forward(self, batch):
        return self.model(batch['input_ids'], batch['attn_mask'], batch['token_type_ids'])

    def training_step(self, batch, batch_nb):
        start_logits, pooled_output = self.forward(batch)
        train_loss = self._compute_loss(start_logits, batch)
        self.am_tloss.update(train_loss.item(), len(batch['input_ids']))
        return {'loss': train_loss, 'log': {f'train_loss_{self.fold_i}': train_loss}}

    def validation_step(self, batch, batch_nb):
        start_logits, pooled_output = self.forward(batch)
        start_logits = start_logits.detach()

        valid_loss = self._compute_loss(start_logits, batch)
        m = nn.Softmax()
        pred = []
        for i in start_logits:
            tmp = [0] * 8
            i = m(i).tolist()
            for num_, _ in enumerate(i):
                if _ > 0.3:
                    tmp[num_] = 1
            pred.append(tmp)
        answer = []
        for i in batch['sentiment']:
            i = i.tolist()
            answer.append(i)
        self.am_vloss.update(valid_loss.item(), len(batch['input_ids']))
        self.am_jscore.update(answer, pred)

    def validation_epoch_end(self, _):
        tqdm_dict = {
            f'avg_train_loss_{self.fold_i}': self.am_tloss.avg,
            f'avg_valid_loss_{self.fold_i}': self.am_vloss.avg,
            f'avg_jaccard_{self.fold_i}': self.am_jscore.avg
        }
        print(self.am_jscore.avg)
        self.am_tloss.reset()
        self.am_vloss.reset()
        self.am_jscore.reset()
        return {'progress_bar': tqdm_dict, 'log': tqdm_dict}

    def predict(self, batch, device, mode='test'):
        self.eval()

        with torch.no_grad():
            batch = move_to_device(batch, device)
            start_logits, pooled_output = self.forward(batch)
            return start_logits.detach(), pooled_output.detach()

    def configure_optimizers(self):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 1e-3
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.
            }
        ]
        optimizer = optim.AdamW(optimizer_parameters, lr=GCONF.lr)
        train_steps = (self.train_len * GCONF.epochs) // GCONF.accum_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(GCONF.warmup_steps * train_steps),
            num_training_steps=train_steps
        )

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return [optimizer], [scheduler_dict]

GCONF = GlobalConfig()
import os
if not os.path.exists(GCONF.saved_model_path):
  os.mkdir(GCONF.saved_model_path)

seed_everything(GCONF.seed)
all_labels = ['经济', '科技', '其他', '社会', '环境', '军事', '文化', '政治']
label_num = [649, 106, 1319, 3193, 107, 48, 119, 1161]
label_weight = [log(6702/ i) for i in label_num]
print(label_weight)

from transformers import AutoTokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(GCONF.roberta_path, do_lower_case=True, add_prefix_space=True)


fold_i=0#@param

with open('train.json') as f:
  train_data = json.load(f)

train = []
for i in train_data:
  train.append(
      [i['body'],i['label']]
  )

# with open('pseduo_with_label_20201229_1.0.json') as f:
#   train_data = json.load(f)
#
# for i in train_data:
#   train.append(
#       [i['body'],i['label']]
#   )

train_df = pandas.DataFrame(train)
train_df.columns = ['text_a','label']

with open('dev.json') as f:
  dev_data = json.load(f)

dev = []
for i in dev_data:
  dev.append(
      [i['body'],i['label']]
  )
valid_df = pandas.DataFrame(dev)
valid_df.columns = ['text_a','label']
start = time.time()

print("Fold {} start".format(fold_i))
train_ds = TweetBertDataset(pd.concat([train_df]), tokenizer, GCONF.max_length, is_testing=False)
train_dl = DataLoader(train_ds, batch_size=GCONF.batch_size, shuffle=True)

valid_ds = TweetBertDataset(valid_df, tokenizer, GCONF.max_length, is_testing=False)
valid_dl = DataLoader(valid_ds, batch_size=GCONF.batch_size*4, shuffle=False)

mc_cb = get_callbacks(fold_i)
model = TweetBert(GCONF.roberta_path, fold_i, len(train_dl))

trainer = pl.Trainer(
    checkpoint_callback=mc_cb,
    gpus='0',
    max_epochs=GCONF.epochs,
    accumulate_grad_batches=GCONF.accum_steps,
)
trainer.fit(model, train_dl, valid_dl)
torch.cuda.empty_cache()

checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]
print(checkpoint_path)
trainer.restore(checkpoint_path, on_gpu=False)
model_path = f"fold{fold_i}_epoch{GCONF.epochs}.pt"
model_path = os.path.join(GCONF.saved_model_path,model_path)
print(f"saving model to {model_path}")
torch.save(model.model,model_path)

model.to('cpu')
del train_ds, valid_ds, model, trainer
gc.collect()
torch.cuda.empty_cache()
