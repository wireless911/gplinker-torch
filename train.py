import json
from transformers import BertTokenizer, BertTokenizerFast, BertModel
from utils.data import MyDataset
from model.model import GlobalPointer, GPLinker


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object'])
                             for spo in l['spo_list']]
            })
    return D


# 加载数据集
train_data = load_data('datasets/SKE/train_data.json')
valid_data = load_data('datasets/SKE/dev_data.json')

predicate2id, id2predicate = {}, {}

with open('datasets/SKE/all_50_schemas', encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert_encoder = BertModel.from_pretrained("bert-base-chinese")
train_dataset = MyDataset(data=train_data, tokenizer=tokenizer)

model = GPLinker(encoder=bert_encoder, predicate2id=predicate2id)

pass
