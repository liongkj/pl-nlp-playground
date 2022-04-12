import re
import unicodedata

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

SOS_token = 0
EOS_token = 1

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def filterPair(p,MAX_LENGTH):
    eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
    b1 = len(p[0].split(' ')) < MAX_LENGTH
    b2 =len(p[1].split(' ')) < MAX_LENGTH
    b3 = p[0].startswith(eng_prefixes)
    return  b1 and b2 and b3
        

def readVocab(lang1,lang2,reversed=False, MAX_LENGTH=10):
    

    lines = open("data/%s-%s.txt" % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    

   
    
    
    
    if reversed:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Vocab(lang2)
        output_lang = Vocab(lang1)
    else:
        input_lang = Vocab(lang1)
        output_lang = Vocab(lang2)
    pairs =  [pair for pair in pairs if filterPair(pair,MAX_LENGTH)]
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang,pairs

    

def sent2index(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def index2tensor(lang,sentence):
    index = sent2index(lang,sentence)
    index.append(EOS_token)
    return torch.tensor(index, dtype=torch.long).view(-1,1)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,input_lang,output_lang):
        self.input_lang,self.output_lang,self.dataset = readVocab(input_lang,output_lang)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        x, label = self.dataset[idx][:2]
        # print(example)
        return index2tensor(self.input_lang,x),index2tensor(self.output_lang,label)

class DataModule(pl.LightningDataModule):
    def __init__(self,input_lang,output_lang):
        super().__init__()
        self.dataset = MyDataset(input_lang,output_lang)
        self.input_lang,self.output_lang = self.dataset.input_lang,self.dataset.output_lang
        self.train_dataset, self.test_dataset = random_split(self.dataset, [int(len(self.dataset)*0.7), len(self.dataset)-int(len(self.dataset)*0.7)])
    
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,batch_size=1,shuffle=True)
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,batch_size=1,shuffle=True)
    
    # def val_dataloader(self):
        # return torch.utils.DataLoader(self.dataset,batch_size=1,shuffle=True)

# dm  = DataModule(input_lang = 'en',output_lang='ms')
