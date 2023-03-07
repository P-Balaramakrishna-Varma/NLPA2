import sys
import pickle
import torch
from nltk.tokenize import word_tokenize

import neural_tag as helper





## Hyperparameters make sure same as the one once used for the stored model
#voc = helper.get_vocab_index("./UD_English-Atis/en_atis-ud-train.conllu")
#pickle.dump(voc, open( "save.p", "wb" ) )
vocabulary = pickle.load( open( "save.p", "rb" ) )

pos_tag_index = {"Pad": 0, "ADJ": 17, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16}
embedding_dim = 128
hidden_dim  = 128
no_layers = 2

model = helper.PosTagModel(len(vocabulary), len(pos_tag_index), embedding_dim, hidden_dim, no_layers)
model.load_state_dict(torch.load('model_weights.pth'))


while(1):
    sentence_str = input("Enter a sentence: ")
    #tokenize
