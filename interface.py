import sys
import pickle
import torch
from torch.nn import Softmax
import string
from nltk.tokenize import word_tokenize

import neural_tag as helper





## Hyperparameters make sure same as the one once used for the stored model
voc = helper.get_vocab_index("./UD_English-Atis/en_atis-ud-train.conllu")
pickle.dump(voc, open( "save.p", "wb" ) )
vocabulary = pickle.load( open( "save.p", "rb" ) )
pos_tag_index = {"Pad": 0, "ADJ": 17, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16}
tag_pos_index = {0: "Pad", 17: "ADJ", 1: "ADP", 2: "ADV", 3: "AUX", 4: "CCONJ", 5: "DET", 6: "INTJ", 7: "NOUN", 8: "NUM", 9: "PART", 10: "PRON", 11: "PROPN", 12: "PUNCT", 13: "SCONJ", 14: "SYM", 15: "VERB", 16: "X"}
embedding_dim = 200
hidden_dim  = 300
no_layers = 1

model = helper.PosTagModel(len(vocabulary), len(pos_tag_index), embedding_dim, hidden_dim, no_layers)
model.load_state_dict(torch.load('model_weights.pth'))
sft_max = Softmax(dim=1)


### reading input processing input etc
sentence_str = input("Enter a sentence: ")
tokenized_sent = word_tokenize(sentence_str)
words = [word for word in tokenized_sent if word not in set(string.punctuation)]
words_indexed = []
for word in words:
    if word in vocabulary:
        words_indexed.append(vocabulary[word])
    else:
        words_indexed.append(vocabulary["<unk>"])
input_to_model = torch.LongTensor(words_indexed)
pred = model(input_to_model)
pred = sft_max(pred)

assigned_tags_idx = torch.argmax(pred, dim=1)
assigned_tags_idx = assigned_tags_idx.tolist()
assigned_tags = [tag_pos_index[assigned] for assigned in assigned_tags_idx]
for i in range(len(words)):
    print(words[i], "\t", assigned_tags[i])
