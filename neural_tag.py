import conllu
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os, sys
import numpy
from sklearn.metrics import classification_report


# Main Abstraction
## vocabindex to created from traning file
class PosTagDataset(Dataset):
    def __init__(self, data_file, vocab_index):
        self.vocab_index = vocab_index
        self.pos_tag_index = {"Pad": 0, "ADJ": 17, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16}
        self.Sentences, self.Tag_Sequences = get_data(data_file, self.vocab_index, self.pos_tag_index)

    def __len__(self):
        return len(self.Sentences)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.Sentences[idx]), torch.LongTensor(self.Tag_Sequences[idx])
    

class PosTagModel(torch.nn.Module):
    def __init__(self, vocab_size, targe_size, embedding_dim, hidden_dim, no_layers):
        super().__init__()

        # Embeding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        # BLSTM layer
        self.blstm = torch.nn.LSTM(embedding_dim, hidden_dim, no_layers, batch_first=True, bidirectional=True)

        # Output layer (*2 because of bidirectional)
        self.out_linear = torch.nn.Linear(hidden_dim * 2, targe_size)
        self.out_activation = torch.nn.ReLU()

    def forward(self, X):
        X = self.embedding(X)
        
        X, _ = self.blstm(X)

        X = self.out_linear(X)
        X = self.out_activation(X)
        return X
        

def train_loop(model, loss_fn, optimizer, train_dataloader, device):
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        # Getting data
        X, y = X.to(device), y.to(device)
        
        # Forward pass and loss
        pred = model(X)               
        y = y.reshape(-1)
        pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
        loss = loss_fn(pred, y)
    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def eval_model(model, loss_fn, data_loader, device):
    model.eval()
    y_true_report = torch.ones((1,)).to(device)
    y_pred_report = torch.ones((1,)).to(device)
    total_loss, correct, total_pred = 0, 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.reshape(-1)
            pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
            loss = loss_fn(pred, y)
            
            total_loss += loss.item()
            mask = y != 0
            correct += (pred.argmax(1)[mask] == y[mask]).type(torch.float).sum().item()
            total_pred += y[mask].shape[0]
            y_true_report = torch.cat((y_true_report, y[mask]), 0)
            y_pred_report = torch.cat((y_pred_report, pred.argmax(1)[mask]), 0)  
    return total_loss/ total_pred, (correct * 100) / total_pred, y_true_report.to(torch.device("cpu")), y_pred_report.to(torch.device("cpu")) 




# Helper Functions
def get_data(data_file, vocab_index, pos_tag_index):
    TokenLists = conllu.parse_incr(open(data_file, "r", encoding="utf-8"))
    Sentences = []
    Tag_Sequences = []
    for TokenList in TokenLists:
        Sentence = []
        tags = []
        for token in TokenList:
            if token["form"] in vocab_index:
                Sentence.append(vocab_index[token["form"]])
            else:
                Sentence.append(vocab_index["<unk>"])
            tags.append(pos_tag_index[token["upos"]])
        Sentences.append(Sentence)
        Tag_Sequences.append(tags)
    return Sentences, Tag_Sequences


def get_vocab_index(data_file):
    vocab_index = {"pad": 0, "<unk>": 1}
    sigletons = {}
    TokenLists = conllu.parse_incr(open(data_file, "r", encoding="utf-8"))
    for TokenList in TokenLists:
        for token in TokenList:
            if token["form"] not in sigletons:
                sigletons[token["form"]] = 1
            else:
                if token["form"] not in vocab_index:
                    vocab_index[token["form"]] = len(vocab_index)
    return vocab_index


def custom_collate(batch):
    Sentences = [sample[0] for sample in batch]
    PosTags = [sample[1] for sample in batch]

    Sentences = pad_sequence(Sentences, batch_first=True)
    PosTags = pad_sequence(PosTags, batch_first=True)
    return Sentences, PosTags




## Running Environment for code
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12540'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


def main_distributed_GPU(rank, world_size, hyper_params, qeue, Event, Store):
    # Configuaration
    ddp_setup(rank, world_size)
    device = torch.device("cuda", rank)



    # Hyperparameters
    embedding_dim = hyper_params["embedding_dim"]
    hidden_dim  = hyper_params["hidden_dim"]
    no_layers = hyper_params["no_layers"]
    
    epochs = hyper_params["epochs"]
    batch_size = hyper_params["batch_size"]
    lr = hyper_params["lr"]




    # Loading data
    train_file = "./UD_English-Atis/en_atis-ud-train.conllu"
    vocab_index = get_vocab_index(train_file)
    train_dataset = PosTagDataset(train_file, vocab_index)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, collate_fn=custom_collate, sampler=DistributedSampler(train_dataset))

    dev_file = "./UD_English-Atis/en_atis-ud-dev.conllu"
    dev_dataset = PosTagDataset(dev_file, vocab_index)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False, collate_fn=custom_collate, sampler=DistributedSampler(dev_dataset))

    test_file = "./UD_English-Atis/en_atis-ud-test.conllu"
    test_dataset = PosTagDataset(test_file, vocab_index)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=custom_collate, sampler=DistributedSampler(test_dataset))




    # Creating model loss function and optimizer
    vocab_size = len(train_dataset.vocab_index)
    no_pos_tags = len(train_dataset.pos_tag_index)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    model = PosTagModel(vocab_size, no_pos_tags, embedding_dim, hidden_dim, no_layers).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)

    model = DDP(model, device_ids=[device])



  # Training
    loss_values = torch.zeros((epochs, 2))
    for t in tqdm(range(epochs)):
        train_loop(model, loss_fn, optimizer, train_dataloader, device)
        loss_values[t, 0] = eval_model(model, loss_fn, train_dataloader, device)[0]   # Training Loss
        loss_values[t, 1] = eval_model(model, loss_fn, dev_dataloader, device)[0]     # validation Loss
    qeue.put(loss_values)
    Event.wait()

    test_eval = eval_model(model, loss_fn, test_dataloader, device)
    print("Testing accuracy", test_eval[1])
    print(classification_report(test_eval[2].numpy(), test_eval[3].numpy()))

    if rank == 0 and Store == True:
        param_data = model.module.state_dict()
        torch.save(param_data, "model_weights.pth")
    destroy_process_group()    


def get_loss_values(hyperpar, Save):
    world_size = torch.cuda.device_count()
    print("Number of GPUs: ", world_size)
    Events = [mp.Event() for _ in range(world_size)]
    qeue = mp.SimpleQueue()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=main_distributed_GPU, args=(rank, world_size, hyperpar, qeue, Events[rank], Save))
        processes.append(p)
        p.start()
    
    Data = torch.zeros((hyperpar["epochs"], 2))
    for _ in range(world_size):
        Data += qeue.get()
    Data = Data / world_size

    for event in Events:
        event.set()
    for p in processes:
        p.join()

    return Data


if __name__ == "__main__":
    hyperpar = {"embedding_dim": 128, "hidden_dim": 128, "no_layers": 2, "epochs": 10, "batch_size": 32, "lr": 0.01}
    Data = get_loss_values(hyperpar, False)
    print(Data)