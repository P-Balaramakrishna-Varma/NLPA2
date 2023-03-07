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




if __name__ == "__main__" and sys.argv[1] == "TPU":
    import torch_xla
    import torch_xla.core.xla_model as xm


# Main Abstraction
class PosTagDataset(Dataset):
    def __init__(self, data_file):
        self.vocab_index = get_vocab_index(data_file)
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
    total_loss, correct, total_pred = 0, 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.reshape(-1)
            pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
            loss = loss_fn(pred, y)
            
            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_pred += y.shape[0]
    return total_loss, (correct * 100) / total_pred 




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
                vocab_index[token["form"]] = 1
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
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


def main_distributed_GPU(rank, world_size):
    # Configuaration
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddp_setup(rank, world_size)
    device = torch.device("cuda", rank)



    # Hyperparameters
    embedding_dim = 128
    hidden_dim  = 128
    no_layers = 2
    
    epochs = 10
    batch_size = 32
    lr = 0.01




    # Loading data
    train_file = "./UD_English-Atis/en_atis-ud-train.conllu"
    train_dataset = PosTagDataset(train_file)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, collate_fn=custom_collate, sampler=DistributedSampler(train_dataset))

    dev_file = "./UD_English-Atis/en_atis-ud-dev.conllu"
    dev_dataset = PosTagDataset(dev_file)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False, collate_fn=custom_collate, sampler=DistributedSampler(dev_dataset))

    test_file = "./UD_English-Atis/en_atis-ud-test.conllu"
    test_dataset = PosTagDataset(test_file)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=custom_collate, sampler=DistributedSampler(test_dataset))




    # Creating model loss function and optimizer
    vocab_size = len(train_dataset.vocab_index)
    no_pos_tags = len(train_dataset.pos_tag_index)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    model = PosTagModel(vocab_size, no_pos_tags, embedding_dim, hidden_dim, no_layers).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)

    model = DDP(model, device_ids=[device])



  # Training
    for t in tqdm(range(epochs)):
        train_loop(model, loss_fn, optimizer, train_dataloader, device)
        train_metrics = eval_model(model, loss_fn, train_dataloader, device)
        dev_metrics = eval_model(model, loss_fn, dev_dataloader, device)
    print("Done!")
    destroy_process_group()    


def main__GPU():
    # Configuaration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # Hyperparameters
    embedding_dim = 128
    hidden_dim  = 128
    no_layers = 2
    
    epochs = 10
    batch_size = 32
    lr = 0.01




    # Loading data
    train_file = "./UD_English-Atis/en_atis-ud-train.conllu"
    train_dataset = PosTagDataset(train_file)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=custom_collate)

    dev_file = "./UD_English-Atis/en_atis-ud-dev.conllu"
    dev_dataset = PosTagDataset(dev_file)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=True, collate_fn=custom_collate)

    test_file = "./UD_English-Atis/en_atis-ud-test.conllu"
    test_dataset = PosTagDataset(test_file)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=custom_collate)




    # Creating model loss function and optimizer
    vocab_size = len(train_dataset.vocab_index)
    no_pos_tags = len(train_dataset.pos_tag_index)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    model = PosTagModel(vocab_size, no_pos_tags, embedding_dim, hidden_dim, no_layers).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)




  # Training
    for t in tqdm(range(epochs)):
        train_loop(model, loss_fn, optimizer, train_dataloader, device)
        train_metrics = eval_model(model, loss_fn, train_dataloader, device)
        dev_metrics = eval_model(model, loss_fn, dev_dataloader, device)
    print("Done!")
    #torch.save(model.state_dict(), 'model_weights.pth')


def main_tpu():
    # Configuaration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # Hyperparameters
    embedding_dim = 128
    hidden_dim  = 128
    no_layers = 2
    
    epochs = 10
    batch_size = 32
    lr = 0.01




    # Loading data
    train_file = "./UD_English-Atis/en_atis-ud-train.conllu"
    train_dataset = PosTagDataset(train_file)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=custom_collate)

    dev_file = "./UD_English-Atis/en_atis-ud-dev.conllu"
    dev_dataset = PosTagDataset(dev_file)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=True, collate_fn=custom_collate)

    test_file = "./UD_English-Atis/en_atis-ud-test.conllu"
    test_dataset = PosTagDataset(test_file)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=custom_collate)




    # Creating model loss function and optimizer
    vocab_size = len(train_dataset.vocab_index)
    no_pos_tags = len(train_dataset.pos_tag_index)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    model = PosTagModel(vocab_size, no_pos_tags, embedding_dim, hidden_dim, no_layers).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)






if __name__ == "__main__":
    if(sys.argv[1] == "distributed-gpu"):
        world_size = torch.cuda.device_count()
        print(world_size, "GPUs available")
        mp.spawn(main_distributed_GPU, args=(world_size,), nprocs=world_size)
    elif(sys.argv[1] == "gpu"):
        main__GPU()
    elif(sys.argv[1] == "tpu"):
        pass
    else:
        print("Invalid argument")