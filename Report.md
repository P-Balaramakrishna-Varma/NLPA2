# Architecture Choice
I have chosen to use BiLSTM among RNN, LSTM, GRU, BiLSTM, BiRNN. The reason being BiLSTM takes of long dependencies and it takes both the forward and backward part of context.    

Architecture: Vocab_index --> word_embedding --> BiLSTM --> Linear --> ReLU

The model has three architectural hyperparameters **word embedding dimension**,  **hidden dimension**, **number of BiLSTM layers**.


<br>
<br>
<br>

# Training observation
word_embedding_dim | hidden_dim | no_BiLSTM | accuray |
-------------------|------------|---------|----------|
100 | 100 | 1 | 98.48
100 | 200 | 1 | 98.46
100 | 300 | 1 | 98.46
200 | 200 | 1 | 98.05
200 | 400 | 1 | 98.58
200 | 600 | 1 | 67.15
300 | 300 | 1 | 75.73
300 | 600 | 1 | 98.7
300 | 900 | 1 | 95.0


<br>

## No_BiLSTM layers
After experiments like 2 BiLSTM vs **1 BiLSTM tell that 1 BiLSTM model does as good as 2 BiLSTM or 4 BiLSTM**.

<br>

## word_embedding and hidden_dim
- when ```word_dim = hidden_dim``` we have okish performance for all size of word_dim.
- when ```word_dim = 2 * hidden_dim``` we have the best performance for all size of word_dim
- when ```word_dim = 3 * hiddem_dim``` we have the worst performance for all sizes of word_dim (may be due to to much contex and less data)

## Word embeddings
- The accuracy increases as the word embedding dimension increases.