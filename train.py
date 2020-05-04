# Much of the code is from https://github.com/pytorch/examples/tree/master/word_language_model

import argparse
import time
import math
import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import Dictionary, QueryDataset
from model import LSTMModel

parser = argparse.ArgumentParser(description='PyTorch AOL Queries LSTM & Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/aol/full',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of model: (1) LSTM, (2) Transformer')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.005,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--load_latest', action='store_true',
                    help='load latest trained model')
parser.add_argument('--nhead', type=int, default=4,
                    help='the number of heads in the encoder/decoder of the transformer model')

def batchify(data, bsz, nchar):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print(data.shape)
    data = nn.functional.one_hot(data, num_classes=nchar)

    return data.to(device)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def save(model, path):
    if isinstance(model, LSTMModel):
        name = "lstm_" + str(int(time.time())) + ".pkl"
    path = os.path.join(path, name)
    with open(path, 'wb') as f:
        torch.save(model, f)

def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    nchar = len(dictionary)
    if args.model == "LSTM":
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i, (data, target, lengths) in enumerate(valid_dataloader):
            batch_size = data.size()[0]
            data = torch.transpose(data, 0, 1).to(device)
            target = target.to(device)
            lengths = lengths.to(device)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, nchars)
            else:
                output, hidden = model(data, hidden, lengths)
                output = torch.flatten(output, 0, 1)
                target = torch.flatten(target)
                hidden = repackage_hidden(hidden)

            total_loss += criterion(output, target).item()

    return total_loss / len(valid_data)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.0
    start_time = time.time()
    log_interval = batch_size * 10

    if args.model == "LSTM":
        hidden = model.init_hidden(batch_size)

    for i, (data, target, lengths) in enumerate(train_dataloader):
        target = target.to(device)
        # convert tensor from (B x T x V) --> (T x B x V)
        data = torch.transpose(data, 0, 1).to(device)
        lengths = lengths.to(device)
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, nchar)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden, lengths)
            output = torch.flatten(output, 0, 1)
            target = torch.flatten(target)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.4f} | ppl {:8.4f}'.format(
                epoch, i, len(train_data) // batch_size + 1, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        del data
        del target
        del lengths

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")
    
    print("Arguments: \n ", args)
    print("Device:", device)

    query_files = [
        os.path.join(args.data, "train.query.txt"),
        os.path.join(args.data, "valid.query.txt"),
        os.path.join(args.data, "test.query.txt")
    ]

    if os.path.exists("./saved/dictionary.pkl"):
        print("Loading previously saved dictionary...")
        with open("./saved/dictionary.pkl", "rb") as f:
            dictionary = pickle.load(f)
    else:
        print("Creating dictionary...")
        dictionary = Dictionary(query_files)
        with open("./saved/dictionary.pkl", "wb") as f:
            pickle.dump(dictionary, f)
    
    nchar = len(dictionary)
    max_seq_len = dictionary.max_seq_len

    lr = args.lr
    clip = args.clip
    batch_size = args.batch_size
    eval_batch_size = 10
    best_val_loss = None

    if args.model == 'LSTM':
        model = LSTMModel(nchar, args.nhid, args.nlayers, max_seq_len, args.dropout)
        if args.load_latest:
            latest = max([f for f in os.listdir("./saved/lstm")])
            latest_path = os.path.join("./saved/lstm", latest)
            model = model.load_state_dict(torch.load(latest_path))
        model = model.to(device)

    save(model, args.save)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print("Start training...")
        for epoch in tqdm(range(1, args.epochs+1)):
            train_data = QueryDataset(query_files[0], dictionary)
            train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=2)

            valid_data = QueryDataset(query_files[1], dictionary)
            valid_dataloader = DataLoader(valid_data, batch_size=10, num_workers=2)

            epoch_start_time = time.time()
            print(f"{'=' * 40} Epoch {epoch} {'=' * 40}")
            train()
            save(model, args.save)
            val_loss = evaluate()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} |'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(save_name, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()
    
    test_data = QueryDataset(query_files[2], dictionary)
    test_dataloader = DataLoader(test_data, batch_size=10, num_workers=2)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
