import os
import argparse

import torch
from torch import nn, optim

from allennlp.data import Vocabulary
from allennlp.data.iterators.bucket_iterator import BucketIterator

from data import BpeTokenizer, MorphTokenizer, KorDatasetReader
from model import CNNSentenceClassifier


def test(model, generator, device):
    total_corrects, total_instances, step = 0, 0, 0

    model.eval()
    for batch in generator:
        step += 1
        batch_size = batch['sentence']['tokens'].size()[0]
        total_instances += batch_size

        inputs = batch['sentence']['tokens'].to(device)
        outputs = model(inputs)

        labels = batch['label'].to(device)
        _, predicts = torch.max(outputs, dim=1)
        total_corrects += (predicts == labels).type(torch.LongTensor).sum()
    acc = float(total_corrects)/total_instances
    return acc

def train(opt):
    if (opt['token'] == 'bpe'):
        tokenizer = BpeTokenizer()
    elif (opt['token'] == 'morph'):
        tokenizer = MorphTokenizer()
    else:
        print('invalid token unit')
        tokenizer = None

    # Dataset Reader
    train_reader = KorDatasetReader(tokenizer=tokenizer)
    valid_reader = KorDatasetReader(tokenizer=tokenizer, tqdm=False)
    print("Reading instances from lines in file at: %s", opt['train_data'])
    train_instances = train_reader.read(opt['train_data'])
    print("Reading instances from lines in file at: %s", opt['valid_data'])
    valid_instances = valid_reader.read(opt['valid_data'])

    # Load or Build Vocab
    if os.path.isdir(opt['vocab_dir']) and os.listdir(opt['vocab_dir']):
        print("Loading Vocabulary")
        train_reader.set_total_instances(opt['train_data'])
        valid_reader.set_total_instances(opt['valid_data'])
        vocab = Vocabulary.from_files(opt['vocab_dir'])
    else:
        print("Building Vocabulary")
        vocab = Vocabulary.from_instances(train_instances)
        vocab.save_to_files(opt['vocab_dir'])

    # Iterator
    print("Making Iterators")
    train_iterator = BucketIterator(sorting_keys=[("sentence", "num_tokens")],
                                    batch_size=opt['batch_size'],
                                    track_epoch=True)
    valid_iterator = BucketIterator(sorting_keys=[("sentence", "num_tokens")],
                                    batch_size=opt['batch_size'],
                                    track_epoch=True)
    train_iterator.vocab = vocab
    valid_iterator.vocab = vocab
    train_generator = train_iterator(train_instances, num_epochs=opt['max_epoch'], shuffle=True)

    # Model
    print("Building Model")
    device = torch.device(opt['device'])
    model = CNNSentenceClassifier(opt, vocab).to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    print(f'number of parameters: {num_params}')
    optimizer = optim.Adadelta(params, lr=opt['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    epoch, step = 0, 0
    corrects, num_instance = 0, 0

    print("Training Start!")
    for batch in train_generator:
        if batch['epoch_num'][0] + 1 > epoch:
            if epoch != 0:
                train_acc = float(corrects) / num_instance
                valid_generator = valid_iterator(valid_instances, num_epochs=1)
                valid_acc = test(model, valid_generator, device)
                print(f'epoch {epoch}: '
                      f'train_acc {train_acc:.4f} '
                      f'valid_acc {valid_acc:.4f}')
                corrects, num_instance = 0, 0

            epoch = batch['epoch_num'][0] + 1
            step = 0
            print(f'Epoch {epoch} start')
        step += 1
        model.train()

        batch_size = batch['sentence']['tokens'].size()[0]
        inputs = batch['sentence']['tokens'].to(device)
        outputs = model(inputs)

        labels = batch['label'].to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(params, max_norm=opt['grad_clip'])
        optimizer.step()

        num_instance += batch_size
        _, predicts = torch.max(outputs, dim=1)
        corrects += (predicts == labels).type(torch.LongTensor).sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default="morph", help="available: morph and bpe")
    parser.add_argument("--train_data", default="data/NSMC/train.txt")
    parser.add_argument("--valid_data", default="data/NSMC/test.txt")
    parser.add_argument("--vocab_dir", default="data/NSMC/vocab")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--emb_dim", default=300, type=int)
    parser.add_argument("--mode", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--max_epoch", default=300, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="learning rate")
    parser.add_argument("--dropout_rate", default=0.5, type=float, help="dropout rate")
    parser.add_argument("--grad_clip", default=3.0, type=float, help="gradient norm limitation")
    parser.add_argument("--batch_size", default=50, type=int, help="batch size")

    args = parser.parse_args()
    opt = vars(args)
    opt['FILTER_SIZES'] = [3, 4, 5]
    train(opt)