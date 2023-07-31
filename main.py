import math

from torch.utils.data import DataLoader
from model import *
from data_preprocess import *
from MyDataset import *
import matplotlib.pyplot as plt
import time
from transformers import BertTokenizer
from transformers import logging
from scipy.special import softmax
import argparse
import tqdm
import numpy as np

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')


def train(tokenizer, model, args, train_data, test_data):
    train_features = convert_examples_to_features(train_data, label_list, 128, tokenizer)
    test_features = convert_examples_to_features(test_data, label_list, 128, tokenizer)
    train_dataset = MyDataset(train_features, 'train')
    test_dataset = MyDataset(test_features, 'test')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    train_data_len = len(train_dataset)
    test_data_len = len(test_dataset)
    print(f"size of training set：{train_data_len}")
    print(f"size of validation set：{test_data_len}")

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    total_train_step = 0
    total_test_step = 0
    step = 0
    epoch = args.epoches

    train_loss_his = []
    train_totalaccuracy_his = []
    test_totalloss_his = []
    test_totalaccuracy_his = []
    start_time = time.time()
    model.train()
    best_test_loss = math.inf
    for i in range(epoch):
        print(f"-------Epoch {i + 1} starts-------")
        train_total_accuracy = 0
        train_total_loss = 0
        tbar = tqdm.tqdm(train_data_loader, desc='Epoch {}/{}, loss: {:.4f}'.format(i + 1, args.epoches, 0.0))
        for step, batch_data in enumerate(tbar):
            # writer.add_images("tarin_data", imgs, total_train_step)
            # print(batch_data['input_ids'].shape)
            output = model(**batch_data)
            loss = loss_fn(output, batch_data['label_id'])
            train_accuracy = (output.argmax(1) == batch_data['label_id']).sum()
            train_total_accuracy = train_total_accuracy + train_accuracy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step = total_train_step + 1
            train_total_loss += loss.item()
            train_loss_his.append(train_total_loss / (step + 1))
            tbar.set_description('Epoch {}/{}, loss: {:.4f}'.format(i + 1, args.epoches, train_total_loss / (step + 1)))
        train_total_accuracy = train_total_accuracy / train_data_len
        print(f"Accuracy on training set：{train_total_accuracy}")
        train_totalaccuracy_his.append(train_total_accuracy)
        # 测试开始
        print("Validating...")
        total_test_loss = 0
        model.eval()
        test_total_accuracy = 0
        with torch.no_grad():
            for batch_data in tqdm.tqdm(test_data_loader):
                output = model(**batch_data)
                loss = loss_fn(output, batch_data['label_id'])
                total_test_loss = total_test_loss + loss.item()
                test_accuracy = (output.argmax(1) == batch_data['label_id']).sum()
                test_total_accuracy = test_total_accuracy + test_accuracy
            test_total_accuracy = test_total_accuracy / test_data_len
            print(f"Accuracy on validation set：{test_total_accuracy}")
            print(f"Loss on validation set：{total_test_loss / len(test_data_loader)}")
            test_totalloss_his.append(total_test_loss / len(test_data_loader))
            test_totalaccuracy_his.append(test_total_accuracy)

            if total_test_loss < best_test_loss:
                torch.save(model.state_dict(), 'best.pth')
                best_test_loss = total_test_loss

            torch.save(model.state_dict(), 'epoch_{}.pth'.format(i+1))

    end_time = time.time()
    total_train_time = end_time - start_time
    total_train_time = time.strftime("%H:%M:%S", time.gmtime(total_train_time))
    print(f'total train time: {total_train_time}')

    plt.plot(train_loss_his, label='Train Loss')
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.savefig('train_loss.png')

    plt.figure()
    plt.plot(test_totalloss_his, label='Validation Loss')
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.savefig('valid_loss.png')

    plt.figure()
    plt.plot(train_totalaccuracy_his, label='Train accuracy')
    plt.plot(test_totalaccuracy_his, label='Validate accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epoches')
    plt.savefig('accuracy.png')


def test(tokenizer, model, args, test_data):
    test_features = convert_examples_to_features(test_data, label_list, 128, tokenizer)
    test_dataset = MyDataset(test_features, 'test')
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    test_data_len = len(test_dataset)
    print(f"size of testing set：{test_data_len}")

    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()

    print("Testing...")
    total_test_loss = 0
    model.eval()
    test_total_accuracy = 0
    with torch.no_grad():
        for batch_data in tqdm.tqdm(test_data_loader):
            output = model(**batch_data)
            loss = loss_fn(output, batch_data['label_id'])
            total_test_loss = total_test_loss + loss.item()
            test_accuracy = (output.argmax(1) == batch_data['label_id']).sum()
            test_total_accuracy = test_total_accuracy + test_accuracy
        test_total_accuracy = test_total_accuracy / test_data_len
        print(f"Accuracy on testing set：{test_total_accuracy}")
        print(f"Loss on testing set：{total_test_loss / len(test_data_loader)}")

    end_time = time.time()
    total_train_time = end_time - start_time
    total_train_time = time.strftime("%H:%M:%S", time.gmtime(total_train_time))
    print(f'total test time: {total_train_time}')


def predict(text, tokenizer, model):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(input_ids=encoded_input['input_ids'], input_mask=encoded_input['attention_mask'], segment_ids=encoded_input['token_type_ids'])
    scores = output.detach().numpy()
    scores = softmax(scores).squeeze(0)

    labels = ['negative', 'positive']
    result = {}
    for i in range(scores.shape[0]):
        l = labels[i]
        s = scores[i]
        print(f"{i + 1}) {l} {np.round(float(s), 4)}")
        result[l] = float(s)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action="store_true", help="Run training task")
    parser.add_argument('--test', default=False, action="store_true", help="Run testing task")
    parser.add_argument('--predict', default=False, action="store_true", help="Predict on a piece of text")
    parser.add_argument('--train_file', default='datasets/train.csv', type=str, help="File path of the training set")
    parser.add_argument('--valid_file', default='datasets/valid.csv', type=str, help="File path of the validation set")
    parser.add_argument('--test_file', default='datasets/test.csv', type=str, help="File path of the testing set")
    parser.add_argument('--lr', default=3e-5, type=float, help="Learning rate")
    parser.add_argument('--epoches', default=3, type=int, help="Number of epoches")
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size")
    parser.add_argument('--load_model', default='none', type=str, help="Path of pretrained model")
    parser.add_argument('--text', default="今天我真高兴", type=str, help="Text to be analyzed")
    args = parser.parse_args()

    logging.set_verbosity_warning()

    datadir = "data"
    my_processor = MyProcessor()
    label_list = my_processor.get_labels()

    train_data = my_processor.get_examples(args.train_file)
    valid_data = my_processor.get_examples(args.valid_file)
    test_data = my_processor.get_examples(args.test_file)

    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = ClassifierModel()
    if args.load_model != 'none':
        model.load_state_dict(torch.load(args.load_model))

    if args.train:
        train(tokenizer, model, args, train_data, valid_data)
    elif args.test:
        test(tokenizer, model, args, test_data)
    elif args.predict:
        result = predict(args.text, tokenizer, model)
    else:
        print("Wrong task!")
