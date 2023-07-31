from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import argparse
import pymysql
import torch
from model import ClassifierModel

parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str, default='I love you')
args = parser.parse_args()


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

def init():
    labels = ['negative', 'positive']
    model = ClassifierModel()
    model.load_state_dict(torch.load('best.pth'))
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    return model, tokenizer, labels

def analyze(model, tokenizer, labels, text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(input_ids=encoded_input['input_ids'], input_mask=encoded_input['attention_mask'], segment_ids=encoded_input['token_type_ids'])
    scores = output.detach().numpy()
    scores = softmax(scores).squeeze(0)

    result = {}
    for i in range(scores.shape[0]):
        l = labels[i]
        s = scores[i]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
        result[l] = float(s)

    return result

def insertIntoDB(db, cursor, id, result):
    try:
        update = "UPDATE WEIBO SET positive={}, negative={} WHERE tweet_id='{}'".format(result['positive'], result['negative'], id)
        print(update)
        try:
            cursor.execute(update)
            db.commit()
        except:
            print("update error!")
            db.rollback()
    except:
        print("Fetch data error!")

    db.close()

if __name__ == '__main__':
    model, tokenizer, labels = init()
    db = pymysql.connect(host='host name', user='username', password='password', database='database')
    cursor = db.cursor()
    sql = "SELECT mid, text FROM WEIBO WHERE keyword='港大' AND positive IS NULL LIMIT 0, 100"
    try:
        cursor.execute(sql)
        res = cursor.fetchall()
        if len(res) > 0:
            for item in res:
                id = item[0]
                print("processing: ", id)
                text = item[1]
                try:
                    result = analyze(model, tokenizer, labels, text)
                    insertIntoDB(db, cursor, id, result)
                    print(id, result)

                except:
                    print("Analyze error")

        else:
            flag = False

    except:
        print("Fetch error")

    sql = "SELECT mid, text FROM WEIBO WHERE keyword='香港大学' AND positive IS NULL LIMIT 0, 100"
    try:
        cursor.execute(sql)
        res = cursor.fetchall()
        if len(res) > 0:
            for item in res:
                id = item[0]
                print("processing: ", id)
                text = item[1]
                try:
                    result = analyze(model, tokenizer, labels, text)
                    insertIntoDB(db, cursor, id, result)
                    print(id, result)

                except:
                    print("Analyze error")

        else:
            flag = False

    except:
        print("Fetch error")

    db.close()
