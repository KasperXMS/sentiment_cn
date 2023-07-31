# Chinese Text Sentiment Analysis Model

## For HKU Projects

This is the source code of our sentiment analysis model, fine-tuned based on `hfl/chinese-roberta-wwm-ext` model and trained with a Weibo comment dataset (`all.csv` in the repository).

The model receive text as input, and output a list `[float32, float32]`, corresponding to positive and negative probability.

Pretrained weights from Huggingface, see: https://huggingface.co/hfl/chinese-roberta-wwm-ext



### Environment and Dependency

Python >= 3.6

CUDA ~= 11.4

torch >= 1.6.0

transformers

tqdm

matplotlib

scipy

argparse



### Performance

trained with different number of epochs, the performance is seen as in the following table:

| model       | epoch | acc  | prec-N | prec-P | recall-N | recall-P | F1-N | f1-P |
| ----------- | ----- | ---- | ------ | ------ | -------- | -------- | ---- | ---- |
| roberta-wwm | 3     | 86.6 | 86.6   | 86.6   | 86.6     | 86.6     | 86.6 | 86.6 |
| roberta-wwm | 6     | 86.9 | 85.6   | 88.4   | 88.8     | 85       | 87.2 | 86.7 |
| roberta-wwm | 9     | 87.5 | 89     | 86.1   | 85.6     | 89.4     | 87.3 | 87.7 |
| roberta-wwm | 12    | 86.7 | 84.7   | 89     | 89.6     | 83.8     | 87.1 | 86.3 |

Selection of number of epochs depends on your demand for overall accuracy / recall on specific label.



#### **Recommended Hyperparamer**

learning rate: 1e-4

Max_seq_length: 128

Batch_size: depends on your GPU memory



### How to run

Train:

```shell
python main.py --train --train_file "training set filepath" --valid_file "validation set filepath" --lr "learning_rate" --epoches "number_of_epochs" --batch_size "batch size" --load_model "model path if you want to start your training from a checkpoint"
```



Test:

```shell
python main.py --test test_file "testing set filepath" --batch_size "batch size" --load_model "model path"
```



Predict a single piece of text:

```shell
python main.py --predict --load_model "model path" --text "text"
```

