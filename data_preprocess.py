import csv
import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, text, label=None):
        self.text = text
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):

    def get_examples(self, filepath):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        return data


class MyProcessor(DataProcessor):

    def get_examples(self, filepath):
        return self._create_examples(
            self._read_csv(filepath))

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, data):
        labels = {'消极': 0, '积极': 1}
        examples = []
        for row in data:
            # guid = "%s-%s" % (set_type, i)
            text = row[0]
            label = labels[row[1]]
            examples.append(
                InputExample(text=text, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=True):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        encode_dict = tokenizer.encode_plus(text=tokens,
                                            max_length=max_seq_length,
                                            pad_to_max_length=True,
                                            is_pretokenized=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        input_ids = encode_dict['input_ids']
        input_mask = encode_dict['attention_mask']
        segment_ids = encode_dict['token_type_ids']

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5 and show_exp:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

