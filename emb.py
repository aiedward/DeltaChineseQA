import tensorflow as tf
import json
import jieba 
import numpy as np
import pickle
import sys


def test_process_file(filename):
    examples = []
    
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in source["data"]:
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = [token for token in jieba.cut(context,cut_all=False)]
                context_chars = [list(token) for token in context_tokens]
                
                current = 0
                spans = []
                for token in context_tokens:
                    current = context.find(token, current)
                    spans.append((current, current + len(token)))
                    current += len(token)
                
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    #ques_tokens = word_tokenize(ques)
                    ques_tokens = [token for token in jieba.cut(ques,cut_all=False)]
                    ques_chars = [list(token) for token in ques_tokens]
                    
                    y1s, y2s = [], []
                    answer_texts = []
                    
                    answer_text = '123'
                    answer_start = 0
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    
        
    return examples


def make_record(examples, out_file, word2idx_dict, char2idx_dict):

    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    for example in examples:
        total_ += 1

        if len(example["context_tokens"]) > 1000 or len(example["ques_tokens"]) > 100:
            continue

        total += 1
        context_idxs = np.zeros([1000], dtype=np.int32)
        context_char_idxs = np.zeros([1000, 16], dtype=np.int32)
        ques_idxs = np.zeros([100], dtype=np.int32)
        ques_char_idxs = np.zeros([100, 16], dtype=np.int32)
        y1 = np.zeros([1000], dtype=np.float32)
        y2 = np.zeros([1000], dtype=np.float32)

        for i, token in enumerate(example["context_tokens"]):
            
            context_idxs[i] = word2idx_dict[token] if token in word2idx_dict else 1
            

        for i, token in enumerate(example["ques_tokens"]):
            
            ques_idxs[i] = word2idx_dict[token] if token in word2idx_dict else 1

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == 16:
                    break
                
                context_char_idxs[i, j] = char2idx_dict[char] if char in char2idx_dict else 1

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == 16:
                    break
                
                ques_char_idxs[i, j] = char2idx_dict[char] if char in char2idx_dict else 1

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0
        g = tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                                  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                                  "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                                  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                                  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                                  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                  })
        record = tf.train.Example(features=g)
        writer.write(record.SerializeToString())
    writer.close()




def main(argv):
     
    with open('./embeddings/char_emb.pickle', 'rb') as handle:
        ce = pickle.load(handle)
    with open('./embeddings/word_emb.pickle', 'rb') as handle:
        we = pickle.load(handle)
    
    test_examples = test_process_file(argv[0])

    make_record(test_examples,"./data/test.tfrecords", we[0], ce[0])

    

if __name__ == '__main__':
    main(sys.argv[1:])




