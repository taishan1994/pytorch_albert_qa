import json
import os
import numpy as np

from tokenization import BasicTokenizer
from tqdm import tqdm
from transformers import BertTokenizer

from cut_sentence import cut_sent_for_bert, refactor_labels
from rematch import rematch

basicTokenizer = BasicTokenizer(do_lower_case=True)


class InputExample:
    def __init__(self,
                 question,
                 context,
                 answers,
                 doc_tokens=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 ):
        self.question = question
        self.context = context
        self.answers = answers
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return "context: " + self.context + '\n' + \
               "question: " + self.question + '\n' + \
               "start_position: " + str(self.start_position) + '\n' + \
               "end_postion: " + str(self.end_position) + '\n' + \
               "answer: " + "".join(self.answers)


class InputFeatures:
    def __init__(self,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


def read_qa_examples(path, input_file, max_seq_len, do_refactor=True):
    with open(os.path.join(path, input_file), 'r', encoding="utf-8") as f:
        input_data = json.load(f)

    examples = []
    for data in input_data['data']:
        for para in data['paragraphs']:
            context = para['context']
            qas = para['qas']
            for q in qas:
                question = q['question']
                answers = q['answers']
                start_position = []
                end_position = []
                answers_result = []
                if not do_refactor:
                    for answer in answers:
                        start_postion_tmp = answer['answer_start']
                        end_position_tmp = answer['answer_start'] + len(answer['text']) - 1
                        # print(question)
                        # print(context)
                        # print(answer['text'])
                        # print(context[start_postion:end_position+1])
                        if context[start_postion_tmp:end_position_tmp + 1] == answer['text']:
                            start_position.append(start_postion_tmp)
                            end_position.append(end_position_tmp)
                            answers_result.append(answer['text'])
                            # labels.append((start_postion_tmp, end_position_tmp, answer['text']))
                            examples.append(
                                InputExample(
                                    question=question,
                                    context=context,
                                    answers=answers_result,
                                    start_position=start_position,
                                    end_position=end_position
                                )
                            )
                else:
                    labels = []
                    for answer in answers:
                        start_postion_tmp = answer['answer_start']
                        end_position_tmp = answer['answer_start'] + len(answer['text'])
                        # print(question)
                        # print(context)
                        # print(answer['text'])
                        # print(context[start_postion:end_position+1])
                        if context[start_postion_tmp:end_position_tmp] == answer['text']:
                            # start_position.append(start_postion_tmp)
                            # end_position.append(end_position_tmp)
                            # answers_result.append(answer['text'])
                            # print(context[start_postion_tmp:end_position_tmp])
                            # print(answer['text'])
                            labels.append((start_postion_tmp, end_position_tmp, answer['text']))

                    start_index = 0
                    new_labels = []
                    token_seq_len = max_seq_len - len(question)
                    sentences = cut_sent_for_bert(context, token_seq_len)
                    for sent in sentences:
                        new_labels = refactor_labels(sent, labels, question, start_index)
                        # print("=" * 50)
                        # print(question)
                        # print(sent)
                        # print(len(sent))
                        # print(new_labels)
                        # print("=" * 50)
                        start_index += len(sent)
                        if new_labels:
                            for label in new_labels:
                                start_position.append(label[0])
                                end_position.append(label[1]-1)
                                answers_result.append(label[-1])

                        examples.append(
                            InputExample(
                                question=question,
                                context=sent,
                                answers=answers_result,
                                start_position=start_position,
                                end_position=end_position
                            )
                        )
                        start_position = []
                        end_position = []
                        answers_result = []

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 do_lower_case=True, pad_sign=True):
    features = []
    for (example_idx, example) in enumerate(tqdm(examples, ncols=100)):

        question_tokens = tokenizer.tokenize(example.question)
        whitespace_doc = basicTokenizer.tokenize(example.context)  # 先初步对context进行token化
        max_tokens_for_doc = max_seq_length - len(question_tokens) - 3

        all_doc_tokens = []

        for token_item in whitespace_doc:
            tmp_subword_lst = tokenizer.tokenize(token_item)
            all_doc_tokens.extend(tmp_subword_lst)

        if len(example.start_position) == 0 and len(example.end_position) == 0:
            doc_start_pos = [0] * len(all_doc_tokens)
            doc_end_pos = [0] * len(all_doc_tokens)
        else:
            doc_start_pos = [0] * len(all_doc_tokens)
            doc_end_pos = [0] * len(all_doc_tokens)
            mapping = rematch(example.context, all_doc_tokens, do_lower_case)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            tmp_start = []
            tmp_end = []

            for start in example.start_position:
                if start in start_mapping:
                    doc_start_pos[start_mapping[start]] = 1
                    tmp_start.append(start_mapping[start])
            for end in example.end_position:
                if end in end_mapping:
                    doc_end_pos[end_mapping[end]] = 1
                    tmp_end.append(end_mapping[end])
 

            # for start, end, answer in zip(tmp_start, tmp_end, example.answers):
            #     print(start, end, answer)
            #     print(all_doc_tokens[start:end+1])
            #     # print(tokenizer.decode(all_doc_tokens[start:end+1]))
            #     print(answer)
            #     print("".join(tokenizer.convert_tokens_to_string(all_doc_tokens[start:end+1]).split()))

        assert len(all_doc_tokens) == len(doc_start_pos)
        assert len(all_doc_tokens) == len(doc_end_pos)
        assert len(doc_start_pos) == len(doc_end_pos)

        if len(all_doc_tokens) >= max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[: max_tokens_for_doc]
            doc_start_pos = doc_start_pos[: max_tokens_for_doc]
            doc_end_pos = doc_end_pos[: max_tokens_for_doc]

        # if len(example.start_position) == 0 and len(example.end_position) == 0:
        #     doc_span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)

        # input_mask:
        #   the mask has 1 for real tokens and 0 for padding tokens.
        #   only real tokens are attended to.
        # segment_ids:
        #   segment token indices to indicate first and second portions of the inputs.

        input_tokens = []
        segment_ids = []
        input_mask = []
        start_pos = []
        end_pos = []

        input_tokens.append("[CLS]")
        segment_ids.append(0)
        start_pos.append(0)
        end_pos.append(0)

        for question_item in question_tokens:
            input_tokens.append(question_item)
            segment_ids.append(0)
            start_pos.append(0)
            end_pos.append(0)

        input_tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)
        start_pos.append(0)
        end_pos.append(0)

        input_tokens.extend(all_doc_tokens)
        segment_ids.extend([1] * len(all_doc_tokens))
        start_pos.extend(doc_start_pos)
        end_pos.extend(doc_end_pos)

        input_tokens.append("[SEP]")
        segment_ids.append(1)
        start_pos.append(0)
        end_pos.append(0)
        input_mask = [1] * len(input_tokens)

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

        # zero-padding up to the sequence length
        if len(input_ids) < max_seq_length and pad_sign:
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            start_pos += padding
            end_pos += padding
        
        # 如不存在答案的都设置为0
        start_position = np.where(np.array(start_pos) == 1)[0][0] if np.where(np.array(start_pos) == 1)[0] else 0
        end_position = np.where(np.array(end_pos) == 1)[0][0] if np.where(np.array(end_pos) == 1)[0] else 0

        if example_idx < 3:
            print("=" * 100)
            print("example.context:", example.context)
            print("tokens:", input_tokens)
            print("input_ids:", input_ids)
            print("input_mask:", input_mask)
            print("segment_ids:", segment_ids)
            print("start_position:", start_position)
            print("end_position:", end_position)
            print("answer:", "".join(example.answers))
            st_pos = np.where(np.array(start_pos) == 1)[0]
            en_pos = np.where(np.array(end_pos) == 1)[0]
            print(st_pos, en_pos)
            if st_pos:
                for st, en in zip(st_pos, en_pos):
                    print("parse answer:", "".join(input_tokens[st:en+1]))
            if start_position:
              print("parse answer2: ", "".join(input_tokens[start_position:end_position+1]))
            print("=" * 100)

        features.append(
            InputFeatures(
                tokens=input_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
            ))

    return features


if __name__ == '__main__':
    path = 'cmrc2018'
    input_file = "cmrc2018_train_squad.json"
    max_seq_length = 512
    examples = read_qa_examples(path, input_file, max_seq_length)
    # for e in examples:
    #     print("=" * 100)
    #     print(e)
    #     print("=" * 100)
    tokenizer = BertTokenizer.from_pretrained('./model_hub/voidful-albert-chinese-tiny/')
    convert_examples_to_features(examples, tokenizer, max_seq_length)
