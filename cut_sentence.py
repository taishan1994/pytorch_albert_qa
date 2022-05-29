# coding=utf-8
"""
该模块主要是对句子进行切割并重组，
具体可查看main下面的样例
"""
import re


def cut_sentences_v1(sent):
    """
    the first rank of sentence cut
    """
    sent = re.sub('([。！？，\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    return sent.split('\n')


def cut_sentences_v2(sent):
    """
    the second rank of spilt sentence, split '；' | ';'
    """
    sent = re.sub('([；;])([^”’])', r"\1\n\2", sent)
    return sent.split("\n")


def cut_sentences_v3(sent):
    """
    将一段文本切分成多个句子
    :param sentence: ['虽然BillRoper正忙于全新游戏
    :return: ['虽然BillRoper正..接近。' , '与父母，之首。' , '很多..常见。' , '”一位上..推进。' , ''”一直坚..市场。'' , '如今，...的70%。']
    """
    new_sentence = []
    sen = []
    for i in sent:  # 虽
        sen.append(i)
        if i in ['。', '！', '？', '?']:
            new_sentence.append("".join(sen))  # ['虽然BillRoper正...接近。' , '与父母，...之首。' , ]
            sen = []

    if len(new_sentence) <= 1:  # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
        new_sentence = []
        sen = []
        for i in sent:
            sen.append(i)
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                new_sentence.append("".join(sen))
                sen = []

    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))

    return new_sentence


def cut_sent_for_bert(text, seq_len):
    # 将句子分句，细粒度分句后再重新合并
    # sentences = []
    #
    # # 细粒度划分
    # sentences_v1 = cut_sentences_v1(text)
    # # print("sentences_v1=", sentences_v1)
    # for sent_v1 in sentences_v1:
    #     if len(sent_v1) > seq_len:
    #         sentences_v2 = cut_sentences_v2(sent_v1)
    #         sentences.extend(sentences_v2)
    #     else:
    #         sentences.append(sent_v1)
    #

    sentences = cut_sentences_v3(text)

    if ''.join(sentences) != text:
        print(sentences)
        print(text.split('\n'))
        assert ''.join(sentences) == text

    # 合并
    merged_sentences = []
    start_index_ = 0

    while start_index_ < len(sentences):
        tmp_text = sentences[start_index_]

        end_index_ = start_index_ + 1
        while end_index_ < len(sentences) and \
                len(tmp_text) + len(sentences[end_index_]) <= seq_len:
            tmp_text += sentences[end_index_]
            end_index_ += 1

        start_index_ = end_index_

        merged_sentences.append(tmp_text)

    return merged_sentences


def refactor_labels(sent, labels, question, start_index):
    """
    分句后需要重构 labels 的 offset
    :param sent: 切分并重新合并后的句子
    :param labels: 原始文档级的 labels
    :param start_index: 该句子在文档中的起始 offset
    :return (start, end, answer)
    """
    new_labels = []
    end_index = start_index + len(sent)
    # _label： 答案起始位置， 答案结束位置, 答案
    for _label in labels:
        if start_index <= _label[0] <= _label[1] <= end_index:
            new_offset = _label[0] - start_index
            # print(sent[new_offset: new_offset + len(_label[-1])])
            # print(_label[-1])
            assert sent[new_offset: new_offset + len(_label[-1])] == _label[-1]

            new_labels.append((new_offset, new_offset + len(_label[-1]), _label[-1]))
        # label 被截断的情况
        elif start_index <= _label[0] < end_index < _label[1]:
            # raise RuntimeError(f'{sent}, {_label}, {question}')
            # 这里发生了截断就直接取end_index
            # new_labels.append((_label[0] - start_index, end_index, _label[-1]))
            continue
    return new_labels


if __name__ == '__main__':
    max_seq_len = 512
    t = 2
    if t == 1:
        # print(len("范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生；童年时接受良好教育后，被一位越南神父带到河内继续其学业。范廷颂于1940年在河内大修道院完成神学学业。范廷颂于1949年6月6日在河内的主教座堂晋铎；及后被派到圣女小德兰孤儿院服务。1950年代，范廷颂在河内堂区创建移民接待中心以收容到河内避战的难民。1954年，法越战争结束，越南民主共和国建都河内，当时很多天主教神职人员逃至越南的南方，但范廷颂仍然留在河内。翌年管理圣若望小修院；惟在1960年因捍卫修院的自由、自治及拒绝政府在修院设政治课的要求而被捕。1963年4月5日，教宗任命范廷颂为天主教北宁教区主教，同年8月15日就任；其牧铭为「我信天主的爱」。由于范廷颂被越南政府软禁差不多30年，因此他无法到所属堂区进行牧灵工作而专注研读等工作。范廷颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会团体等。1990年，教宗若望保禄二世在同年6月18日擢升范廷颂为天主教河内总教区宗座署理以填补该教区总主教的空缺。1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理；同年11月26日，若望保禄二世擢升范廷颂为枢机。范廷颂在1995年至2001年期间出任天主教越南主教团主席。2003年4月26日，教宗若望保禄二世任命天主教谅山教区兼天主教高平教区吴光杰主教为天主教河内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴光杰同日真除天主教河内总教区总主教职务。范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。"))
        example = {
            "paragraphs": [
                {
                    "id": "TRAIN_186",
                    "context": "范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生；童年时接受良好教育后，被一位越南神父带到河内继续其学业。范廷颂于1940年在河内大修道院完成神学学业。范廷颂于1949年6月6日在河内的主教座堂晋铎；及后被派到圣女小德兰孤儿院服务。1950年代，范廷颂在河内堂区创建移民接待中心以收容到河内避战的难民。1954年，法越战争结束，越南民主共和国建都河内，当时很多天主教神职人员逃至越南的南方，但范廷颂仍然留在河内。翌年管理圣若望小修院；惟在1960年因捍卫修院的自由、自治及拒绝政府在修院设政治课的要求而被捕。1963年4月5日，教宗任命范廷颂为天主教北宁教区主教，同年8月15日就任；其牧铭为「我信天主的爱」。由于范廷颂被越南政府软禁差不多30年，因此他无法到所属堂区进行牧灵工作而专注研读等工作。范廷颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会团体等。1990年，教宗若望保禄二世在同年6月18日擢升范廷颂为天主教河内总教区宗座署理以填补该教区总主教的空缺。1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理；同年11月26日，若望保禄二世擢升范廷颂为枢机。范廷颂在1995年至2001年期间出任天主教越南主教团主席。2003年4月26日，教宗若望保禄二世任命天主教谅山教区兼天主教高平教区吴光杰主教为天主教河内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴光杰同日真除天主教河内总教区总主教职务。范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。",
                    "qas": [
                        {
                            "question": "范廷颂是什么时候被任为主教的？",
                            "id": "TRAIN_186_QUERY_0",
                            "answers": [
                                {
                                    "text": "1963年",
                                    "answer_start": 30
                                }
                            ]
                        },
                        {
                            "question": "1990年，范廷颂担任什么职务？",
                            "id": "TRAIN_186_QUERY_1",
                            "answers": [
                                {
                                    "text": "1990年被擢升为天主教河内总教区宗座署理",
                                    "answer_start": 41
                                }
                            ]
                        },
                        {
                            "question": "范廷颂是于何时何地出生的？",
                            "id": "TRAIN_186_QUERY_2",
                            "answers": [
                                {
                                    "text": "范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生",
                                    "answer_start": 97
                                }
                            ]
                        },
                        {
                            "question": "1994年3月，范廷颂担任什么职务？",
                            "id": "TRAIN_186_QUERY_3",
                            "answers": [
                                {
                                    "text": "1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理",
                                    "answer_start": 548
                                }
                            ]
                        },
                        {
                            "question": "范廷颂是何时去世的？",
                            "id": "TRAIN_186_QUERY_4",
                            "answers": [
                                {
                                    "text": "范廷颂于2009年2月22日清晨在河内离世",
                                    "answer_start": 759
                                }
                            ]
                        }
                    ]
                }
            ],
            "id": "TRAIN_186",
            "title": "范廷颂"
        }
        create_input = []
        data = example['paragraphs']
        context = data[0]['context']
        qas = data[0]['qas']
        for q in qas:
            question = q['question']
            answers = q['answers']
            labels = []
            token_seq_len = max_seq_len - len(question)
            sentences = cut_sent_for_bert(context, token_seq_len)
            for ans in answers:
                answer = ans['text']
                start = ans['answer_start']
                end = ans['answer_start'] + len(answer)  # 这里的end是到末尾的下一位
                token_seq_len = max_seq_len - len(answer)  # 这里仅仅是根据原始文本的长度计算
                assert context[start:end] == answer
                labels.append((start, end, answer))
            # print(sentences)
            # print(labels)
            start_index = 0
            for sent in sentences:
                new_labels = refactor_labels(sent, labels, start_index)
                print(question)
                print(sent)
                print(len(sent))
                print(new_labels)
                if new_labels:
                    assert sent[new_labels[0][0]:new_labels[0][1]] == new_labels[0][-1]
                print("=" * 50)
                start_index += len(sent)
    if t == 2:
        context = "我爱，北京天安门，你知道吗"
        sentences = cut_sent_for_bert(context, 6)
        for sent in sentences:
            print(sent)
