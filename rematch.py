# coding=utf-8
"""
该文件用于对token化后标签和文本对齐，
来源bert4keras
"""
import unicodedata


def stem(token):
    """获取token的“词干”（如果是##开头，则自动去掉##）
    """
    if token[:2] == '##':
        return token[2:]
    else:
        return token


def is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')


def lowercase_and_normalize(text):
    """转小写，并进行简单的标准化
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text

def is_special(ch):
    """判断是不是有特殊含义的符号
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')


def rematch(text, tokens, do_lower_case):
    """给出原始的text和tokenize后的tokens的映射关系
    """

    if do_lower_case:
        text = text.lower()

    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
        if do_lower_case:
            ch = lowercase_and_normalize(ch)
        ch = ''.join([
            c for c in ch
            if not (ord(c) == 0 or ord(c) == 0xfffd or is_control(c))
        ])
        normalized_text += ch
        char_mapping.extend([i] * len(ch))

    # print(normalized_text)
    text, token_mapping, offset = normalized_text, [], 0

    for token in tokens:
        if is_special(token):
            token_mapping.append([])
        else:
            token = stem(token)
            start = text[offset:].index(token) + offset
            end = start + len(token)
            token_mapping.append(char_mapping[start:end])
            offset = end

    return token_mapping


if __name__ == '__main__':
    from tokenization import BasicTokenizer
    from transformers import BertTokenizer

    max_seq_length = 512
    basicTokenizer = BasicTokenizer(do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained('./model_hub/voidful-albert-chinese-tiny/')
    question_tokens = tokenizer.tokenize("1994年3月，范廷颂担任什么职务？")
    context = "范廷颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会团体等。1990年，教宗若望保禄二世在同年6月18日擢升范廷颂为天主教河内总教区宗座署理以填补该教区总主教的空缺。1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理；同年11月26日，若望保禄二世擢升范廷颂为枢机。范廷颂在1995年至2001年期间出任天主教越南主教团主席。2003年4月26日，教宗若望保禄二世任命天主教谅山教区兼天主教高平教区吴光杰主教为天主教河内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴光杰同日真除天主教河内总教区总主教职务。范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。"
    start_position = [97]
    end_position = [147]
    print("context长度：", len(context))
    print(context[start_position[0]:end_position[0]])

    whitespace_doc = tokenizer.tokenize(context)  # 先初步对context进行token化
    max_tokens_for_doc = max_seq_length - len(question_tokens) - 3
    mapping = rematch(context, whitespace_doc, do_lower_case=True)
    start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
    end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}

    start_res = []
    end_res = []

    for start in start_position:
        if start in start_mapping:
            start_res.append(start_mapping[start])
    for end in end_position:
        if end in end_mapping:
            end_res.append(end_mapping[end])
    print(start_res)
    print(end_res)
    print("".join(whitespace_doc[start_res[0]:end_res[0]]))