from json import decoder
import torch

from copy import deepcopy
def mention_flag(
    input_ids: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    original_cue_: list,
    list_cues: list,
):
    """
    @param input_ids: [batch_size, input_ids_len]
    @param decoder_input_ids: [batch_size, decoder_ids_len]

    @return mention_flag_matrix: [batch_size, decoerer_ids_len, input_ids_len]
    """
    decoder_ids_len_ = decoder_input_ids.shape[1]
    # print("entered mention_flag")
    # print("decoder_ids_len_:", decoder_ids_len_)
    # print(input_ids.shape)
    # print(original_cue_)
    batch_size, input_ids_len = input_ids.shape
    _, decoder_ids_len = decoder_input_ids.shape

    mention_flag_matrix = torch.zeros((batch_size, decoder_ids_len, input_ids_len), dtype=torch.long)

    for i in range(batch_size):
        original_cue = original_cue_[i]
        input_ids_ = input_ids[i].clone().tolist()
        decoder_ids_ = decoder_input_ids[i].clone().tolist()

        # i: batch index
        # j: decoder index
        # k: input index
        index_cue = find_index_sublist(input_ids_, original_cue)
        if index_cue is None:
            # print(original_cue)
            # print(input_ids_)
            raise Exception("original cue should be in the input_ids.")

        for k in range(input_ids_len):
            if k != index_cue + len(original_cue) - 1:
                continue
            
            for j in range(decoder_ids_len):
                if j == 0:
                    for m in range(len(original_cue)):
                        mention_flag_matrix[i, j, k-m] = 1
                    continue

                if j == 1:
                    token = decoder_ids_[j]
                    for list_neg in list_cues:
                        if len(list_neg) == 1:
                            if token == list_neg[0]:
                                for m in range(len(original_cue)):
                                    mention_flag_matrix[i, j, k-m] = 1
                    continue

                negation_ = find_neg(decoder_ids_[:j+1], list_cues)

                if negation_ is None:
                    for m in range(len(original_cue)):
                        mention_flag_matrix[i, j, k-m] = mention_flag_matrix[i, j-1, k-m]
                    continue
                len_negation = len(negation_)
                # print("len_negation:", len_negation)
                # print("len(original_cue):", len(original_cue))
                # print("j:", j)
                # print("k:", k)
                for l in range(len_negation):
                    for m in range(len(original_cue)):
                        mention_flag_matrix[i, j-l, k-m] = 1
                        # print(f"mention_flag_matrix[{i}, {j-l}, {k-m}  ]:", mention_flag_matrix[i, j-l, k-m])
    # print("decoder_ids_len:", decoder_ids_len)
    # print("finished mention_flag_matrix")
    return mention_flag_matrix


def find_original(
    input_ids: list,
    neg: list,  # 1d
):
    # returns the beginning index of sublist: neg
    input_ids_ = deepcopy(input_ids)
    neg_ = neg.copy()
    index = 0
    while len(input_ids_) >= len(neg_):
        if input_ids_[: len(neg_)] == neg_:
            return index
        else:
            input_ids_.pop(0)
            index += 1
    return None

def find_neg(first_, negations):
    first_rev = deepcopy(first_)[::-1]
    # first_rev = first_.copy()[::-1]
    for neg_list in negations:
        neg_list_rev = neg_list.copy()[::-1]
        if len(neg_list_rev) > len(first_rev):
            continue
        else:
            if neg_list_rev == first_rev[: len(neg_list_rev)]:
                return neg_list
    return None

def find_neg_lists(first_, second_, negations):
    first_rev = deepcopy(first_)[::-1]
    second_rev = deepcopy(second_)[::-1]

    # first_rev = first_.copy()[::-1]
    # second_rev = second_.copy()[::-1]
    list_match = []
    for i in range(len(first_rev)):
        if first_rev[i] == second_rev[i]:
            list_match.append(first_rev[i])
        else:
            break

    for neg_list in negations:
        neg_list_rev = neg_list.copy()[::-1]
        if len(neg_list_rev) > len(list_match):
            continue
        else:
            if neg_list_rev == list_match[: len(neg_list_rev)]:
                return neg_list
    return None


def split_list_by_list(init_list, list_split):
    """
    This function is used to split a list by another list.
    For example,
    init_list = [1, 2, 3, 4, 5]
    list_split = [3, 4]
    Then, the result is [[1, 2], [5]]
    If list_split is not a subset of init_list with the same order, then return None.
    """

    result = []

    index = find_index_sublist(init_list, list_split)
    if index is None:
        return None

    result.append(init_list[:index])
    result.append(init_list[index + len(list_split) :])

    return result


def find_index_sublist(init_list, sub_list):
    """
    This function is used to find the index of sub_list in init_list.
    """
    init_list_ = deepcopy(init_list)
    sub_list_ = deepcopy(sub_list)
    # init_list_ = init_list.copy()
    # sub_list_ = sub_list.copy()
    index = 0
    while len(init_list_) >= len(sub_list_):
        if init_list_[: len(sub_list_)] == sub_list_:
            return index
        else:
            init_list_.pop(0)
            index += 1
    return None


def split_list_by_value(list_, value):
    """
    This function is used to split a list by a value.
    For example,
    list_ = [1, 2, 3, 4, 3, 5]
    value = 3
    Then, the result is [[1, 2], [4], [5]]
    """
    result = []
    temp = []
    for i in list_:
        if i == value:
            result.append(temp)
            temp = []
        else:
            temp.append(i)
    result.append(temp)
    return result


def pretty_mf_printer(input_ids, decoder_id, mention_flag_matrix):
    # print input_ids as the first column
    # print decoder_id as the first row
    # print mention_flag_matrix [decoder_len, input_len] in the middle

    input_ids_ = input_ids
    decoder_id_ = decoder_id
    print("decoder_id_:", decoder_id_)
    print(len(decoder_id_))
    mention_flag_matrix = mention_flag_matrix.clone().tolist()


    dict_ = {}
    dict_["input_ids"] = input_ids_
    for i in range(len(decoder_id_)):
        dict_[decoder_id_[i] + str(i)] = mention_flag_matrix[i]

    from pandas import DataFrame as df
    data = df(dict_)
    print(data.shape)
    print(data)

    return None

'''
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-large")



input_sentence = "Hello this is undamaged"
output_sentence = "<pad> I un am widely here and unharmed here do not like"
input_id = tokenizer(input_sentence, return_tensors="pt")['input_ids']
decoder_id = tokenizer(output_sentence, return_tensors="pt")['input_ids']
print(decoder_id)
orig_cue = tokenizer("undamaged")['input_ids'][:-1]
orig_cue = [orig_cue]
list_cues = tokenizer(["not"])['input_ids']
list_cues = [l[:-1] for l in list_cues]

mentionflag = mention_flag(input_id, decoder_id, orig_cue, list_cues)
print(mentionflag.shape)

input_id = [f"{id} ({tokenizer.decode([id])})" for id in input_id[0]]
decoder_id = [f"{id} ({tokenizer.decode([id])})" for id in decoder_id[0]]
pretty_mf_printer(input_id, decoder_id, mentionflag[0])
'''