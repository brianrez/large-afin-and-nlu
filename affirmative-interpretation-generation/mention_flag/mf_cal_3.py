import torch
# implements three value mention flags definition. Needs more testing, but some debugging has been already done.

def mention_flag(
    input_ids: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    original_cue: list,
):
    """
    @param input_ids: [batch_size, input_ids_len]
    @param decoder_input_ids: [batch_size, decoder_ids_len]

    @return mention_flag_matrix: [batch_size, decoerer_ids_len, input_ids_len]
    """

    batch_size, input_ids_len = input_ids.shape
    _, decoder_ids_len = decoder_input_ids.shape

    mention_flag_matrix = torch.zeros((batch_size, decoder_ids_len, input_ids_len))

    for i in range(batch_size):
        input_ids_ = input_ids[i].clone().tolist()
        decoder_ids_ = decoder_input_ids[i].clone().tolist()

        # split  input_ids by [14261, 123, 15, 7, 10] to [before] and [after]
        input_sent_ = split_list_by_list(input_ids_, [14261, 123, 15, 7, 10])
        if input_sent_ is None:
            continue
        list_neg = input_sent_[1].copy()
        list_neg = split_list_by_value(list_neg, 6)
        input_sent_ = input_sent_[0]

        # i: batch index
        # j: decoder index
        # k: input index

        for j in range(decoder_ids_len):
            for k in range(input_ids_len):
                decoder_token = decoder_ids_[j]
                input_token = input_ids_[k]

                if input_token == 0 or input_token == 1 or input_token == 6:
                    mention_flag_matrix[i, j, k] = 0
                    continue

                if j > 1:
                    if decoder_token != input_token:
                        mention_flag_matrix[i, j, k] = mention_flag_matrix[i, j - 1, k]
                        continue
                    else:
                        neg_ = find_neg_lists(
                            input_ids_[: k + 1], decoder_ids_[: j + 1], list_neg + [original_cue]
                        )
                        if neg_ is None:
                            mention_flag_matrix[i, j, k] = mention_flag_matrix[
                                i, j - 1, k
                            ]
                            continue
                        else:
                            l = len(neg_)
                            for n in range(l):
                                for m in range(l):
                                    mention_flag_matrix[
                                        i, j - m, k - n
                                    ] = mention_flag_matrix[i, 0, k - n]

                elif j == 1:
                    if k > len(input_sent_) - 1 + 5:
                        if (
                            input_token != 6
                            and input_token == decoder_token
                            and input_ids_[k - 1] == 6
                            and input_ids_[k + 1] == 6
                        ):
                            mention_flag_matrix[i, j, k] = 2

                    else:
                        if len(original_cue) == 1:
                            if (
                                input_token == decoder_token
                                and input_token == original_cue[0]
                            ):
                                mention_flag_matrix[i, j, k] = 1

                elif j == 0:
                    if k > len(input_sent_) - 1 + 5:
                        if input_token != 6:
                            mention_flag_matrix[i, j, k] = 2

                    else:
                        ind = find_original(input_ids_, original_cue)
                        if ind is not None:
                            if k >= ind and k < ind + len(original_cue):
                                mention_flag_matrix[i, j, k] = 1
                        else:
                            raise Exception("original cue should be in the input_ids.")
                else:
                    raise Exception("j should be non-negative.")

    return mention_flag_matrix


def find_original(
    input_ids: list,
    neg: list,  # 1d
):
    # returns the beginning index of sublist: neg
    input_ids_ = input_ids.copy()
    neg_ = neg.copy()
    index = 0
    while len(input_ids_) >= len(neg_):
        if input_ids_[: len(neg_)] == neg_:
            return index
        else:
            input_ids_.pop(0)
            index += 1
    return None


print(find_original([1, 2, 3, 4, 5], [3, 4, 5]))


def find_neg(list_, negations):
    results = []
    for neg_list in negations:
        pass


def find_neg_lists(first_, second_, negations):
    first_rev = first_.copy()[::-1]
    second_rev = second_.copy()[::-1]
    list_match = []
    for i in range(len(first_rev)):
        if first_rev[i] == second_rev[i]:
            list_match.append(first_rev[i])
        else:
            break
    # list_match = list_match[::-1]

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
    init_list_ = init_list.copy()
    sub_list_ = sub_list.copy()
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

    mention_flag_matrix = mention_flag_matrix.clone().tolist()

    dict_ = {}
    dict_["input_ids"] = input_ids_
    for i in range(len(decoder_id_)):
        dict_[decoder_id_[i]] = mention_flag_matrix[i]

    from pandas import DataFrame as df
    data = df(dict_)
    print(data)

    return None


from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-large")

input_sentence = "Hello I am disagreeable here with this thing. neg cues: unwidely, nothing, undamaged,"

output_sentence = "<pad> Hi I am unwidely here, talking to you, and disagreeable here"

input_id = tokenizer(input_sentence, return_tensors="pt")['input_ids']
print(type(input_id))
decoder_id = tokenizer(output_sentence, return_tensors="pt")['input_ids']
orig_cue = tokenizer("disagreeable")['input_ids'][:-1]




mentionflag = mention_flag(input_id, decoder_id, orig_cue)

input_id = [f"{id} ({tokenizer.decode([id])})" for id in input_id[0]]
decoder_id = [f"{id} ({tokenizer.decode([id])})" for id in decoder_id[0]]
pretty_mf_printer(input_id, decoder_id, mentionflag[0])

    

