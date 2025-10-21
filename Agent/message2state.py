import numpy as np

CARD_ORDER = ['H2', 'C2', 'S2', 'D2', 'H3', 'C3', 'S3', 'D3', 'H4', 'C4', 'S4', 'D4',
              'H5', 'C5', 'S5', 'D5', 'H6', 'C6', 'S6', 'D6', 'H7', 'C7', 'S7', 'D7',
              'H8', 'C8', 'S8', 'D8', 'H9', 'C9', 'S9', 'D9', 'HT', 'CT', 'ST', 'DT',
              'HJ', 'CJ', 'SJ', 'DJ', 'HQ', 'CQ', 'SQ', 'DQ', 'HK', 'CK', 'SK', 'DK',
              'HA', 'CA', 'SA', 'DA', 'HB', 'SR']

# def convert_message_to_state(actions, origin_cards, played_cards, up_player_played, teammate_played, others_played1
#                              , others_played2, others_played3, remaining_counts_others, wild_cards, round_num=0, max_cards=54, max_actions=50):
#     """
#     转换消息为固定维度的状态向量
#     :param cards: 自己手牌 list[int] or list[str]
#     :param actions: 当前可选动作列表，每个动作是 dict，至少包含 "index"
#     :param round_num: 当前轮次
#     :param max_cards: 最大牌数（掼蛋54张）
#     :param max_actions: 最大候选动作数
#     :return: state向量 (numpy array), action_mask
#     """
#
#     # --- 动作 mask ---
#     action_mask = np.zeros(max_actions, dtype=np.float32)
#     for a in actions:
#         try:
#             idx = int(a["index"])  # 这里强制转 int
#         except Exception:
#             continue
#         if 0 <= idx < max_actions:
#             action_mask[idx] = 1
#
#     # --- 轮次简单归一化 ---
#     round_feature = np.array([float(round_num) / 100.0], dtype=np.float32)
#
#     # --- 拼接状态 ---
#     state = np.concatenate([card_vec, round_feature])
#
#     return state, action_mask


def cards_to_vector(cards, card_order):
    """
    把手牌列表映射为固定顺序的计数向量
    :param cards: list[str]  输入的牌，比如 ['HA','DA','DA']
    :param card_order: list[str]  固定顺序，比如 ['H2','C2',...,'HA','DA']
    :return: numpy.ndarray, shape=(len(card_order),), dtype=int
    """
    vec = np.zeros(len(card_order), dtype=int)
    index_map = {c: i for i, c in enumerate(card_order)}  # 建立映射
    for card in cards:
        if card in index_map:
            vec[index_map[card]] += 1
    return vec

def proc_universal(handCards, cur_rank):
    res = np.zeros(12, dtype=np.int8)

    if handCards[cur_rank * 4] == 0:
        return res

    res[0] = 1
    rock_flag = 0
    for i in range(4):
        left, right = 0, 5
        temp = [handCards[i + j * 4] if i + j * 4 != cur_rank * 4 else 0 for j in range(5)]
        while right <= 12:
            zero_num = temp.count(0)
            if zero_num <= 1:
                rock_flag = 1
                break
            else:
                temp.append(handCards[i + right * 4] if i + right * 4 != cur_rank * 4 else 0)
                temp.pop(0)
                left += 1
                right += 1
        if rock_flag == 1:
            break
    res[1] = rock_flag

    num_count = [0] * 13
    for i in range(4):
        for j in range(13):
            if handCards[i + j * 4] != 0 and i + j * 4 != cur_rank * 4:
                num_count[j] += 1
    num_max = max(num_count)
    if num_max >= 6:
        res[2:8] = 1
    elif num_max == 5:
        res[3:8] = 1
    elif num_max == 4:
        res[4:8] = 1
    elif num_max == 3:
        res[5:8] = 1
    elif num_max == 2:
        res[6:8] = 1
    else:
        res[7] = 1
    temp = 0
    for i in range(13):
        if num_count[i] != 0:
            temp += 1
            if i >= 1:
                if num_count[i] == 2 and num_count[i - 1] >= 3 or num_count[i] >= 3 and num_count[i - 1] == 2:
                    res[9] = 1
                elif num_count[i] == 2 and num_count[i - 1] == 2:
                    res[11] = 1
            if i >= 2:
                if num_count[i - 2] == 1 and num_count[i - 1] >= 2 and num_count[i] >= 2 or \
                        num_count[i - 2] >= 2 and num_count[i - 1] == 1 and num_count[i] >= 2 or \
                        num_count[i - 2] >= 2 and num_count[i - 1] >= 2 and num_count[i] == 1:
                    res[10] = 1
        else:
            temp = 0
    if temp >= 4:
        res[8] = 1
    return res

def convert_message_to_state(origin_cards, played_cards, teammate_played, others_played1
                             , others_played2, others_played3, remaining_counts_others, rank ,last_pos):
    origin_cards_vec = cards_to_vector(origin_cards, CARD_ORDER)
    wild_cards = proc_universal(origin_cards_vec, rank)
    if last_pos == -1:
        played_cards_vec = cards_to_vector(played_cards, CARD_ORDER)
    else:
        if not played_cards:
            played_cards_vec = [-1] * 54
        else:
            played_cards_vec = cards_to_vector(played_cards, CARD_ORDER)
    # up_player_played_vec = cards_to_vector(up_player_played, CARD_ORDER)
    if remaining_counts_others[1] == 0:
        teammate_played_vec = [-1] * 54
    else:
        teammate_played_vec = cards_to_vector(teammate_played, CARD_ORDER)
    others_played1_vec = cards_to_vector(others_played1, CARD_ORDER)
    others_played2_vec = cards_to_vector(others_played2, CARD_ORDER)
    others_played3_vec = cards_to_vector(others_played3, CARD_ORDER)

    # 生成初始牌池向量：前 (N-2) 位为4，最后 2 位为2
    N = len(CARD_ORDER)
    init_vec = np.full(N, 4.0, dtype=np.float32)
    init_vec[-2:] = 2.0

    # 一行减去已知的牌（origin + played + others1 + others2 + others3）
    unknown_cards_vec = init_vec - (
        origin_cards_vec + played_cards_vec +
        others_played1_vec + others_played2_vec + others_played3_vec
    )

    # 防止负值（如果某个计数被多次减到负数，裁剪到0）
    unknown_cards_vec = np.clip(unknown_cards_vec, 0.0, None)
    #剩余牌
    others_played1_left_vec = np.zeros(28,int)
    others_played1_left_vec[remaining_counts_others[0]] = 1
    others_played2_left_vec = np.zeros(28,int)
    others_played2_left_vec[remaining_counts_others[1]] = 1
    others_played3_left_vec = np.zeros(28,int)
    others_played3_left_vec[remaining_counts_others[2]] = 1
    remaining_counts_others = np.atleast_1d(np.array(remaining_counts_others, dtype=np.float32))
    #级牌
    rank_vec = np.zeros(13, int)
    rank_vec[rank] = 1
    # 把所有向量合并
    state = np.concatenate([
        origin_cards_vec,       #自己当前手牌
        wild_cards,             #万能牌
        unknown_cards_vec,      #剩下的牌
        played_cards_vec,       #最近一次出的牌
        # up_player_played_vec,
        teammate_played_vec,    #队友出的牌，队友结束全为-1
        others_played1_vec,     #下家出过的所有牌
        others_played2_vec,     #对家出过的所有牌
        others_played3_vec,     #上家出过的所有牌
        others_played1_left_vec,    #三位剩余牌的数目
        others_played2_left_vec,
        others_played3_left_vec,
        rank_vec,                    #自家级数，对手级数，当前级数（根据规则不升级，都是一样的）
        rank_vec,
        rank_vec
    ])
    return state


def action_to_state(actions):
    new_actions = []
    for a in actions:
        if a['action'][2] == 'PASS':
            action_vec = np.zeros(54,int)
            new_actions.append(action_vec)
        else:
            action_vec = cards_to_vector(a['action'][2], CARD_ORDER)
            new_actions.append(action_vec)
    return new_actions
