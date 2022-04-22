class EdgeConsts:
    WEIGHT_STR = 'weight'
    CAPACITY_STR = 'capacity'
    TTL_FLOW_STR = 'ttl_flow'
    
    MAX_WEIGHT = 50
    MIN_WEIGHT = 1


class RewardType:
    AVG = "avg"
    MEAN = "mean"
    MAX = "max"
    WEIGHTED_AVG = "wavg"


class HistoryConsts:
    BASIC_LENGTH = 10
    SOFTMIN_ALPHA = -2.0
    SOFTMAX_ALPHA = 1.0 
    EPSILON = 1.0e-3
    PERC_DEMAND = 0.999
    INFTY = 1.0e6
    ZERO = 0.0


class RLMode:
    TRAIN = "train"
    TEST = "test"
    BOTH = "both"

class SOMode:
    TRAIN = "train"
    TEST = "test"

class ActionConsts:
    ACTION_SPLITTINT_RATIOS = "splitting"
    ACTION_PATHS_SPLITING_RATIOS ="splitting_paths"
    ACTION_W_EPSILON = "w_eps"
    ACTION_W_INFTY = "w_inf" #127 -> 3.27
    ACTION_TM = "tm"

    ACTIONS_W = [ACTION_W_EPSILON, ACTION_W_INFTY]


class ActionPostProcess:
    CLIP_ZERO = "clip"
    SCALE_TO_ENV = "scale_env"
    SCALE_MIN = "scale_min"
    CEIL_TO_ENV = "ceil_env"
    

class ExtraData:
    REWARD_OVER_FUTURE = "over_future"
    REWARD_OVER_PREV = "over_prev"
    REWARD_OVER_AVG = "over_avg"
    REWARD_OVER_AVG_EXPECTED = "over_avg_expected"
    REWARD_OVER_AVG_ACTUAL = "over_avg_actual"
    REWARD_OVER_RANDOM = "over_random"
