class ObservationType:
    NO_TM = "no_tm"
    WITH_TM = "with_tm"


class SizeConsts:
    ONE_BIT = 1
    ONE_BYTE = 8 * ONE_BIT
    ONE_KB = 1024 * ONE_BYTE
    ONE_MB = 1024 * ONE_KB
    ONE_GB = 1024 * ONE_MB
    
    ONE_Kb = 1000 * ONE_BIT
    ONE_Mb = 1000 * ONE_Kb
    ONE_Gb = 1000 * ONE_Mb
    
    GB_TO_MB_SCALE = ONE_Gb / ONE_Mb
    
    ONE_NS = 1
    ONE_US = 1000 * ONE_NS
    ONE_MS = 1000 * ONE_US
    ONE_S = 1000 * ONE_US

    ONE = 1
    ONE_TEN = 10 * ONE
    ONE_HUNDRED = 10 * ONE_TEN
    ONE_THOUSAND = 10 * ONE_HUNDRED
    ONE_MILLION = ONE_THOUSAND * ONE_THOUSAND

    MSS_SIZE = 1460*ONE_BYTE
    RTT = 100
    
    BPS_TO_GBPS = lambda x: x / SizeConsts.ONE_Gb
    GBPS_TO_BPS = lambda x: x * SizeConsts.ONE_Gb
    
    GBPS_TO_MBPS = lambda x: x * SizeConsts.GB_TO_MB_SCALE
    
    BPS_TO_MBPS = lambda x: x / SizeConsts.ONE_Mb
    MBPS_TO_BPS = lambda x: x * SizeConsts.ONE_Mb


class Environments:
    FLOWS = "flows"
    ECMP = "ecmp"
    ECMP_HISTORY = "ecmp_history"
    TEST_LOAD = "load"
    TEST_UNLOAD = "unload"
    TEST_POINT = "point"


class Policies:
    RANDOM = "random"
    STATIC = "static"
    ROUND_ROBIN = "round_robin"
    HEDERA_FIRST_FIT = "hedera_ff"
    HEDERA_MIN_FIRST_FIT = "hedera_min_ff"
    HEDERA_SIMULATED_ANNEALING = "hedera_sa"
    EQUILIBRIUM = "equilibrium"
    POLICY_GRADIENT_MLP = "pg_mlp"
    POLICY_GRADIENT_CONTINIOUS_MLP = "pg_cont_mlp"
    POLICY_GRADIENT_GRU = "pg_gru"
    Q_LEARNING = "q_learning"
    POLICY_GRADIENT_GRAPH_EMBEDDING = "pg_embedding"
    PG_TYPES = [POLICY_GRADIENT_GRU, POLICY_GRADIENT_MLP, POLICY_GRADIENT_GRAPH_EMBEDDING, POLICY_GRADIENT_CONTINIOUS_MLP]
    GRAPH_EMBEDDINGS = ["pg_embedding"]


class GraphConsts:
    SRC_META_POS = 0
    DST_META_POS = 1
    EDGE_META_POS = 2
    WEIGHT_STR = 'weight'
    CAPACITY_STR = 'capacity'
    TTL_FLOW_STR = 'ttl_flow'


class TrainingConsts:
    ITERATION_PER_TEST = "iter_per_test"
    TRAIN_BATCH_SIZE = "train_batch_size"
    TEST_BATCH_SIZE = "test_batch"


class FolderPathCosts:
    BASE_PATH_LOCAL = "LOCAL/networking_envs"
    BASE_PATH_REMOTE = "REMOTE/networking_envs"