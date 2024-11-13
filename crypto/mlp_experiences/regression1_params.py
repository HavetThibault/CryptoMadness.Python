class Regression1Params:
    INPUTS = 60
    TOTAL_INPUTS = INPUTS * 2
    AFTER = 2
    OUTPUTS = 1
    ITERATIONS = 500
    BATCH_SIZE = 64
    MIN_DELTA = 15
    LR_PATIENCE = 5
    LR_FACTOR = 0.1
    STOP_PATIENCE = 11
    H1_INIT_LR = 0.01
    H2_INIT_LR = 0.05
    REPEAT = 4
    VERSION = 1
    INTERVALS = 0
    NORMALIZED = 1

    _ROOT = 'C:/MyProgs/Python/CryptoMadness.Python/data/'
    _STR_ID = f'b{INPUTS}_a{AFTER}_i{INTERVALS}_v{VERSION}'
    SRC_DS_PATH = _ROOT + f'CV_BTC_Data_{_STR_ID}.csv'
    _STR_ID += f'_n{NORMALIZED}'
    TRAIN_PATH = _ROOT + f'train_val/btc_{_STR_ID}_train.csv'
    VAL_PATH = _ROOT + f'train_val/btc_{_STR_ID}_val.csv'
    H1_DEST_DIR = _ROOT + f'Models_{_STR_ID}_1/'
    H2_DEST_DIR = _ROOT + f'Models_{_STR_ID}_2/'
    MODEL_NAME = f'mlp_v{VERSION}'