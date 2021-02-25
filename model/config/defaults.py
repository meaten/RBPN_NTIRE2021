from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
# _C.MODEL.SCALE_FACTOR = 4
_C.MODEL.BASE_FILTER = 64
_C.MODEL.FEAT = 64
_C.MODEL.BURST_SIZE = 8
_C.MODEL.NUM_RESBLOCK = 5
_C.MODEL.USE_FLOW = False
# _C.MODEL.INPUT_CHANNEL = 3
_C.MODEL.PREPROCESS = "Nearest"  # 'Nearest' or 'RGGB2channel'
_C.MODEL.OUTPUT_CHANNEL = 3
_C.MODEL.EXTRACTOR_TYPE = 'normal'  # 'original', 'deep', 'deform', 'deepdeform', 'pcd_align'
_C.MODEL.LOSS = 'l1'  # 'l1', 'pit' or 'alignedl2'

_C.SOLVER = CN()
_C.SOLVER.LR = 1e-4
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.MAX_ITER = 1000000
_C.SOLVER.SYNC_BATCHNORM = True
_C.SOLVER.PATCH_SIZE = 128
_C.SOLVER.LR_STEP = [600000, 900000]
_C.SOLVER.PRETRAIN_ITER = 50000  # only for real track
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITER = 5000

_C.SOLVER.PITLOSS = CN()
_C.SOLVER.PITLOSS.MSE_WEIGHT = 1.
_C.SOLVER.PITLOSS.VGG_WEGIHT = 6e-3
_C.SOLVER.PITLOSS.TV_WEIGHT = 2e-8

_C.DATASET = CN()
_C.DATASET.TRACK = 'synthetic'
_C.DATASET.TRAIN_SYNTHETIC = 'dataset/Zurich'
_C.DATASET.VAL_SYNTHETIC = 'dataset/syn_burst_val'
_C.DATASET.REAL = 'dataset/burstsr_dataset'

_C.PWCNET_WEIGHTS = 'weights/pwcnet-network-default.pth'

_C.OUTPUT_DIR = ''
_C.SEED = 123
