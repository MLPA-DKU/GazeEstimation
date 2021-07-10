import os.path
import yaml

from yacs.config import CfgNode as CN


_C = CN()

_C.BASE = ['']
_C.NAME = ''
_C.VERBOSE = True

_C.MODEL = CN()
_C.MODEL.NAME = 'cls_cvt'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.PRETRAINED_LAYERS = ['*']
_C.MODEL.NUM_CLASSES = 2

_C.MODEL.SPEC = CN(new_allowed=True)
_C.MODEL.SPEC.INIT = 'trunc_norm'
_C.MODEL.SPEC.NUM_STAGES = 3
_C.MODEL.SPEC.PATCH_SIZE = [7, 3, 3]
_C.MODEL.SPEC.PATCH_STRIDE = [4, 2, 2]
_C.MODEL.SPEC.PATCH_PADDING = [2, 1, 1]
_C.MODEL.SPEC.DIM_EMBED = [64, 192, 384]
_C.MODEL.SPEC.NUM_HEADS = [1, 3, 6]
_C.MODEL.SPEC.DEPTH = [1, 2, 10]
_C.MODEL.SPEC.MLP_RATIO = [4.0, 4.0, 4.0]
_C.MODEL.SPEC.ATTN_DROP_RATE = [0.0, 0.0, 0.0]
_C.MODEL.SPEC.DROP_RATE = [0.0, 0.0, 0.0]
_C.MODEL.SPEC.DROP_PATH_RATE = [0.0, 0.0, 0.1]
_C.MODEL.SPEC.QKV_BIAS = [True, True, True]
_C.MODEL.SPEC.CLS_TOKEN = [False, False, True]
_C.MODEL.SPEC.POS_EMBED = [False, False, False]
_C.MODEL.SPEC.QKV_PROJ_METHOD = ['dw_bn', 'dw_bn', 'dw_bn']
_C.MODEL.SPEC.KERNEL_QKV = [3, 3, 3]
_C.MODEL.SPEC.PADDING_KV = [1, 1, 1]
_C.MODEL.SPEC.STRIDE_KV = [2, 2, 2]
_C.MODEL.SPEC.PADDING_Q = [1, 1, 1]
_C.MODEL.SPEC.STRIDE_Q = [1, 1, 1]


def _update_config_from_file(config, file):
    config.defrost()
    with open(file, 'r') as f:
        file_config = yaml.load(f, Loader=yaml.FullLoader)
    for conf in file_config.setdefault('BASE', ['']):
        if conf:
            _update_config_from_file(config, os.path.join(os.path.dirname(file), conf))
    config.merge_from_file(file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args)
    config.defrost()
    config.merge_from_list(args)
    file_name, _ = os.path.splitext(os.path.basename(args))
    config.NAME = file_name + config.NAME
    config.freeze()
