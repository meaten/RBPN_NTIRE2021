from collections import OrderedDict

def str2bool(s):
    return s.lower() in ('true', '1')

def fix_model_state_dict(state_dict):
    # remove 'module.' of dataparallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

