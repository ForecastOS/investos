import copy

def deep_dict_merge(default_d, update_d):
    "Deep copies update_d onto default_d recursively"
    
    default_d = copy.deepcopy(default_d)
    update_d = copy.deepcopy(update_d)

    def deep_dict_merge_inner(default_d, update_d):
        for k, v in update_d.items():
            if (k in default_d and isinstance(default_d[k], dict) and isinstance(update_d[k], dict)):
                deep_dict_merge_inner(default_d[k], update_d[k])
            else:
                default_d[k] = update_d[k]

    deep_dict_merge_inner(default_d, update_d)
    return default_d # With update_d values copied onto it