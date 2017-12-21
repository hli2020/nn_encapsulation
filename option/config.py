v2 = {
    'feature_maps':     {
        '300':      [38, 19, 10, 5, 3, 1],
        '512':      [64, 32, 16, 8, 6, 4],
    },
    'aspect_ratios':    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'min_scale':        0.1,
    'max_scale':        0.9,
    'beyond_max':       1.0,
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v2',
}

v3 = {
    'feature_maps':     {
        '300':      [38, 19, 10, 5, 3, 1],
        '512':      [64, 32, 16, 8, 6, 4],
    },
    'aspect_ratios':    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'min_scale':        0.1,
    'max_scale':        0.8,
    'beyond_max':       1.05,
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v3',
}

# damn it
v2_512 = {
    'image_size':       512,
    'steps':            [8, 16, 32, 64, 86, 128],
    'feature_maps':     [64, 32, 16, 8, 6, 4],
    'min_sizes':        [45, 100, 150, 256, 360, 460],
    'max_sizes':        [100, 150, 256, 360, 460, 530],
    'aspect_ratios':    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v2_512',
}

# min_scale, 0.1, max 0.9, linearly spaced
v2_512_standard = {
    'image_size':       512,
    'steps':            [8, 16, 32, 64, 86, 128],
    'feature_maps':     [64, 32, 16, 8, 6, 4],
    'min_sizes':        [51, 134, 215, 296, 378, 460],
    'max_sizes':        [134, 215, 296, 378, 460, 530],
    'aspect_ratios':    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v2_512_standard',
}

v2_512_stan_more_ar = {
    'image_size':       512,
    'steps':            [8, 16, 32, 64, 86, 128],
    'feature_maps':     [64, 32, 16, 8, 6, 4],
    'min_sizes':        [51, 134, 215, 296, 378, 460],
    'max_sizes':        [134, 215, 296, 378, 460, 530],
    'aspect_ratios':    [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v2_512_stan_more_ar',
}

v2_634 = {
    'image_size':       634,
    # step = image_size / feature_map
    'steps':            [8, 16, 32, 64, 80, 106],
    'feature_maps':     [79, 39, 20, 10, 8, 6],
    'min_sizes':        [63, 165, 266, 367, 469, 570],
    'max_sizes':        [165, 266, 367, 469, 570, 650],
    'aspect_ratios':    [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v2_634',
}

# min_scale, 0.1, max 0.9, linearly spaced
v2_634_standard = {
    'image_size':       634,
    'steps':            [8, 16, 32, 64, 80, 106],
    'feature_maps':     [79, 39, 20, 10, 8, 6],
    'min_sizes':        [63, 165, 266, 367, 469, 570],
    'max_sizes':        [165, 266, 367, 469, 570, 650],
    'aspect_ratios':    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v2_634_standard',
}

# the original config:
# 30.0
# 78.0
# 126.00000000000001
# 174.0
# 222.0
# 270.0
# v2 = {
#     'feature_maps': [38, 19, 10, 5, 3, 1],
#     'min_dim': 300,
#     'steps': [8, 16, 32, 64, 100, 300],
#     'min_sizes': [30, 60, 111, 162, 213, 264],
#     'max_sizes': [60, 111, 162, 213, 264, 315],
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'v2',
# }

# in the 'ssd.py' file:
# mbox = {
#     '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#     '512': [4, 6, 6, 6, 4, 4],
# }
