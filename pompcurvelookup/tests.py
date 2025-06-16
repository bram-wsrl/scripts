from pathlib import Path

import numpy as np

import interpolate as itp


test_figs = Path('test_figs')
test_figs.mkdir(exist_ok=True)


########## TEST 1 ##########
'''
Two contours with the same number of nodes
'''

contours = [
    {
        'speed': 200,
        'head': np.array([0.5, 0.4, 0.3, 0.2, 0]),
        'discharge': np.array([0, 0.2, 0.3, 0.4, 0.5])
    },
    {
        'speed': 400,
        'head': np.array([1, 0.9, 0.7, 0.5, 0]),
        'discharge': np.array([0, 0.3, 0.6, 0.8, 1])
    }]

itp.update_contours(contours)
itp.add_contours(contours, dtt=100)
itp.plot(contours, test_figs / 'test1.png')


########## TEST 2 ##########
'''
Two contours with the same number of nodes
and many interpolated contours
'''

contours = [
    {
        'speed': 200,
        'head': np.array([0.5, 0.4, 0.3, 0.2, 0]),
        'discharge': np.array([0, 0.2, 0.3, 0.4, 0.5])
    },
    {
        'speed': 400,
        'head': np.array([1, 0.9, 0.7, 0.5, 0]),
        'discharge': np.array([0, 0.3, 0.6, 0.8, 1])
    }]

itp.update_contours(contours)
itp.add_contours(contours, dtt=4)
itp.plot(contours, test_figs / 'test2.png')


########## TEST 3 ##########
'''
Two contours with more nodes in the tt max contour
'''

contours = [
    {
        'speed': 200,
        'head': np.array([0.5, 0.4, 0.3, 0.2, 0]),
        'discharge': np.array([0, 0.2, 0.3, 0.4, 0.5])
    },
    {
        'speed': 400,
        'head': np.array([1, 0.9, 0.7, 0.5, 0.3, 0.1, 0]),
        'discharge': np.array([0, 0.3, 0.6, 0.8, 0.9, 0.95, 1])
    }]

itp.update_contours(contours)
itp.add_contours(contours, dtt=100)
itp.plot(contours, test_figs / 'test3.png')


########## TEST 4 ##########
'''
Two contours with more nodes in the tt min contour
'''
contours = [
    {
        'speed': 200,
        'head': np.array([0.5, 0.45, 0.4, 0.3, 0.2, 0]),
        'discharge': np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    },
    {
        'speed': 400,
        'head': np.array([1, 0.9, 0.7, 0.5, 0]),
        'discharge': np.array([0, 0.3, 0.6, 0.8, 1])
    }]

itp.update_contours(contours)
itp.add_contours(contours, dtt=100)
itp.plot(contours, test_figs / 'test4.png')


########## TEST 5 ##########
'''
Three contours with the same number of nodes
'''
contours = [
    {
        'speed': 200,
        'head': np.array([0.5, 0.4, 0.3, 0.2, 0. ]),
        'discharge': np.array([0. , 0.2, 0.3, 0.4, 0.5])
    },
    {
        'speed': 400,
        'head': np.array([1. , 0.9, 0.7, 0.5, 0. ]),
        'discharge': np.array([0. , 0.3, 0.6, 0.8, 1. ])
    },
    {
        'speed': 300,
        'head': np.array([0.75, 0.65, 0.5 , 0.35, 0.  ]),
        'discharge': np.array([0.  , 0.25, 0.45, 0.6 , 0.75])
    }]

itp.update_contours(contours)
itp.add_contours(contours, dtt=50)

itp.plot(contours, test_figs / 'test5.png')

########## TEST 6 ##########
'''
Three contours with the same number of nodes
where the contour scaling is non-linear
'''
contours = [
    {
        'speed': 200,
        'head': np.array([0.5, 0.4, 0.3, 0.2, 0. ]),
        'discharge': np.array([0. , 0.2, 0.3, 0.4, 0.5])
    },
    {
        'speed': 400,
        'head': np.array([1. , 0.9, 0.7, 0.5, 0. ]),
        'discharge': np.array([0. , 0.3, 0.6, 0.8, 1. ])
    },
    {
        'speed': 330,
        'head': np.array([0.75, 0.65, 0.5 , 0.35, 0.  ]),
        'discharge': np.array([0.  , 0.25, 0.45, 0.6 , 0.75])
    }]

itp.update_contours(contours)
itp.add_contours(contours, dtt=50)

itp.plot(contours, test_figs / 'test6.png')


########## TEST 7 ##########
'''
Three contours with more nodes in the middle contour
'''
contours = [
    {
        'speed': 200,
        'head': np.array([0.5, 0.4, 0.3, 0.2, 0. ]),
        'discharge': np.array([0. , 0.2, 0.3, 0.4, 0.5])
    },
    {
        'speed': 400,
        'head': np.array([1. , 0.9, 0.7, 0.5, 0. ]),
        'discharge': np.array([0. , 0.3, 0.6, 0.8, 1. ])
    },
    {
        'speed': 300,
        'head': np.array([0.75, 0.65, 0.5 , 0.35, 0.2, 0.  ]),
        'discharge': np.array([0.  , 0.25, 0.45, 0.6, 0.7, 0.75])
    }]

itp.update_contours(contours)
itp.add_contours(contours, dtt=50)

itp.plot(contours, test_figs / 'test7.png')


########## TEST 8 ##########
'''
Three contours with more nodes in the middle contour
where the contour scaling is non-linear
'''
contours = [
    {
        'speed': 200,
        'head': np.array([0.5, 0.4, 0.3, 0.2, 0. ]),
        'discharge': np.array([0. , 0.2, 0.3, 0.4, 0.5])
    },
    {
        'speed': 400,
        'head': np.array([1. , 0.9, 0.7, 0.5, 0. ]),
        'discharge': np.array([0. , 0.3, 0.6, 0.8, 1. ])
    },
    {
        'speed': 270,
        'head': np.array([0.75, 0.65, 0.5 , 0.35, 0.2, 0.  ]),
        'discharge': np.array([0.  , 0.25, 0.45, 0.6, 0.7, 0.75])
    }]

itp.update_contours(contours)
itp.add_contours(contours, dtt=5)

itp.plot(contours, test_figs / 'test8.png')
