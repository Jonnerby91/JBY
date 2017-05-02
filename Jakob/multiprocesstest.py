import multiprocessing
from functools import partial
import numpy as np

def worker(x,y):
    """worker function"""
   # print x
    return 2*x, y**2

partial_worker=partial(worker,y=2)

#if __name__ == '__main__':
p = multiprocessing.Pool(processes=4)
x=p.map(partial_worker,range(10))
print np.array(x)[0,0]

