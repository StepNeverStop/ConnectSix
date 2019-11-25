import time


def timer(func):
    def inner(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        time_cost = time.time() - start
        print(f'消耗时间: {time_cost}')
        return ret
    return inner
