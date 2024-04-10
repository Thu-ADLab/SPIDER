import copy
from types import FunctionType


def closure(function, *args):
    '''
    创建闭包函数，避免参数被修改。
    '''
    new_function = lambda: function(*args)
    return new_function


class LazyList(list):
    '''
    惰性计算的list。
    self 本身存放的是函数，调用 self 时，执行 self 所包含的函数，并将结果缓存下来。
    在后续调用时，直接返回缓存的结果。
    call by need
    目前仅检查了索引、切片操作，能完成惰性计算。
    '''
    def __init__(self, iterable_of_callable=None):
        if iterable_of_callable is None:
            super().__init__()
        else:
            super().__init__(iterable_of_callable)

        self._cache = {}

    def __iter__(self): # 返回的也是一个生成器
        return (self[i] for i in range(len(self)))

    def __getitem__(self, index):
        if isinstance(index, slice): # 切片。LazyList切片操作并不高效。
            return [self.__getitem__(i) for i in range(index.start, index.stop, index.step)]

        else: # 索引
            if index in self._cache:
                return self._cache[index]
            else:
                generate_func = super().__getitem__(index)
                assert callable(generate_func), \
                    'LazyList is used to store callable. Please disable calc_by_need mode if you want to store the result.'
                self._cache[index] = result = generate_func()
                return result


    @staticmethod
    def wrap_generator(function, *args):
        return closure(function, *args)

    def to_instance_list(self):
        return list(self)


if __name__ == '__main__':
    samples = LazyList()
    for x in [1, 2, 3]:
        for y in [4,5,6]:
            samples.append(  # append一个callable函数
                closure(lambda x, y: x + y, x, y)
            )

    for val in samples:
        print(val)
        print(samples._cache)

    print("--------------------------------")
    samples = LazyList()
    for x in [1, 2]:
        for y in [4, 5]:
            samples.append(  # append一个callable函数
                closure(lambda x, y: x + y, x, y)
            )
    a,b,c,d = samples
    print(a,b,c,d)
    print(samples._cache)




