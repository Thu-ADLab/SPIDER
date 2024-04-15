import time

def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def get_class_name(obj):
    return obj.__class__.__name__


