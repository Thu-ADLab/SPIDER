

def _try_import(package_name):
    '''
    qzl: if you use this function, you will no longer receive the coding hint
    '''
    try:
        return __import__(package_name)
    except (ModuleNotFoundError, ImportError) as e:
        return _virtual_import(package_name, e)


def _try_import_from(module_name, *object_names):
    """
    qzl: if you use this function, you will no longer receive the coding hint
    Imports multiple objects from a module and returns them as a tuple.

    Args:
        module_name (str): Name of the module.
        *object_names (str): Names of the objects to import from the module.

    Returns:
        tuple: Tuple containing the imported objects.
    """
    assert len(object_names) > 0, "Must provide at least one object name. " \
                                  "If you only want to import {}, use _try_import() instead.".format(module_name)
    try:
        module = __import__(module_name, fromlist=object_names)
        imported_objects = tuple(getattr(module, obj_name) for obj_name in object_names)
    except (ModuleNotFoundError, ImportError) as e:
        imported_objects = [_virtual_import(module_name, e)] * len(object_names)

    if len(imported_objects) == 1:
        return imported_objects[0]
    else:
        return imported_objects



def _virtual_import(package_name, error_message=None):
    """
    qzl:
    If a package exsits, it will return the package directly.
    If a package does not exist, this method is used to import it temporally for no error.
    When you actually call some property or method of it, it will raise an error.
    """

    return _VirtualPackage(package_name, error_message)




class _VirtualPackage:
    def __init__(self, package_name, error_message=None):
        self._name = package_name
        if error_message is None:
            self._error_message = 'No module named %s' % self._name
        else:
            self._error_message = error_message

    def __getattr__(self, item):
        raise ModuleNotFoundError(self._error_message)

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(self._error_message)



