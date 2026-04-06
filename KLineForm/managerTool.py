class ManagerBoolean:
    def __init__(self, value:str,_bool: bool=False):
        self.value = value
        self._bool = _bool
    def __bool__(self):
        return self._bool
    def __str__(self):
        return self.value+':'+str(self._bool)
def manager_boolean(value:str):
    def waper(func):
        def __waper(*args, **kwargs):
            res =  func(*args, **kwargs)
            return ManagerBoolean(value, bool(res))
        return __waper
    return waper