import asyncio

class ObservableSet:
    def __init__(self, val = None):
        self.__value = set() if val == None else val
        self.__futures = {
            "size_change": []
        }

    def on_size_change(self):
        future = asyncio.get_running_loop().create_future()
        self.__futures["size_change"].append(future)
        return future
    
    def _resolve_size_change_futures(self):
        for future in self.__futures["size_change"]:
            future.set_result(len(self.__value))
        self.__futures["size_change"] = []
    
    def __getattr__(self, attr):
        # If the attribute exists in the internal set, forward the call to it
        if hasattr(self.__value, attr):
            method = getattr(self.__value, attr)

            # Intercept size-modifying methods to resume all futures pending
            if attr in ['add', 'remove', 'discard', 'clear']:
                def wrapped_method(*args, **kwargs):
                    try:
                        result = method(*args, **kwargs)
                    except Exception as e:
                        raise Exception(f"Error in {attr} method: {e}")

                    self._resolve_size_change_futures()
                    return result
                return wrapped_method
            return method
        else:
            raise AttributeError(f"'ObservableSet' object has no attribute '{attr}'")
        
    
    def __len__(self):
        return len(self.__value)
    
    def extract(self) -> set:
        return self.__value.copy()
    
