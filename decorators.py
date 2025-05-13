import time
from functools import wraps

def timing(label="Method"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"{label}: {elapsed:.4f} seconds")
            return result
        return wrapper
    return decorator