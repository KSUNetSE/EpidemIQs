import inspect
import json
from functools import wraps
from datetime import datetime
from termcolor import colored
from inspect import signature

LOG_FILE = 'function_history.json'

def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        arguments = bound_args.arguments
        
        func_info = {
            "Function Name": function_name,
            "Arguments": arguments,
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(colored(f"Logging function call: {function_name}", "yellow"))
        
        try:
            with open(LOG_FILE, 'r+') as log_file:
                try:
                    log_file.seek(0)
                    history = json.load(log_file)
                except json.JSONDecodeError:
                    history = []
                history.append(func_info)
                log_file.seek(0)
                json.dump(history, log_file, indent=4)
                log_file.truncate()
        except Exception as e:
            print(f"Error logging function call: {e}")
        
        return func(*args, **kwargs)
    
    return wrapper

@log_function_call
def some_function(a, b):
    print(a + b)

@log_function_call
def another_function(x, y, z):
    print(x * y * z)
if __name__ == '__main__':  
    some_function(1, 2)
    another_function(2, 3, 4)