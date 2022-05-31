
import functools
from typing import Callable
from dill import dumps, loads
# from pickle import dumps, loads


class NodeContext:
    def __init__(self, role, protocol, datasets, func=None):
        self.role = role
        self.protocol = protocol
        self.datasets = datasets
        print("func type: ", type(func))

        self.dumps_func = None
        if isinstance(func, Callable):
            # pikle dumps func 
            self.dumps_func = dumps(func)
        elif type(func) == str:
            self.dumps_func = func

        if self.dumps_func:
            print("dumps func:", self.dumps_func)


class TaskContext:
    """ key: role, value:NodeContext """
    nodes_context = dict()
    dataset_service = None
    datasets = []
    # dataset meta information
    dataset_map = dict()

    def get_protocol(self):
        """Get current task support protocol.
           NOTE: Only one protocol is supported in one task now.
            Maybe support mix protocol in one task in the future.
        Returns:
            string: protocol string
        """
        procotol = None
        try:
            protocol = list(self.nodes_context.values())[0].protocol
        except IndexError:
            protocol = None

        return protocol

    def get_roles(self):
        return list(self.nodes_context.keys())

    def get_datasets(self):
        return self.datasets
    
    
Context = TaskContext()

def set_node_context(role, protocol, datasets):
    print("========", role, protocol, datasets)
    n = NodeContext(role, protocol, datasets)
    # Context.nodes_context[role] = NodeContext(role, protocol, datasets)


def set_text(role, protocol, datasets, dumps_func):
    print("========", role, protocol, datasets, dumps_func)

# def set_node_context(node_context: NodeContext):
#     Context.nodes_context[node_context.role] = node_context

# Register dataset decorator
def reg_dataset(func):
    @functools.wraps(func)
    def reg_dataset_decorator(dataset):
        print("Register dataset:", dataset)
        Context.datasets.append(dataset)        
        return func(dataset)
    return reg_dataset_decorator


# Register task decorator
def function(protocol, role, datasets):
    def function_decorator(func):
        print("Register task:", func.__name__)
        Context.nodes_context[role] = NodeContext(role, protocol, datasets, func)
        @functools.wraps(func)
        def wapper(*args, **kwargs):   
            return func(*args, **kwargs)
        return wapper
    return function_decorator