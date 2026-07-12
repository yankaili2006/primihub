
class LinkMode:
    GRPC = 0

class Node:
    def __init__(self, ip, port, use_tls, nodename='default'):
        self.ip = ip
        self.port = port
        self.use_tls = use_tls
        self.nodename = nodename

class Channel:
    def __init__(self):
        self._store = {}
    def send(self, key, val):
        self._store[key] = val
    def recv(self, key):
        return self._store.get(key)

class LinkContext:
    def __init__(self):
        self._channels = {}
    def setTaskInfo(self, task_id, job_id, request_id, sub_task_id):
        pass
    def getChannel(self, session):
        return Channel()
    def initCertificate(self, root_ca, key, cert):
        pass

class LinkFactory:
    @staticmethod
    def createLinkContext(mode):
        return LinkContext()
