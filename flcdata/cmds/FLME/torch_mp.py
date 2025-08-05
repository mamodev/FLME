class MasterSlaveCommunicator:
    def __init__(self, ctx, n: int):
        self.n = n
        self.barrier = ctx.Barrier(n)
        self.queue = ctx.Queue()
    
    def recv_broadcast(self):
        self.barrier.wait()
        res = self.queue.get()       
        self.barrier.wait()
        return res
    
    def send_broadcast(self, item):
        self.barrier.wait()
        for _ in range(self.n - 1):
            self.queue.put(item)
        self.barrier.wait()
        
    def send_gather(self, item):
        self.barrier.wait()
        self.queue.put(item)
        self.barrier.wait()
        
    def recv_gather(self, with_my_data=None):
        self.barrier.wait()
        res = []
        if with_my_data is not None:
            res.append(with_my_data)
            
        for _ in range(self.n - 1):
            res.append(self.queue.get())
        self.barrier.wait()
        return res
    
    def all_to_all(self, idx, item):
        results = [item]
        for i in range(self.n):
            if i == idx:
                res=self.recv_broadcast()
                results.append(res)
            else:
                self.send_broadcast(item)
                
        return results
    
    def f_all_to_all(self, idx, item):
        results = [item]
        for i in range(self.n):
            if i == idx:
                res = self.recv_gather()
                results.append(res)
            else:
                self.send_gather(item)

        return [r for r in results if r is not None]


    