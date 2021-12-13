# import ray
# import cma

# @ray.remote
# def evaluate(x):
#     return cma.ff.rosen(x)

# es = cma.CMAEvolutionStrategy(12 * [0], 0.5)
# while not es.stop():
#     solutions = es.ask()
#     remotes = [evaluate.remote(x) for x in solutions]
#     es.tell(solutions, ray.get(remotes))
#     es.logger.add()  # write data to disc to be plotted
#     es.disp()


# trainer.py
from collections import Counter
import os
import socket
import sys
import time
import ray

num_cpus = 5

ray.init(address=os.environ["ip_head"])

print("Nodes in the Ray cluster:")
print(ray.nodes())
print("----------------")


@ray.remote
def f():
    time.sleep(1)
    # ~ print("x")
    return socket.gethostbyname(socket.gethostname())


# The following takes one second (assuming that
# ray was able to access all of the allocated nodes).
for i in range(4):
    start = time.time()
    ip_addresses = ray.get([f.remote() for _ in range(num_cpus)])
    print(Counter(ip_addresses))
    end = time.time()
    print(end - start)