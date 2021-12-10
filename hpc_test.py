import ray
import cma

@ray.remote
def evaluate(x):
    return cma.ff.rosen(x)

es = cma.CMAEvolutionStrategy(12 * [0], 0.5)
while not es.stop():
    solutions = es.ask()
    remotes = [evaluate.remote(x) for x in solutions]
    es.tell(solutions, ray.get(remotes))
    es.logger.add()  # write data to disc to be plotted
    es.disp()