import time
import ray

@ray.remote
def f():
    time.sleep(1)
    return 1

ray.init()
results = ray.get([f.remote() for i in range(4)])
print(results)