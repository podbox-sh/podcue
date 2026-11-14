import os, sys, time
from pathlib import Path
from podcue.store import try_lock, release

path = Path(sys.argv[1])
fd = try_lock(path)
assert fd is not None
child = os.fork()
if child == 0:
    time.sleep(2)
    os._exit(0)
release(fd, path)
second = try_lock(path)
print("while child lives", "none" if second is None else second)
os.kill(child, 15)
os.waitpid(child, 0)
third = try_lock(path)
print("after child exits", "taken" if third is not None else "none")
if third is not None:
    release(third, path)
