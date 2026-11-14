import os, tempfile, time, fcntl, multiprocessing as mp
from pathlib import Path
from podcue.store import try_lock

def orphan(path, ready):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.ftruncate(fd, 0); os.write(fd, b"999999\n")
    ready.set(); time.sleep(30)

def contender(path, barrier, out):
    barrier.wait()
    fd = try_lock(Path(path))
    out.put(fd is not None)
    time.sleep(2)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    worst = 0
    for r in range(20):
        path = str(Path(tempfile.mkdtemp()) / "start.lock")
        ready = mp.Event(); o = mp.Process(target=orphan, args=(path, ready)); o.start(); ready.wait()
        barrier, out = mp.Barrier(8), mp.Queue()
        ps = [mp.Process(target=contender, args=(path, barrier, out)) for _ in range(8)]
        [p.start() for p in ps]
        winners = sum(out.get() for _ in ps)
        [p.join() for p in ps]; o.terminate()
        worst = max(worst, winners)
        print(f"round {r}: {winners} of 8 starts took the stale lock")
    print("BUG: more than one start holds the lock" if worst > 1 else "ok")
