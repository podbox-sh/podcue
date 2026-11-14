import os, tempfile, time, multiprocessing as mp
from pathlib import Path
from podcue.store import Episode, try_lock, release

def writer(root, iterations, out):
    ep = Episode(Path(root), "ep_0123456789abcdef")
    stale = 0
    for _ in range(iterations):
        while True:
            fd = try_lock(ep.write_lock)
            if fd is not None:
                break
            time.sleep(0.001)
        held = os.fstat(fd)
        try:
            current = os.stat(ep.write_lock)
            match = held.st_ino == current.st_ino and held.st_dev == current.st_dev
        except FileNotFoundError:
            match = False
        if not match:
            stale += 1
        time.sleep(0.001)
        release(fd, ep.write_lock)
    out.put(stale)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    root = tempfile.mkdtemp()
    Episode(Path(root), "ep_0123456789abcdef").dir.mkdir(parents=True)
    out = mp.Queue()
    ps = [mp.Process(target=writer, args=(root, 150, out)) for _ in range(12)]
    [p.start() for p in ps]
    total = sum(out.get() for _ in ps)
    [p.join() for p in ps]
    print(f"{total} locks held on an inode that was no longer write.lock")
    print("ok" if total == 0 else "BUG")
