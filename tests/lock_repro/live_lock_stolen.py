import os, tempfile, time, multiprocessing as mp
from pathlib import Path
from podcue.store import try_lock, lock_pid, pid_alive

def holder(path, ready):
    fd = try_lock(Path(path))
    assert fd is not None
    ready.set()
    time.sleep(30)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    path = Path(tempfile.mkdtemp()) / "start.lock"
    ready = mp.Event()
    a = mp.Process(target=holder, args=(str(path), ready)); a.start(); ready.wait()
    print("holder pid", a.pid, "alive:", pid_alive(a.pid), "pid in file:", lock_pid(path))
    print("second start while holder is young:", try_lock(path))
    old = time.time() - 13 * 3600
    os.utime(path, (old, old))
    fd = try_lock(path)
    print("second start after 13 h of mtime age:", fd, "| holder still alive:", a.is_alive())
    print("BUG: two live starts hold start.lock" if fd is not None and a.is_alive() else "ok")
    a.terminate()
