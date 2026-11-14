import os, tempfile, time, multiprocessing as mp
from pathlib import Path
from podcue.store import Episode, writing

def writer(root, iterations, out):
    ep = Episode(Path(root), "ep_0123456789abcdef")
    marker = ep.dir / "inside"
    overlaps = 0
    for _ in range(iterations):
        with writing(ep):
            try:
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                overlaps += 1
                continue
            time.sleep(0.001)
            os.close(fd); os.unlink(marker)
    out.put(overlaps)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    root = tempfile.mkdtemp()
    Episode(Path(root), "ep_0123456789abcdef").dir.mkdir(parents=True)
    out = mp.Queue()
    ps = [mp.Process(target=writer, args=(root, 150, out)) for _ in range(12)]
    [p.start() for p in ps]
    total = sum(out.get() for _ in ps); [p.join() for p in ps]
    print(f"{total} times a writer entered while another was inside")
