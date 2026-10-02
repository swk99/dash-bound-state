"""Experiment-only TCP delay proxy for Redis (no netem in this kernel).
Each chunk read in a direction is released at recv_time + max(0, N(delay, jitter)) (clipped at 0),
but never before the previous chunk of the same direction (FIFO; no reordering).
No bandwidth limit, no loss. Seeded numpy Generator per direction.
usage: delay_proxy.py LISTEN_PORT UPSTREAM_PORT DELAY_MS JITTER_SD_MS SEED"""
import socket, sys, threading, time, queue, numpy as np
LP, UP, DLY, JIT, SEED = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) / 1e3, float(sys.argv[4]) / 1e3, int(sys.argv[5])
def precise_sleep_until(t):
    while True:
        dt = t - time.perf_counter()
        if dt <= 0: return
        if dt > 0.0003: time.sleep(dt - 0.0002)   # coarse sleep then short spin
def pipe(src, dst, rng):
    q = queue.SimpleQueue(); last = [0.0]
    def reader():
        while True:
            try: b = src.recv(65536)
            except OSError: b = b''
            t = time.perf_counter()
            d = max(0.0, rng.normal(DLY, JIT)) if JIT > 0 else DLY
            rel = max(t + d, last[0]); last[0] = rel
            q.put((rel, b))
            if not b: return
    def writer():
        while True:
            rel, b = q.get()
            if not b:
                try: dst.shutdown(socket.SHUT_WR)
                except OSError: pass
                return
            if rel > time.perf_counter(): precise_sleep_until(rel)
            try: dst.sendall(b)
            except OSError: return
    threading.Thread(target=reader, daemon=True).start(); threading.Thread(target=writer, daemon=True).start()
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(('127.0.0.1', LP)); srv.listen(64)
print('ready', flush=True); cid = 0
while True:
    c, _ = srv.accept(); u = socket.create_connection(('127.0.0.1', UP))
    for s in (c, u): s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    pipe(c, u, np.random.default_rng([SEED, cid, 0])); pipe(u, c, np.random.default_rng([SEED, cid, 1])); cid += 1
