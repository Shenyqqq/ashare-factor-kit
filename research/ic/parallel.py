"""Bounded thread-pool parallel helper."""
from __future__ import annotations


def run_bounded_parallel(fn, items: list, n_workers: int, progress_every: int = 0):
    """
    At most n_workers tasks in flight. n_workers=1 → serial (32GB safe default).
    """
    n_workers = max(1, min(n_workers, len(items) or 1))
    if n_workers == 1:
        for i, item in enumerate(items, 1):
            yield fn(item)
            if progress_every and (i % progress_every == 0 or i == len(items)):
                print(f"  进度: {i}/{len(items)}", flush=True)
        return

    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    it = iter(items)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pending: set = set()
        for _ in range(n_workers):
            try:
                pending.add(pool.submit(fn, next(it)))
            except StopIteration:
                break
        done_count = 0
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                yield fut.result()
                done_count += 1
                if progress_every and (done_count % progress_every == 0):
                    print(f"  进度: {done_count}/{len(items)}", flush=True)
                try:
                    pending.add(pool.submit(fn, next(it)))
                except StopIteration:
                    pass
        if progress_every and done_count:
            print(f"  进度: {done_count}/{len(items)}", flush=True)
