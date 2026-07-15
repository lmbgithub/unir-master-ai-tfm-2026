import multiprocessing
import signal
import threading
import time
from collections.abc import Callable


class ProcessManager:
    def __init__(self, target: Callable, workers: int = 1) -> None:
        self._target = target
        self._workers = workers
        self._processes: list[multiprocessing.Process] = []
        self._running = False

    def start(self) -> None:
        self._running = True
        for _ in range(self._workers):
            self._spawn()

    def stop(self) -> None:
        self._running = False
        for p in self._processes:
            if p.is_alive():
                p.terminate()

        deadline = time.monotonic() + 5
        for p in self._processes:
            remaining = max(0, deadline - time.monotonic())
            p.join(timeout=remaining)
            if p.is_alive():
                p.kill()

        self._processes.clear()

    def monitor(self) -> None:
        def _handle_signal(signum: int, frame: object) -> None:
            self.stop()
            raise SystemExit(0)

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, _handle_signal)

        try:
            while self._running:
                dead = [p for p in self._processes if not p.is_alive()]
                for p in dead:
                    self._processes.remove(p)
                    self._spawn()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _spawn(self) -> None:
        p = multiprocessing.Process(target=self._target, daemon=True)
        p.start()
        self._processes.append(p)
