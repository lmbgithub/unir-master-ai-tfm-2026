import time

from urgenurse.agents.agent.process import ProcessManager


def _immediate_exit() -> None:
    pass


def _long_running() -> None:
    time.sleep(60)


def test_restarts_dead_worker() -> None:
    pm = ProcessManager(target=_immediate_exit, workers=1)
    pm.start()

    import threading

    t = threading.Thread(target=pm.monitor, daemon=True)
    t.start()

    time.sleep(5)
    pm.stop()

    assert len(pm._processes) == 0


def test_stop_kills_all_processes() -> None:
    pm = ProcessManager(target=_long_running, workers=2)
    pm.start()

    pids = [p.pid for p in pm._processes]
    assert all(p.is_alive() for p in pm._processes)

    pm.stop()

    import os

    for pid in pids:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        assert not alive
