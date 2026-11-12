"""Shared-world controller bound to all three robots in scene.yaml.

The kernel owns lifecycle/pausing; each advance executes one control tick on
the kernel's own model and data. No subprocess replicas or second world.
"""
import sys
from demo.maintenance_relay.core import Demo

class RelayController:
    def __init__(self, simulation):
        self.demo = Demo(simulation=simulation, log_stream=sys.stderr)
        self.iterator = self.demo.run()

    def advance(self):
        if self.demo.done:
            return False
        try:
            next(self.iterator)
            return True
        except StopIteration:
            return False

    @property
    def done(self):
        return self.demo.done

    @property
    def success(self):
        return self.demo.success

    @property
    def detail(self):
        return self.demo.detail

    def close(self):
        self.iterator.close()
        self.demo.close()

def create_controller(simulation):
    return RelayController(simulation)
