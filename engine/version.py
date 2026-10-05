"""One place for the version numbers every GradMesh process reports.

`__version__` is the product release. `PROTOCOL` is the wire contract between a
worker agent and the coordinator, and it only moves when one side would
misunderstand the other: a worker on a newer protocol than its coordinator is
told to rejoin from the host's join page rather than half working.

Stdlib only. The join flow imports this before any dependency is installed.
"""

__version__ = "5.1.1"

# 5: binary weight transfer, image-cache shards, phase timings, instance ids.
PROTOCOL = 5

# The reference software stack every backend is pinned to. Results from an
# NVIDIA, an Intel and an Apple machine are only comparable when they ran the
# same framework, so the pins live here and the requirement files follow them.
REFERENCE_STACK = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "ultralytics": "8.4.46",
}

# Interpreters PyTorch 2.13 ships wheels for on every backend GradMesh supports.
SUPPORTED_PYTHON = ((3, 10), (3, 13))
PREFERRED_PYTHON = (3, 12)
