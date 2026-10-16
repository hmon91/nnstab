"""nnstab: exponential stability of positive feedback loops with a neural network.

Give a plant x' = A x + B u, y = C x (B >= 0, C >= 0) and a trained
feedforward network u = pi(y) with pi(0) = 0. nnstab computes sector bounds
of the network with nnsector and checks, for initial states x0 >= 0, global
exponential stability first; if that cannot be shown, it looks for a local
region of attraction.

>>> import nnstab as nst
>>> A, B, C, net = nst.example_system("local")
>>> res = nst.verify(A, B, C, net)
>>> res.status
'local'
"""

from .checks import (GlobalCheck, LocalCheck, as_network, check_global, check_local,
                     check_system)
from .demo import example_controller, example_system
from .linalg import (copositive_vector, hurwitz_certificate, is_hurwitz, is_metzler,
                     largest_region_vector, perron_vector, spectral_abscissa)
from .simulate import sample_region, simulate
from .verify import Verification, verify

__version__ = "0.1.0"

__all__ = [
    "GlobalCheck",
    "LocalCheck",
    "Verification",
    "as_network",
    "check_global",
    "check_local",
    "check_system",
    "copositive_vector",
    "example_controller",
    "example_system",
    "hurwitz_certificate",
    "is_hurwitz",
    "is_metzler",
    "largest_region_vector",
    "perron_vector",
    "sample_region",
    "simulate",
    "spectral_abscissa",
    "verify",
]
