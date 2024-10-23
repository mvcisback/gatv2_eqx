# ruff: noqa: F403
try:
    from beartype.claw import beartype_this_package
    beartype_this_package()
except ModuleNotFoundError:
    pass
from gatv2_eqx.model import *
