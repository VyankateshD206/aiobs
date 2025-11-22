import warnings

warnings.warn(
    "Package 'llm_observability' has been renamed to 'aiobs'. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from aiobs import *  # noqa: F401,F403

