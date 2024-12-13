import os
from typing import Final

_PROJECT_ROOT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def get_root_path():
    return _PROJECT_ROOT