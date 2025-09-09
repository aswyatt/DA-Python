""" Import this module to add python-lib to the python path in jupyter notebooks"""

import sys
from pathlib import Path

sys.path.append(
    str(
        Path(list(filter(lambda x: ".venv" in x, sys.path))[0])
        .joinpath("..", "python-lib")
        .resolve()
    )
)
