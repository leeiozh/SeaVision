# Runs before any src.* module (and therefore before numpy/scipy) is imported:
# keeps the stdlib from spawning cmd.exe at startup — see winguard.py.
from src.runtime.winguard import disable_cmd_probe

disable_cmd_probe()
