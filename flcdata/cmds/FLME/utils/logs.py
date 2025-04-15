import traceback
import sys

def stderr(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)

def print_except(e):
    if sys.stdout.isatty():
        FG_RED = "\033[31m"
        FG_GREEN = "\033[32m"
        FG_YELLOW = "\033[33m"
        FG_BLUE = "\033[34m"
        FG_MAGENTA = "\033[35m"
        FG_CYAN = "\033[36m"
        FG_WHITE = "\033[37m"
        FG_RESET = "\033[0m"
    else:
        FG_RED = ""
        FG_GREEN = ""
        FG_YELLOW = ""
        FG_BLUE = ""
        FG_MAGENTA = ""
        FG_CYAN = ""
        FG_WHITE = ""
        FG_RESET = ""

    stderr(f"{FG_RED}================================")
    stderr(f"{FG_MAGENTA}{type(e)}{FG_RED}: {FG_YELLOW}{e}{FG_RED}")
    stderr()
    stderr(f"Error Traceback: {traceback.format_exc()}")
    stderr(f"================================{FG_RESET}")
