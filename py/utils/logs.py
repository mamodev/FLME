
import traceback

FG_RED = "\033[31m"
FG_GREEN = "\033[32m"
FG_YELLOW = "\033[33m"
FG_BLUE = "\033[34m"
FG_MAGENTA = "\033[35m"
FG_CYAN = "\033[36m"
FG_WHITE = "\033[37m"
FG_RESET = "\033[0m"

def print_except(e):
    print(f"{FG_RED}================================")
    print(f"{FG_MAGENTA}{type(e)}{FG_RED}: {FG_YELLOW}{e}{FG_RED}")
    # print(f"Error Message: {e}")
    # print(f"Error Arguments: {e.args}")
    print()
    print(f"Error Traceback: {traceback.format_exc()}")
    print(f"================================{FG_RESET}")