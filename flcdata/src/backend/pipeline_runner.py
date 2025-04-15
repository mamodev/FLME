import os 
import sys
import json
import subprocess

print("Pipeline runner started", file=sys.stderr, flush=True)
print("Pipeline runner started", file=sys.stdout, flush=True)

example_json = {
    "type": "sequential",
    "ignoreErrors": False,
    "steps": [
        ["python", "-c", "print('Hello world')"],
        ["python", "-c", "import time; time.sleep(1); print('Hello world 2')"],
        {
            "type": "parallel",
            "steps": [
                ["python", "-c", "import time; time.sleep(1); print('Hello world 3')"],
                ["python", "-c", "import time; time.sleep(1); print('Hello world 4')"],
            ]
        }
    ],
}


def isCmdInPath(cmd):
    return os.system(f"command -v {cmd} > /dev/null 2>&1") == 0

def isDict(obj):
    return isinstance(obj, dict)

def isList(obj):
    return isinstance(obj, list)

def isNonEmptyList(obj):
    return isinstance(obj, list) and len(obj) > 0

def isNonEmptyDict(obj):
    return isinstance(obj, dict) and len(obj) > 0

def isValidBlockType(t):
    if type(t) is not str:
        print(f"Block type {t} is not a string")
        return False
    if t not in ["sequential", "parallel"]:
        print(f"Block type {t} is not valid")
        return False
    
    return True

def isValidStep(step):
    if isList(step):
        if not isNonEmptyList(step):
            print("Step is empty")
            return False

        if not isCmdInPath(step[0]):
            print(f"Command {step[0]} not found in PATH")
            return False
        
        return True

    elif isDict(step):
        if not isValidBlock(step):
            print(f"Step is not a valid block: \n{json.dumps(step, indent=4)}")
            return False
        
        return True
    else:
        print("Step is not a list or dict")
        return False

def isValidBlock(block):
    if not isDict(block):
        print("Block is not a dict")
        return False
    if not isNonEmptyDict(block):
        print("Block is empty")
        return False
    if not "type" in block or not isValidBlockType(block["type"]):
        print("Block type is not valid")
        return False
    if not "steps" in block:
        print(f"Block does not have steps, \n{json.dumps(block, indent=4)}")
        return False
    if "ignoreErrors" in block and not isinstance(block["ignoreErrors"], bool):
        print("Block ignoreErrors is not valid")
        return False

    for step in block["steps"]:
        if not isValidStep(step):
            print(f"Step is not valid: \n{json.dumps(step, indent=4)}")
            return False    
    
    return True


def runBlock(block):
    if block["type"] == "sequential":
        for step in block["steps"]:
            if isList(step):
                cmd = step[0]
                args = step[1:]
                try:
                    subprocess.run([cmd] + args, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running command: {e}")
                    if not block.get("ignoreErrors", False):
                        raise e
            elif isDict(step):
                runBlock(step)
            else:
                raise ValueError(f"Invalid step: {step}")
    elif block["type"] == "parallel":
        processes = []
        for step in block["steps"]:
            if isList(step):
                cmd = step[0]
                args = step[1:]
                try:
                    p = subprocess.Popen([cmd] + args)
                    processes.append(p)
                except subprocess.CalledProcessError as e:
                    print(f"Error running command: {e}")
                    if not block.get("ignoreErrors", False):
                        raise e
            elif isDict(step):
                runBlock(step)
            else:
                raise ValueError(f"Invalid step: {step}")
        for p in processes:
            p.wait()
    else:
        raise ValueError(f"Invalid block type: {block['type']}")


def pipeline_runner(arg):
    if arg.endswith(".json"):
        try:
            with open(arg, "r") as f:
                pipeline = json.load(f)
        except json.JSONDecodeError:
            raise Exception("Invalid json file")
        except FileNotFoundError:
            raise Exception(f"File {arg} not found")
        except Exception as e:
            raise Exception(f"Error reading file {arg}: {e}")
    
    else:
        try:
            pipeline = json.loads(arg)
        except json.JSONDecodeError:
            raise Exception("Invalid json string")
        except Exception as e:
            raise Exception(f"Error parsing json string: {e}")

    if not isValidBlock(pipeline):
        raise Exception("Invalid pipeline")

    try:
        runBlock(pipeline)
    except Exception as e:
        raise Exception(f"Error running pipeline: {e}")

    print("Pipeline finished successfully")

def stderr(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

if __name__ == "__main__":
    # ==========================================
    # CRITICAL SECTION
    # ==========================================

    if len(sys.argv) != 2 and len(sys.argv) != 3:
        script_name = os.path.basename(__file__)
        print(f"Usage: python {script_name} <pipeline.json> [<log-file>]")
        print("Json format: ")
        print(json.dumps(example_json, indent=4))
        sys.exit(1)

    arg = sys.argv[1]
    pid_file = sys.argv[2] + ".pid" if len(sys.argv) == 3 else None
    status_file = sys.argv[2] + ".status" if len(sys.argv) == 3 else None
    if pid_file:
        try:
            with open(pid_file, "w") as f:
                f.write(str(os.getpid()))
        except Exception as e:
            print(f"Error writing pid file: {e}")
            sys.exit(1)

    if status_file:
        try:
            with open(status_file, "w") as f:
                pass
        except Exception as e:
            print(f"Error writing status file: {e}")
            sys.exit(1)
    
    # ==========================================
    # CRITICAL SECTION
    # ==========================================
    errored = False
    try:
        pipeline_runner(arg)
    except Exception as e:
        import traceback
        print("", file=sys.stderr, flush=True)
        print("", file=sys.stderr, flush=True)
        print(f"PipelineRunner Error: {e}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print("", file=sys.stderr, flush=True)
        print("", file=sys.stderr, flush=True)
        errored = True
        pass
    finally:
        if status_file:
            try:
                with open(status_file, "w") as f:
                    f.write("ERROR" if errored else "SUCCESS")
            except Exception as e:
                stderr("Error writing status file: " + str(e))