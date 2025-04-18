import json
import os
import sys
import uuid
import time

class PipelineManager():
    def __init__(self, status_file="pipeline_status.json", log_folder="logs"):
        if os.name != "posix":
            raise Exception("This code is only supported on POSIX systems")

        self.status_file = status_file
        self.log_folder = log_folder

        self._ensure_files()

        self.status = {}
        self._load_status()

    def _ensure_files(self):
        os.makedirs(self.log_folder, exist_ok=True)
        if not os.path.exists(self.status_file):
            with open(self.status_file, "w") as f:
                json.dump({}, f)

    def _load_status(self):
        with open(self.status_file, "r") as f:
            self.status = json.load(f)

    def _save_status(self):
        with open(self.status_file, "w") as f:
            json.dump(self.status, f, indent=4)


    def __double_fork(self):
        pid = os.fork()
        if pid < 0:
            raise Exception("Fork failed")
        if pid > 0:
            return pid

        os.setsid()
        os.umask(0)

        pid = os.fork()
        if pid < 0:
            raise Exception("Fork failed")
        if pid > 0:
            os._exit(0)

        return 0
    
    def __wait_file_sync(self, file_path):
        while not os.path.exists(file_path):
            time.sleep(0.1)
        
        time.sleep(0.1)
        with open(file_path, "r") as f:
            return f.read().strip()
    

    def get_pipeline_status(self, id):
        assert isinstance(id, str), "Pipeline id must be a string"
        if id not in self.status:
            raise Exception(f"Pipeline with id {id} not found")

        path = os.path.join(self.log_folder, f"{id}.status")
        if not os.path.exists(path):
            return "UNKNOWN", 0

        # get file last modification time
        last_mod_time = os.path.getmtime(path)

        with open(os.path.join(self.log_folder, f"{id}.status"), "r") as f:
            status = f.read().strip()

        if status == "":
            status = "RUNNING"

        return status, last_mod_time
    
    def get_pipeline_config(self, id):
        assert isinstance(id, str), "Pipeline id must be a string"
        if id not in self.status:
            raise Exception(f"Pipeline with id {id} not found")

        return self.status[id]["config"]

    def list_pipelines(self):
        pipelines = []
        for id, data in self.status.items():
            status = self.get_pipeline_status(id)

            pipeline = {
                "id": id,
                "pid": data["pid"],
                "start_time": data["start_time"],
                "status": status[0],
                "last_status_change": status[1],
                "config": data["config"],
                "dex": data["dex"],
            }
        
            pipelines.append(pipeline)
        
        return pipelines
    
    def get_pipeline_log(self, id):
        assert isinstance(id, str), "Pipeline id must be a string"
        if id not in self.status:
            raise Exception(f"Pipeline with id {id} not found")

        stdout_file = os.path.join(self.log_folder, f"{id}.stdout")
        stderr_file = os.path.join(self.log_folder, f"{id}.stderr")

        if not os.path.exists(stdout_file):
            raise Exception(f"Pipeline stdout file {stdout_file} not found")
        
        if not os.path.exists(stderr_file):
            raise Exception(f"Pipeline stderr file {stderr_file} not found")

        with open(stdout_file, "r") as f:
            stdout_lines = f.readlines()
        
        with open(stderr_file, "r") as f:
            stderr_lines = f.readlines()

        return {
            "stdout": stdout_lines,
            "stderr": stderr_lines,
        }

    
    def cleanup_pipeline(self, id) -> bool:
        assert isinstance(id, str), "Pipeline id must be a string"
        if id not in self.status:
            raise Exception(f"Pipeline with id {id} not found")
        
        # check if not running
        status = self.get_pipeline_status(id)
        if status[0] == "RUNNING":
            return False
        
        # remove all files
        for ext in ["stdout", "stderr", "status", "pid"]:
            try:
                os.remove(os.path.join(self.log_folder, f"{id}.{ext}"))
            except Exception as e:
                pass

        # remove from status
        del self.status[id]
        self._save_status()
        
        return True        
    
    def rerun_pipeline(self, id):
        assert isinstance(id, str), "Pipeline id must be a string"
        if id not in self.status:
            raise Exception(f"Pipeline with id {id} not found")
        
        # check if not running
        status = self.get_pipeline_status(id)
        if status[0] == "RUNNING":
            raise Exception(f"Pipeline with id {id} is still running")

        config = self.status[id]["config"]
        dex = self.status[id]["dex"]

        return self.start_pipeline(config, dex)

    def start_pipeline(self, config, dex=None):
        assert isinstance(config, str), "Pipeline config must be a string"

        id = str(uuid.uuid4())

        log_base = os.path.join(self.log_folder, id)

        pid = self.__double_fork()
        if pid > 0:
            pid = self.__wait_file_sync(f"{log_base}.pid")
            self.status[id] = {
                "id": id,
                "pid": pid,
                "start_time": time.time(),
                "config": config,
                "dex": dex if dex else "---",
            }

            self._save_status()
            return self.status[id]
        
        # Child process
        thisdir = os.path.dirname(os.path.abspath(__file__))
        runner_cmd = os.path.join(thisdir, "pipeline_runner.py")

        stdout_file = f"{log_base}.stdout"
        stderr_file = f"{log_base}.stderr"

        try:
            stderr_fd = os.open(stderr_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
            stdout_fd = os.open(stdout_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        except Exception as e:
            print(f"Error opening log files: {e}")
            exit(1)

        try:
            os.close(0)
            os.close(1)
            os.close(2)
        except:
            pass

        if os.dup2(stderr_fd, 2) == -1:
            print(f"Error redirecting stderr to {stderr_file}")
            os.close(stderr_fd)
            exit(1)
        
        if os.dup2(stdout_fd, 1) == -1:
            print(f"Error redirecting stdout to {stdout_file}")
            os.close(stderr_fd)
            exit(1)

        os.close(stderr_fd)
        os.close(stdout_fd)

        print(f"Starting pipeline with id {id} and pid {os.getpid()}", file=sys.stdout)
        print(f"Starting pipeline with id {id} and pid {os.getpid()}", file=sys.stderr)

        os.execvp("python", ["python", runner_cmd, config, log_base])
        print(f"Error executing pipeline runner: {runner_cmd}", file=sys.stderr, flush=True)
        print(f"Error executing pipeline runner: {runner_cmd}", file=sys.stdout, flush=True)
        os._exit(1)

    def start_from_template(self, templ, args=None):
        assert isinstance(args, dict) or args is None, "Pipeline args must be a dictionary or None"
        assert isinstance(templ, str) or isinstance(templ, dict), "Pipeline template must be a string or a dictionary"

        if isinstance(templ, str):
            # if ends with .json, load as json
            if templ.endswith(".json"):
                dex = templ
                with open(templ, "r") as f:
                    templ = json.load(f)
            else:
                dex = "custom-json"
                templ = json.loads(templ)
        else:
            dex = "python dict"

        if "__DEX__" in args:
            dex = args["__DEX__"]
            del args["__DEX__"]

        def replace_args(string, args_dict):
            for key, value in args_dict.items():
                string = string.replace(f"@@{key}", str(value))
            
            return string

        def rec_replace(obj, args_dict = {}):
            if isinstance(obj, dict):
                obj2 = {}
                for key, value in obj.items():
                    obj2[rec_replace(key, args_dict)] = rec_replace(value, args_dict)          

                return obj2

            elif isinstance(obj, list):
                obj2 = []
                for item in obj:
                    obj2.append(rec_replace(item, args_dict))
                
                return obj2
            
            elif isinstance(obj, str):
                return replace_args(obj, args_dict)
            else:
                return obj

        if args is None:
            args = {}

        # override default args with args
        for key, value in templ["parameters"].items():
            if key in args:
                templ["parameters"][key] = args[key]

        pipeline = rec_replace(templ["pipeline"], templ["parameters"])
        pipeline = json.dumps(pipeline, indent=4)

        return self.start_pipeline(pipeline, dex)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pp_manager.py <pipeline.json>")
        sys.exit(1)

    pipeline_config = sys.argv[1]
    manager = PipelineManager(
        status_file=".pipelines_logs/pipeline_status.json",
        log_folder=".pipelines_logs"
    )

    manager.start_pipeline(pipeline_config)
    
    while True:
        pipelines = manager.list_pipelines()
        for pipeline in pipelines:
            print(f"Pipeline {pipeline['id']} (pid: {pipeline['pid']}) is {pipeline['status']}")

        if all(pipeline['status'] != "RUNNING" for pipeline in pipelines):
            break

    print()
    print()

    pipelines = manager.list_pipelines()
    for pipeline in pipelines:
        ret = manager.cleanup_pipeline(pipeline['id'])
        if ret:
            print(f"Pipeline {pipeline['id']} cleaned up")
        else:
            print(f"Pipeline {pipeline['id']} is still running, not cleaned up")

        print()
