import json
import os
import subprocess
import time
import uuid
import shlex
import psutil
from typing import Dict, List, Tuple, Optional, Any, Callable

class PipelineManager:
    def __init__(self, log_folder: str = "pipeline_logs", state_file: str = "pipeline_state.json"):
        """Initialize the PipelineManager.
        
        Args:
            log_folder: Path to the folder where pipeline logs will be stored.
            state_file: Path to the state file where pipeline information is stored.
        """
        self.log_folder = os.path.abspath(log_folder)  # Use absolute path
        self.state_file = state_file
        self.pipelines = {}  # Dictionary to store running pipeline information
        self.callbacks = {}  # Dictionary to store completion callbacks
        
        # Ensure log folder exists
        os.makedirs(self.log_folder, exist_ok=True)
        
        # Load existing state
        self._load_state()
        
        # Check for terminated pipelines on startup
        self._process_terminated_pipelines()
    
    def _load_state(self) -> None:
        """Load pipeline state from the state file if it exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.pipelines = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.pipelines = {}
    
    def _save_state(self) -> None:
        """Save the current pipeline state to the state file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.pipelines, f, indent=2)
    
    def _process_terminated_pipelines(self) -> None:
        """Check for terminated pipelines, process their status, execute callbacks, then clean them up."""
        terminated_pipelines = []
        
        for pipeline_id, pipeline_info in list(self.pipelines.items()):
            pid = pipeline_info.get('pid')
            if pid is None or not self._is_process_running(pid):
                # Process is not running, get status and add to terminated list
                status = self._get_pipeline_status(pipeline_id)
                terminated_pipelines.append((pipeline_id, status))
        
        # Process terminated pipelines - first execute callbacks, then clean up
        for pipeline_id, status in terminated_pipelines:
            print(f"Pipeline {pipeline_id} has terminated with status {status}")
            self._execute_callback(pipeline_id, status)
    
    def _get_pipeline_status(self, pipeline_id: str) -> int:
        """Get the exit status of a pipeline.
        
        Args:
            pipeline_id: The UUID of the pipeline.
            
        Returns:
            The exit status (0 for success, negative for error, -1 if unknown).
        """
        status_file = os.path.join(self.log_folder, f"pipeline_{pipeline_id}.status")
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    status_text = f.read().strip()
                    return int(status_text) if status_text else -1
            except (IOError, ValueError):
                return -9999  # Error reading status file
        return -8888  # No status file found
    
    def _execute_callback(self, pipeline_id: str, status: int) -> None:
        """Execute the callback for a completed pipeline if one exists.
        
        Args:
            pipeline_id: The UUID of the pipeline.
            status: The exit status of the pipeline.
        """
        if pipeline_id in self.callbacks:
            try:
                callback = self.callbacks[pipeline_id]
                callback(pipeline_id, status)
            except Exception as e:
                print(f"Error executing callback for pipeline {pipeline_id}: {e}")
            finally:
                # Remove the callback regardless of whether it succeeded
                del self.callbacks[pipeline_id]
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with the given PID is running.
        
        Args:
            pid: Process ID to check.
            
        Returns:
            True if the process is running, False otherwise.
        """
        try:
            # Using psutil for more reliable cross-platform process checking
            process = psutil.Process(pid)
            return process.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False
    
    def _cleanup_pipeline(self, pipeline_id: str) -> None:
        pass

    def start_pipeline(self, pipeline: List[Tuple[str, str, List[str]]], callback: Optional[Callable[[str, int], None]] = None) -> str:
        """Start a pipeline of commands as a detached background process.
        
        Args:
            pipeline: A list of command tuples (cwd, cmd, args).
            callback: An optional callback function to call when the pipeline completes.
                      The callback will receive the pipeline_id and status code.
            
        Returns:
            The UUID of the started pipeline.
        """
        # Generate a unique ID for this pipeline
        pipeline_id = str(uuid.uuid4())
        
        # Create log file paths
        stdout_path = os.path.join(self.log_folder, f"pipeline_{pipeline_id}_stdout.log")
        stderr_path = os.path.join(self.log_folder, f"pipeline_{pipeline_id}_stderr.log")
        status_path = os.path.join(self.log_folder, f"pipeline_{pipeline_id}.status")
        
        # Construct the shell command with proper escaping
        commands = []
        for cwd, cmd, args in pipeline:
            cmd_str = f"cd {shlex.quote(cwd)} && {cmd} {' '.join(shlex.quote(arg) for arg in args)}"
            commands.append(cmd_str)
        
        shell_command = " && ".join(commands)
        
        # Make sure the log directory exists
        os.makedirs(self.log_folder, exist_ok=True)
        
        # Wrap the shell command to track the exit status
        wrapped_command = (
            f"{{ {shell_command}; "
            f"PIPELINE_STATUS=$?; "
            f"echo $PIPELINE_STATUS > {shlex.quote(status_path)}; "
            f"exit $PIPELINE_STATUS; }}"
        )
        
        # Create a temporary PID file to retrieve the PID of the detached process
        pid_file = f"/tmp/pipeline_{pipeline_id}.pid"
        
        # Construct the full command, ensuring total detachment
        full_command = (
            f"nohup bash -c {shlex.quote(wrapped_command)} > {stdout_path} 2> {stderr_path} "
            f"</dev/null & echo $! > {pid_file}"
        )
        
        # Execute the command to start the detached process
        subprocess.run(full_command, shell=True)
        
        # Wait briefly for the PID file to be created
        time.sleep(0.1)
        
        # Read the PID from the PID file
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.remove(pid_file)  # Clean up PID file
        except (IOError, ValueError, OSError):
            pid = None
        
        # Store pipeline information
        self.pipelines[pipeline_id] = {
            'pipeline_id': pipeline_id,
            'pid': pid,
            'stdout': stdout_path,
            'stderr': stderr_path,
            'status_file': status_path,
            'pipeline': pipeline,
            'start_time': time.time()
        }
        
        # Register callback if provided
        if callback:
            self.callbacks[pipeline_id] = callback
        
        # Save updated state
        self._save_state()
        
        return pipeline_id
    
    def list_running_tasks(self) -> List[Dict[str, Any]]:
        """List all currently running pipeline tasks and process any that have terminated.
        
        Returns:
            A list of dictionaries containing information about running pipelines.
        """
        running_tasks = []
        terminated_pipelines = []
        
        for pipeline_id, pipeline_info in list(self.pipelines.items()):
            pid = pipeline_info.get('pid')
            running_tasks.append(pipeline_info)
        
        return running_tasks
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[int]:
        """Get the status of a pipeline.
        
        Args:
            pipeline_id: The UUID of the pipeline.
            
        Returns:
            0 for success, negative value for error, None if pipeline is not found.
            If the pipeline is still running, returns None.
        """
        if pipeline_id not in self.pipelines:
            return None
        
        pid = self.pipelines[pipeline_id].get('pid')
        
        # First check if process is still running
        if pid is not None and self._is_process_running(pid):
            return None  # Pipeline is still running
        
        # The process is not running, so try to get the status
        status = self._get_pipeline_status(pipeline_id)
        
        # If pipeline has terminated, process its callback and clean it up
        self._execute_callback(pipeline_id, status)
        self._cleanup_pipeline(pipeline_id)
            
        return status
    
    def get_output(self, pipeline_id: str, tail_lines: int = 10) -> Optional[Dict[str, Any]]:
        """Get the recent output from a pipeline's log files.
        
        Args:
            pipeline_id: The UUID of the pipeline.
            tail_lines: Number of lines to return from the end of each log file.
            
        Returns:
            A dictionary with stdout and stderr content, or None if pipeline not found.
        """
        # First check if the pipeline exists in our tracking
        if pipeline_id not in self.pipelines:
            # Try to access logs directly even if pipeline is no longer tracked
            stdout_path = os.path.join(self.log_folder, f"pipeline_{pipeline_id}_stdout.log")
            stderr_path = os.path.join(self.log_folder, f"pipeline_{pipeline_id}_stderr.log")
            status_path = os.path.join(self.log_folder, f"pipeline_{pipeline_id}.status")
            
            result = {
                'stdout': None,
                'stderr': None,
                'status': None
            }
            
            # Check if logs still exist and retrieve them
            files_exist = False
            if os.path.exists(stdout_path):
                files_exist = True
                try:
                    with open(stdout_path, 'r') as f:
                        lines = f.readlines()
                        result['stdout'] = ''.join(lines[-tail_lines:]) if lines else ""
                except IOError:
                    result['stdout'] = f"Error reading from {stdout_path}."
            
            if os.path.exists(stderr_path):
                files_exist = True
                try:
                    with open(stderr_path, 'r') as f:
                        lines = f.readlines()
                        result['stderr'] = ''.join(lines[-tail_lines:]) if lines else ""
                except IOError:
                    result['stderr'] = f"Error reading from {stderr_path}."
            
            if os.path.exists(status_path):
                files_exist = True
                try:
                    with open(status_path, 'r') as f:
                        status = f.read().strip()
                        result['status'] = int(status) if status else -1
                except (IOError, ValueError):
                    result['status'] = -1
            
            return result if files_exist else None
        
        # Pipeline is still tracked, get info and check status
        pipeline_info = self.pipelines[pipeline_id]
        stdout_path = pipeline_info.get('stdout')
        stderr_path = pipeline_info.get('stderr')
        
        result = {
            'stdout': None,
            'stderr': None,
            'status': None
        }
        
        # Get stdout if file exists
        if stdout_path and os.path.exists(stdout_path):
            try:
                with open(stdout_path, 'r') as f:
                    lines = f.readlines()
                    result['stdout'] = ''.join(lines[-tail_lines:]) if lines else ""
            except IOError:
                result['stdout'] = f"Error reading from {stdout_path}."
        
        # Get stderr if file exists
        if stderr_path and os.path.exists(stderr_path):
            try:
                with open(stderr_path, 'r') as f:
                    lines = f.readlines()
                    result['stderr'] = ''.join(lines[-tail_lines:]) if lines else ""
            except IOError:
                result['stderr'] = f"Error reading from {stderr_path}."
        
        # Check if the pipeline is still running
        pid = pipeline_info.get('pid')
        if pid is not None and not self._is_process_running(pid):
            # Pipeline has terminated, get its status
            result['status'] = self._get_pipeline_status(pipeline_id)
            
            # Process its callback and clean it up
            self._execute_callback(pipeline_id, result['status'])
            self._cleanup_pipeline(pipeline_id)
        
        return result
    
    def stop_pipeline(self, pipeline_id: str) -> bool:
        """Stop a running pipeline.
        
        Args:
            pipeline_id: The UUID of the pipeline to stop.
            
        Returns:
            True if the pipeline was stopped successfully, False otherwise.
        """
        if pipeline_id not in self.pipelines:
            return False
        
        pid = self.pipelines[pipeline_id].get('pid')
        if pid is None:
            # Already terminated
            status = self._get_pipeline_status(pipeline_id)
            self._execute_callback(pipeline_id, status)
            self._cleanup_pipeline(pipeline_id)
            return True
        
        # First try SIGTERM for graceful termination
        try:
            process = psutil.Process(pid)
            # Kill the entire process tree to ensure all child processes are terminated
            for child in process.children(recursive=True):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            process.terminate()
            
            # Wait up to 5 seconds for the process to terminate
            process.wait(timeout=5)
            
            # If process is still running, use SIGKILL
            if process.is_running():
                for child in process.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
            
            # Write a termination status to the status file
            status_path = os.path.join(self.log_folder, f"pipeline_{pipeline_id}.status")
            with open(status_path, 'w') as f:
                f.write("-2")  # -2 indicates manual termination
            
            # Execute callback with the termination status
            self._execute_callback(pipeline_id, -2)
                
            # Clean up resources
            self._cleanup_pipeline(pipeline_id)
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process might have already terminated, check its status
            status = self._get_pipeline_status(pipeline_id)
            self._execute_callback(pipeline_id, status)
            self._cleanup_pipeline(pipeline_id)
            return True



def callback_function(pipeline_id, status):
    """Callback function for completed pipelines."""
    status_text = "SUCCESS" if status == 0 else f"ERROR (code: {status})"
    print(f"\nCallback triggered for pipeline {pipeline_id}: {status_text}")

def main():
    # Create a pipeline manager with a custom log folder
    manager = PipelineManager(log_folder="test_pipeline_logs")
    print(f"Pipeline manager initialized with log folder: {manager.log_folder}")
    
    # Test 1: Start a successful pipeline (sleeps and then outputs a message)
    print("\n----- Test 1: Successful Pipeline -----")
    successful_pipeline = [
        ("/tmp", "bash", ["-c", "echo 'Starting successful pipeline'; sleep 3; echo 'Pipeline completed successfully'; exit 0"])
    ]
    
    pipeline_id1 = manager.start_pipeline(successful_pipeline, callback_function)
    print(f"Started successful pipeline with ID: {pipeline_id1}")
    
    # Test 2: Start a failing pipeline (tries to execute a non-existent command)
    print("\n----- Test 2: Failing Pipeline -----")
    failing_pipeline = [
        ("/tmp", "bash", ["-c", "echo 'Starting failing pipeline'; non_existent_command; echo 'This will not be reached'"])
    ]
    
    pipeline_id2 = manager.start_pipeline(failing_pipeline, callback_function)
    print(f"Started failing pipeline with ID: {pipeline_id2}")
    
    # Test 3: Start a long-running pipeline that we'll manually stop
    print("\n----- Test 3: Long-running Pipeline (to be stopped) -----")
    long_pipeline = [
        ("/tmp", "bash", ["-c", "echo 'Starting long pipeline'; for i in {1..30}; do echo \"Working... $i\"; sleep 1; done"])
    ]
    
    pipeline_id3 = manager.start_pipeline(long_pipeline, callback_function)
    print(f"Started long-running pipeline with ID: {pipeline_id3}")
    
    # Monitor all pipelines for 5 seconds
    print("\n----- Monitoring Pipelines -----")
    start_time = time.time()
    while time.time() - start_time < 7:  # Increased to 7 seconds to see the successful pipeline complete
        # List running tasks
        running_tasks = manager.list_running_tasks()
        print(f"\nRunning tasks: {len(running_tasks)}")
        
        # Check status of each pipeline
        for pipeline_id in [pipeline_id1, pipeline_id2, pipeline_id3]:
            # Get the output first (which also checks for termination)
            output = manager.get_output(pipeline_id, tail_lines=3)
            
            # Now get the status (which might be None if still running)
            status = manager.get_pipeline_status(pipeline_id)
            
            status_text = "RUNNING" if status is None else f"FINISHED (status: {status})"
            print(f"Pipeline {pipeline_id}: {status_text}")
            
            if output:
                if output['stdout']:
                    print(f"  Latest stdout: {output['stdout'].strip()}")
                if output['stderr'] and output['stderr'].strip():
                    print(f"  Latest stderr: {output['stderr'].strip()}")
        
        time.sleep(1)
    
    # Stop the long-running pipeline
    print("\n----- Stopping Long-running Pipeline -----")
    if manager.stop_pipeline(pipeline_id3):
        print(f"Successfully stopped pipeline {pipeline_id3}")
    else:
        print(f"Failed to stop pipeline {pipeline_id3}")
    
    # Wait for one more second to ensure all callbacks are processed
    time.sleep(1)
    
    # Check final status of all pipelines - should now access logs directly
    print("\n----- Final Pipeline Statuses -----")
    for pipeline_id in [pipeline_id1, pipeline_id2, pipeline_id3]:
        # Get output which includes status now
        output = manager.get_output(pipeline_id)
        status = None if not output else output.get('status')
        
        if status is None:
            print(f"Pipeline {pipeline_id}: Status unknown (pipeline may have completed without status file)")
        else:
            status_text = "SUCCESS" if status == 0 else f"ERROR (code: {status})"
            print(f"Pipeline {pipeline_id}: {status_text}")
    
    # Check if we can access logs after pipelines have completed
    print("\n----- Accessing Logs After Completion -----")
    for pipeline_id in [pipeline_id1, pipeline_id2, pipeline_id3]:
        log_files = [
            os.path.join(manager.log_folder, f"pipeline_{pipeline_id}_stdout.log"),
            os.path.join(manager.log_folder, f"pipeline_{pipeline_id}_stderr.log"),
            os.path.join(manager.log_folder, f"pipeline_{pipeline_id}.status")
        ]
        
        print(f"Logs for pipeline {pipeline_id}:")
        for log_file in log_files:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    content = f.read().strip()
                    if "stdout" in log_file:
                        print(f"  Stdout: {content[:100]}..." if len(content) > 100 else f"  Stdout: {content}")
                    elif "stderr" in log_file:
                        print(f"  Stderr: {content[:100]}..." if len(content) > 100 else f"  Stderr: {content}")
                    else:
                        print(f"  Status: {content}")
            else:
                print(f"  Log file not found: {log_file}")

if __name__ == "__main__":
    main()