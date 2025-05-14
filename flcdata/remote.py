import paramiko
from scp import SCPClient
from sshtunnel import SSHTunnelForwarder

from typing import List, Tuple
import os
import tarfile
import concurrent.futures
import logging
import time
import random
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Error:
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class Frontend:
    def __init__(self, host: str, username: str, pkey: str, nodes: List[Tuple[str, str]]):
        assert len(nodes) > 0, "At least one node must be provided."
        assert len(set([c[0] for c in nodes])) == len(nodes), "Duplicate nodes names found."

        self.host = host
        self.username = username
        self.pkey = pkey
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.nodes = nodes
        self.connected = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    def __create_tarball(self, files: List[str], tarball_name: str):
        with tarfile.open(tarball_name, "w:gz") as tar:
            for file in files:
                # Normalize the path to remove any redundant `..` or `.` components
                normalized_path = os.path.normpath(file)
                # Split the path into components
                path_components = normalized_path.split(os.sep)
                # Remove the first significant folder (ignoring leading ".." or ".")
                
                path_components = [c for c in path_components if c not in [".", ".."]]

                if len(path_components) > 1:
                    path_components = path_components[1:]

                trimmed_path = os.path.join(*path_components)
                # Add the file to the tarball with the trimmed path as the arcname
                tar.add(file, arcname=trimmed_path)

    def __valid_node_name(self, node: str) -> bool:
        return node in [c[0] for c in self.nodes]

    def _node_username(self, node: str) -> str:
        return [c[1] for c in self.nodes if c[0] == node][0]

    def connect(self):
        self.client.connect(self.host, 22, username=self.username, key_filename=self.pkey)
        self.connected = True

    def close(self):
        self.client.close()
        self.executor.shutdown()
        self.connected = False

    def cmd(self, cmd: str) -> Tuple[str, Error]:
        assert self.connected, "Not connected to the frontend host."
        if cmd.startswith("ssh"):
            parts = cmd.split(" ")
            cmd = " ".join(parts[:1] + ["-o StrictHostKeyChecking=no"] + parts[1:])


        try:
            stdin, stdout, stderr = self.client.exec_command(cmd)
            stdin.close()
            exit_status = stdout.channel.recv_exit_status()
            output = stdout.read().decode()
            error = stderr.read().decode()

            if exit_status != 0:
                error = Error(f"Command '{cmd}' failed with exit status {exit_status}: {error}")
            else:
                error = None

            return output, error
        except Exception as e:
            return "", Error(str(e))

    def node_cmd(self, node: str, cmd: str) -> Tuple[str, Error]:
        assert self.__valid_node_name(node), f"node '{node}' not found."
        assert self.connected, "Not connected to the frontend host."

        idx = -1
        for i, c in enumerate(self.nodes):
            if c[0] == node:
                idx = i
                break

        cmd = cmd.replace('@host_idx', str(idx))
        cmd = cmd.replace('@host', node)
        cmd = cmd.replace('$', "\\$")

        username = self._node_username(node)
        

        cmd = f"ssh {username}@{node} \"{cmd}\""

        return self.cmd(cmd)

    def nodes_cmd(self, nodes: List[str], cmd: str) -> Tuple[List[Tuple[str, str, Error]], Error]:
        assert all([self.__valid_node_name(n) for n in nodes]), "Invalid node name found."

        def run_node_cmd(node):
            out, err = self.node_cmd(node, cmd)
            return (node, out, err)

        futures = [self.executor.submit(run_node_cmd, node) for node in nodes]
        results = []
        errored = 0
        errStr = ""
        for future in concurrent.futures.as_completed(futures):
            node, out, err = future.result()
            results.append((node, out, err))
            if err is not None:
                errStr += f"Node {node}: {err}\n"
                errored += 1

        error = Error(f"{errored}/{len(nodes)} nodes failed to execute command: {cmd}\n{errStr}") if errored > 0 else None
        return results, error





    def dowload(self, node: str, remote_files: List[str], local_folder: str, keep_archive: bool = False) -> Error:
        assert self.connected, "Not connected to the frontend host."
        assert self.__valid_node_name(node), f"node '{node}' not found."

        # Create a remote folder with all the files, a file can be a folder
        random.seed(time.time())
        randID = random.randint(0, 1000000)
        tmp_remote = f"tmp/{randID}"
        _, err = self.cmd(f"mkdir -p {tmp_remote}")
        if err is not None: 
            return err
        
        logger.info(f"Temporary folder {tmp_remote} created on the front-end machine.")
        for file in remote_files:
            username = self._node_username(node)
            # _, err = self.cmd(f"scp -r {username}@{node}:{file} {tmp_remote}")  
            # use rsync instead of scp to copy the files
            _, err = self.cmd(f"rsync  -avz -e 'ssh -o StrictHostKeyChecking=no' {username}@{node}:{file} {tmp_remote + '/'}")

            if err is not None:
                err1 = err
                _, err = self.cmd(f"rm -rf {tmp_remote}")
                if err is not None:
                    logger.warning(f"Failed to remove temporary folder {tmp_remote}")
                return err1
            
        out, err = self.cmd(f"ls {tmp_remote}")
        if err is not None:
            return err

        print(out)

        _, err = self.cmd(f"cd tmp && tar -czf {randID}.tar.gz {randID}")
        if err is not None:
            err1 = err
            _, err = self.cmd(f"rm -rf {tmp_remote}")
            if err is not None:
                logger.warning(f"Failed to remove temporary folder {tmp_remote}")   
            return err1     
        
        tarball = f"tmp/{randID}.tar.gz"
        logger.info(f"Tarball {tarball} created on the front-end machine.")

        if os.path.exists(local_folder):
            if not os.path.isdir(local_folder):
                return Error(f"Path {local_folder} is not a directory.")
        else:
            os.makedirs(local_folder)    
        

        with SCPClient(self.client.get_transport()) as scp:
            scp.get(tarball, local_folder)

        logger.info(f"Tarball {tarball} downloaded to {local_folder}.")

        _, err = self.cmd(f"rm -rf {tmp_remote}; rm -f {tarball}")
        if err is not None:
            logger.warning(f"Failed to remove temporary folder {tmp_remote} and tarball {tarball}")

        try:
            os.system(f"tar -xzf {local_folder}/{randID}.tar.gz -C {local_folder} && rm -f {local_folder}/{randID}.tar.gz && mv {local_folder}/{randID}/* {local_folder} && rm -rf {local_folder}/{randID}")
        except Exception as e:
            return Error(str(e))
        
        return None


    def upload(self, files: List[str]) -> Tuple[str, Error]:
        assert self.connected, "Not connected to the frontend host."
        tb_name = f"pl_{random.randint(0, 1000000)}.tar.gz"
        local_tb = f"/tmp/{tb_name}"
        remote_tb = f"tmp/{tb_name}"
        self.__create_tarball(files, local_tb)

        try:
            _, err = self.cmd("mkdir -p tmp")
            if err is not None:
                raise Exception(err)

            transport = self.client.get_transport()
            with SCPClient(transport) as scp:
                scp.put(local_tb, remote_tb)
        except Exception as e:
            os.remove(local_tb)
            return "", Error(str(e))

        logger.info(f"Tarball {tb_name} uploaded to front-end machine.")
        return remote_tb, None

    def nodes_upload(self, nodes: List[str], files: List[str], folder=None | str) -> Error:
        assert all([self.__valid_node_name(n) for n in nodes]), "Invalid node name found."

        tarball, err = self.upload(files)
        if err is not None:
            return err

        def upload_to_node(node):
            username = self._node_username(node)
            cmd = f"mkdir -p tmp && rm -f {tarball}"
            _, err = self.node_cmd(node, cmd)
            if err is not None:
                return node, err

            cmd = f"scp {tarball} {username}@{node}:{tarball}"
            _, err = self.cmd(cmd)
            if err is not None:
                return node, err
            
            return node, None

        futures = [self.executor.submit(upload_to_node, node) for node in nodes]
        results = []

        for future in concurrent.futures.as_completed(futures):
            node, err = future.result()
            results.append((node, err))

        failed_nodes = [node for node, err in results if err is not None]
        if failed_nodes:
            # Revert the successful uploads
            for node, err in results:
                if err is None:
                    username = self._node_username(node)
                    self.node_cmd(node, f"rm -f {tarball}")

            return Error(f"Failed to upload tarball to nodes: {', '.join(failed_nodes)}")

        logger.info(f"Tarball {tarball} uploaded to {len(nodes)} nodes.")
        self.cmd(f"rm -f {tarball}")
        if folder is not None:
            def extract_tarball(node):
                cmd = f"mkdir -p {folder} && tar -xzf {tarball} -C {folder} && rm -f {tarball}"
                _, err = self.node_cmd(node, cmd)
                return node, err

            futures = [self.executor.submit(extract_tarball, node) for node in nodes]
            for future in concurrent.futures.as_completed(futures):
                node, err = future.result()
                if err is not None:
                    return err

        logger.info(f"Tarball extracted in {folder} on {len(nodes)} nodes.")
        return None

    def node_bg_cmd(self, node: str, cmd: str, out_file: str = None) -> Tuple[int, Error]:
        out_file = '/dev/null' if out_file is None else out_file

        # cmd = f"bash -c '({cmd}) > {out_file} 2>&1 & echo $!' 2>&1 |" + "{ read pid; echo \"$pid\"; }"
        cmd = f"({cmd}) > {out_file} 2>&1 & echo \$!"
        output, err = self.node_cmd(node, cmd)

        if err is not None:
            return -1, err

        stripped = output.strip()

        if not stripped.isdigit():
            return -1, Error(f"Failed to get PID of detached command {cmd} {output}")
        
        return int(stripped), None

    def nodes_bg_cmd(self, nodes: List[str], cmd: str, out_file: str = None) -> Tuple[List[Tuple[str, int, Error]], Error]:
        assert all([self.__valid_node_name(n) for n in nodes]), "Invalid node name found."

        def run_node_bg_cmd(node):
            pid, err = self.node_bg_cmd(node, cmd, out_file)
            return (node, pid, err)

        futures = [self.executor.submit(run_node_bg_cmd, node) for node in nodes]
        results = []

        for future in concurrent.futures.as_completed(futures):
            node, pid, err = future.result()
            results.append((node, pid, err))

        errored = [node for node, pid, err in results if err is not None]
        if len(errored) > 0:
            return results, Error(f"{len(errored)}/{len(nodes)} nodes failed to execute command: {cmd}")

        return results, None

    def node_pid_status(self, node: str, pid: int) -> Tuple[Tuple[str, str, str, str] | None, Error]:
        cmd = f"ps -p {pid} -o pid,ppid,pgid,stat,cmd"
        out, err =  self.node_cmd(node, cmd)
        if err is not None:
            return None, err

        lines = out.split("\n")
        if len(lines) < 2:
            return None, Error(f"Failed to get process status for PID {pid} on node {node}")
        
        line = lines[1]
        _pid, ppid, pgid, stat, cmd = line.split(" ", 4)
        assert _pid == str(pid), f"Invalid PID found: {pid} != {_pid}"
        return (ppid, pgid, stat, cmd), None



front_end_host = "discovery-head.di.unipi.it"
front_end_username = 'morozzi'
front_end_key = '/home/mamo/.ssh/fd_unipi'

machines = [
    ["n01.maas", "ubuntu"],
    ["n02.maas", "ubuntu"],
    ["n03.maas", "ubuntu"],
    ["n04.maas", "ubuntu"],
    ["n05.maas", "ubuntu"],
    ["n06.maas", "ubuntu"],
    ["n07.maas", "ubuntu"],
    ["n08.maas", "ubuntu"],
    ["n09.maas", "ubuntu"],
    ["n10.maas", "ubuntu"],
]

# files = ["ticker.py", "sim_slave2.py", "dataset", "requirements.txt", "sim_master.py"]

def init_env(folder: str, files: List[str], dest: str, skip_install, nodes: List[str]):
    client = Frontend(front_end_host, front_end_username, front_end_key, machines)
    client.connect()

    logger.info("Creating virtual environment on nodes.")


    _, err = client.nodes_cmd(nodes, f"if [ ! -x {dest}/venv/bin/activate ]; then mkdir -p {dest} && rm -rf {dest}/venv && python3 -m venv {dest}/venv && mkdir -p {dest}/src; fi")
    if err is not None:
        logger.error(f"Failed to create virtual environment, let's try to install python3-venv")
        # try install python3-venv
        _, err = client.nodes_cmd(nodes, f"sudo apt-get install -y python3-venv")
        if err is not None:
            logger.error(f"Failed to create virtual environment: {err}")
            exit(1)

        _, err = client.nodes_cmd(nodes, f"if [ ! -x {dest}/venv/bin/activate ]; then mkdir -p {dest} && rm -rf {dest}/venv && python3 -m venv {dest}/venv && mkdir -p {dest}/src; fi")
        if err is not None:
            logger.error(f"Failed to create directory: {err}")
            exit(1)


    logger.info("Uploading files to nodes.")
    err = client.nodes_upload(nodes, files, folder=f"{dest}/src")
    if err is not None:
        logger.error(f"Failed to upload files: {err}")
        exit(1)

    if skip_install:
        client.close()
        return

    logger.info("Installing requirements on nodes.")
    _, err = client.nodes_cmd(nodes, f"cd {dest} && source venv/bin/activate && pip install -r src/requirements.txt --no-deps --ignore-installed")
    if err is not None:
        logger.error(f"Failed to install requirements: {err}")
        exit(1)

    client.close()

def run(nodes: List[str]):
    client = Frontend(front_end_host, front_end_username, front_end_key, machines)
    client.connect()

    outs, err = client.nodes_cmd(nodes, args.cmd)
    if err is not None:
        logger.error(f"Failed to execute command: {err}")
        exit(1)

    for node, out, _ in outs:
        print("=== Node:", node, "===")
        print(out)

    client.close()

def runprog(nodes: List[str], out_file: str | None, interactive: bool = False):
    client = Frontend(front_end_host, front_end_username, front_end_key, machines)
    client.connect()

    outs, err = client.nodes_bg_cmd(nodes, args.cmd, out_file)
    for node, pid, err in outs:
        if err is not None:
            logger.error(f"Failed to start program on node {node}: {err}")
            continue

        print(f"Node {node}: Program started with PID {pid}")


    latest_cmd = None
    latest_tail_lines = 10

    # ask for kill str to stop the programs
    while interactive:
        try:

            cached_cmd_used = False
            i = input("Enter 'kill', 'exit', 'status', 'tail': ")
            
            if i == "": 
                cached_cmd_used = True
                i = latest_cmd
            else:
                latest_cmd = i

            if i == "exit":
                break

            if i == "kill":
                valid = [(n, pid) for n, pid, err in outs if pid != -1 and err is None]
                print("Killing the following processes:" , [f"{n}: {pid}" for n, pid in valid])
                for n, pid in valid:
                    _, err = client.node_cmd(n, f"sudo kill -9 {pid}")
                    if err is not None:
                        logger.warning(f"Failed to kill process {pid} on node {n}: {err}")
                break

            if i == "status":
                for n, pid, err in outs:
                    if pid == -1:
                        print(f"Node {n}: Failed to start program.")
                    else:
                        status, err = client.node_pid_status(n, pid)
                        if err is not None:
                            print(f"Node {n}: Failed to get status for PID {pid}")
                        else:
                            ppid, pgid, stat, _ = status
                            print(f"Node {n}: PID: {pid}, PPID: {ppid}, PGID: {pgid}, STAT: {stat}")

            if i == "tail":
                if out_file is None:
                    print("Output file not provided.")
                    continue

                nLines = latest_tail_lines if cached_cmd_used else 10
                while not cached_cmd_used:
                    li = input(f"Enter number of lines to tail (default 10): ")
                    if li == "":
                        break

                    try:
                        nLines = int(li)
                        break
                    except:
                        print("Invalid number.")

                if not cached_cmd_used:
                    latest_tail_lines = nLines
                
                for n, pid, err in outs:
                    out, err = client.node_cmd(n, f"tail -n {nLines} {out_file}")
                    str = f"[Node {n}]"

                    if out.endswith("\n"):
                        out = out[:-1]
    
                    if err is not None:
                        str += f"Failed to tail file: {err}"
                    else:
                        if nLines > 1:
                            str += "\n"

                        str += out
                    
                        if nLines > 1:
                            str += "\n"
                    
                    print(str)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)

    client.close()  

def whatch_file(node: str, file: str):
    client = Frontend(front_end_host, front_end_username, front_end_key, machines)
    client.connect()

    stdin, stdout, stderr  = client.client.exec_command(f"ssh {client._node_username(node)}@{node} tail -f {file}")
    for line in stdout:
        print(line.strip())

    stdout.close()
    stdin.close()
    stderr.close()
    client.close()



def parse_node_arg(node: str, excluded: str | None):
    excluded = [] if excluded is None else excluded.split(",")
    nodes = [m[0] for m in machines] if node == "all" else node.split(",")

    def node_short_map(node: str):
        if node.isdigit():
            idx = int(node) - 1
            if idx < 0 or idx >= len(machines):
                print("Invalid node index found.")
                exit(1)
            
            return machines[idx][0]
        return node
    
    nodes = [node_short_map(n.strip()) for n in nodes]
    nodes = [n for n in nodes if n not in excluded]

    if not all([n in [m[0] for m in machines] for n in nodes]):
        print("Invalid node name found.")
        exit(1)

    if len(nodes) == 0:
        print("No valid nodes found.")
        exit(1)

    return nodes



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI tool")
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    init_parser = subparsers.add_parser("init", help="Setup the environment on the nodes.")
    init_parser.add_argument("src", help="Path to folder containing the source files.")
    init_parser.add_argument("dest", help="Path to folder where the source files will be copied to.")
    init_parser.add_argument("nodes", type=str, help="Nodes to run the program on.")
    init_parser.add_argument("--exclude", type=str, help="Nodes to exclude from running the program.")
    init_parser.add_argument("--skip-install", action="store_true", help="Skip installing the requirements.")

    tunnel_parser = subparsers.add_parser("tunnel", help="Create a tunnel to the frontend host.")
    tunnel_parser.add_argument("port", type=int, help="Port to create the tunnel on.")
    tunnel_parser.add_argument("node", type=str, help="Nodes to create the tunnel on.")

    dowload_parser = subparsers.add_parser("dowload", help="Dowload files from the nodes.")
    # dowload_parser.add_argument("files", type=str, help="Files to dowload from the nodes.")
    # dowload_parser.add_argument("dest", type=str, help="Destination folder to save the files.")


    upload_parser = subparsers.add_parser("upload", help="Upload files to the nodes.")
    upload_parser.add_argument("folder", type=str, help="Folder containing the files to upload.")
    upload_parser.add_argument("dest", type=str, help="Destination folder on the nodes.")
    upload_parser.add_argument("nodes", type=str, help="Nodes to upload the files to.")
    upload_parser.add_argument("--exclude", type=str, help="Nodes to exclude from uploading the files.")

    run_parser = subparsers.add_parser("run", help="Run cmd")
    run_parser.add_argument("cmd", type=str, help="Command to run on the nodes.")
    run_parser.add_argument("nodes", type=str, help="Nodes to run the command on.")
    run_parser.add_argument("--exclude", type=str, help="Nodes to exclude from running the command.")


    runprog_parser = subparsers.add_parser("runprog", help="Run a program")
    runprog_parser.add_argument("cmd", type=str, help="Program to run on the nodes.")
    runprog_parser.add_argument("nodes", type=str, help="Nodes to run the program on.")
    runprog_parser.add_argument("--exclude", type=str, help="Nodes to exclude from running the program.")
    runprog_parser.add_argument("--file", type=str, help="Output file for the program.")

    watch_file = subparsers.add_parser("watch", help="Watch a file")
    watch_file.add_argument("file", type=str, help="File to watch.")
    watch_file.add_argument("node", type=str, help="Node to watch the file on.")
        
    args = parser.parse_args()

    if args.command == "init":
        nodes = parse_node_arg(args.nodes, args.exclude)

        files = []

        folder = args.src
        dest = args.dest
        if not os.path.isdir(folder):
            files = folder.split(",")
            if not all([os.path.isfile(f) for f in files]):
                print("Invalid source files.")
                exit(1)
        else:
            for root, _, fs in os.walk(folder):
                for f in fs:
                    path = os.path.join(root, f)
                    if os.path.isfile(path):
                        # path = path[len(folder) + 1:]
                        files.append(path)
        try:
            path = Path(dest)
        except:
            print("Invalid destination folder.")
            exit(1)
        
        # get all files in the folder recursively
      
        # if folder[-1] == "/":
        #     folder = folder[:-1]

        skip_install = args.skip_install

        init_env(None, files, dest, skip_install, nodes)

    elif args.command == "run":
        nodes = parse_node_arg(args.nodes, args.exclude)
        run(nodes)

    elif args.command == "tunnel":  
        node = args.node
        port = args.port
        if not any([node == m[0] for m in machines]):
            print("Invalid node name found.")
            exit(1)

        # os.system(f"ssh -L {port}:localhost:{port} {front_end_username}@{node}")
        with SSHTunnelForwarder(
                (front_end_host, 22),
                ssh_username=front_end_username,
                # ssh_password=front_end_password,
                ssh_pkey=front_end_key,
                remote_bind_address=(node, port),
                local_bind_address=('localhost', port)
            ) as tunnel:
        
            print(f"Tunnel created! Access the backend at localhost:{port}")
            input("Press ENTER to close the tunnel...")

    elif args.command == "dowload":
        client = Frontend(front_end_host, front_end_username, front_end_key, machines)
        client.connect()

        files=["fedml/5_5/*.csv", "fedml/1_1/*.csv", "fedml/iid/*.csv", "fedml/0_0/*.csv"]
        nodes=["n01.maas", "n02.maas", "n03.maas", "n04.maas"]

        for node in nodes:
            for file in files:
                err = client.dowload(node, [file], f"../client/{file.split('/')[1]}", keep_archive=True)
                if err is not None:
                    print("Failed to download files:", err)
                    # exit(1)
                print("Downloaded files from node", node, "files:", file)
                print()

        client.close()

    elif args.command == "upload":
        nodes = parse_node_arg(args.nodes, args.exclude)

        folder = args.folder
        dest = args.dest

        files = []

        if not os.path.isdir(folder):
            files = folder.split(",")
            if not all([os.path.isfile(f) for f in files]):
                print("Invalid source files.")
                exit(1)
        else:
            for root, _, fs in os.walk(folder):
                for f in fs:
                    path = os.path.join(root, f)
                    if os.path.isfile(path):
                        # path = path[len(folder) + 1:]
                        files.append(path)

        try:
            path = Path(dest)
        except:
            print("Invalid destination folder.")
            exit(1)

        

        client = Frontend(front_end_host, front_end_username, front_end_key, machines)
        client.connect()

        err = client.nodes_upload(nodes, files, folder=dest)
        if err is not None:
            print("Failed to upload files:", err)
            exit(1)


        client.close()

    elif args.command == "runprog":
        nodes = parse_node_arg(args.nodes, args.exclude)
        runprog(nodes, args.file)

    elif args.command == "watch":
        whatch_file(args.node, args.file)

    else:
        print("Invalid command. use --help to see the available commands.")
        exit(1)