import argparse
import sys
import os

def parse_device_arg(device_args):
    devices = []
    if device_args:
        for arg in device_args:
            try:
                device_id, num_workers = arg.split(':')
                devices.append({
                    'device_id': device_id,
                    'num_workers': int(num_workers)
                })
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Device argument '{arg}' is not in the form <device_id>:<num_workers>"
                )
    return devices


def get_cpu_sockets_linux():
    sockets = {}
    try:
        with open('/proc/cpuinfo') as f:
            cpuinfo = f.read().split('\n\n')
        for entry in cpuinfo:
            lines = entry.strip().split('\n')
            phys_id = None
            core_id = None
            for line in lines:
                if line.startswith('physical id'):
                    phys_id = line.split(':')[1].strip()
                elif line.startswith('core id'):
                    core_id = line.split(':')[1].strip()
            if phys_id is not None:
                if phys_id not in sockets:
                    sockets[phys_id] = {
                        'core_ids': set(),
                        'logical_count': 0
                    }
                if core_id is not None:
                    sockets[phys_id]['core_ids'].add(core_id)
                sockets[phys_id]['logical_count'] += 1

        result = []
        for socket_id, info in sockets.items():
            core_count = len(info['core_ids'])
            logical_count = info['logical_count']
            hyperthreading = logical_count > core_count
            result.append({
                'socket_id': socket_id,
                'core_count': core_count,
                'logical_count': logical_count,
                'hyperthreading': hyperthreading
            })
        return result
    except Exception as e:
        return None
    
def get_gpu_info():
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return "No GPU devices found."
        else:
            gpu_info = []
            for i in range(gpu_count):
                gpu_info.append({
                    'device_id': str(i),
                    'name': torch.cuda.get_device_name(i),
                    'memory': torch.cuda.get_device_properties(i).total_memory
                })
            return gpu_info
    except ImportError:
        return "PyTorch not installed. Cannot list GPU devices."
    except Exception as e:
        return f"Error retrieving GPU information: {str(e)}"
    
def list_devices_command(args):
    cpu_sockets = get_cpu_sockets_linux()
    if cpu_sockets is None:
        print("Error: Unable to retrieve CPU socket information.")
    else:
        for socket in cpu_sockets:
            print(f"CPU {socket['socket_id']}: {socket['core_count']} cores, "
                  f"{socket['logical_count']} logical processors, "
                  f"Hyperthreading: {'Enabled' if socket['hyperthreading'] else 'Disabled'}")
            
    gpu_info = get_gpu_info()
    if isinstance(gpu_info, str):
        print(gpu_info)
    else:
        for gpu in gpu_info:
            print(f"GPU {gpu['device_id']}: {gpu['name']}, "
                  f"Memory: {gpu['memory'] // (1024 ** 2)} MB")
    
def run_command(args):
    cpu_devices = parse_device_arg(args.cpu)
    gpu_devices = parse_device_arg(args.gpu)

    from preprocessors.prep_timeline import preprocess_timeline
    from preprocessors.prep_dataset import preprocess_dataset
    from preprocessors.prep_model import preprocess_model

    dataset_name = os.path.basename(args.dataset_path)

    preprocess_model(args.dataset_path, f".data/{dataset_name}")
    preprocess_timeline(args.timeline_path, f".data/{dataset_name}")
    preprocess_dataset(args.dataset_path, f".data/{dataset_name}")


    # start subprocesses.

    # start worker with devices as arguments
    import subprocess
    
    cmd = ["worker", "0"]

    for device in cpu_devices:
        cmd.append("cpu:" + device['device_id'])
        cmd.append(str(device['num_workers']))

    for device in gpu_devices:
        cmd.append("gpu:" + device['device_id'])
        cmd.append(str(device['num_workers']))





def main():
    parser = argparse.ArgumentParser(description='Simulation CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Run command
    run_parser = subparsers.add_parser('run', help='Run the simulation')
    run_parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    run_parser.add_argument('timeline_path', type=str, help='Path to the timeline')
    run_parser.add_argument('--cpu', type=str, action='append', help='CPU device id')
    run_parser.add_argument('--gpu', type=str, action='append', help='GPU device id')
    run_parser.set_defaults(func=run_command)

    # List-devices command
    list_parser = subparsers.add_parser('list-devices', help='List available devices')
    list_parser.set_defaults(func=list_devices_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()











