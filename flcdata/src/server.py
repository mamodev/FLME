from flask import Flask, jsonify, request
from flask_cors import CORS

from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap

from typing import List, Dict, Any
import os
import importlib
import numpy as np


app = Flask(__name__)
CORS(app, origins='*')  # Replace with your frontend's origin(s)



def load_modules_from_directory(directory_name):
    """
    Loads modules from a specified directory and extracts the 'generator'
    variable (of type Module). Returns a list of Module objects represented
    as dictionaries.
    """
    modules_data = []

    directory_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "backend",
        directory_name
    )


    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_name} not found.")
        return []

    for filename in os.listdir(directory_path):
        if filename.endswith('.py'):
            module_name = filename[:-3]
            try:
                module = importlib.import_module(
                    f'backend.{directory_name}.{module_name}'
                )


                generator_module = getattr(module, 'generator', None)

                if generator_module:
                    modules_data.append(generator_module.to_dict())
                else:
                    print(
                        f"Warning: Module {module_name} in {directory_name} "
                        "has no 'generator' variable."
                    )

            except ImportError as e:
                print(
                    f"Error importing module {module_name} from "
                    f"{directory_name}: {e}"
                )
            except Exception as e:
                print(
                    f"An unexpected error occurred while processing module "
                    f"{module_name} from {directory_name}: {e}"
                )
    return modules_data

def load_module_from_directory(directory_name, module_name):
    """
    Loads a specific module from a specified directory and extracts the 'generator'
    variable (of type Module). Returns the Module object represented as a dictionary.
    """
    directory_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "backend",
        directory_name
    )

    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_name} not found.")
        return {}

    module = importlib.import_module(
        f'backend.{directory_name}.{module_name}'
    )
    generator_module = getattr(module, 'generator', None)

    if generator_module:
        return generator_module
    else:
        raise ImportError("'generator' variable not found in module")

@app.route('/api/config', methods=['GET'])
def get_config():
    """
    Endpoint to retrieve the entire configuration based on modules
    found in the specified directories.
    """
    config = {
        'data_generators': load_modules_from_directory('data_generators'),
        'distributions': load_modules_from_directory('distributions'),
        'partitioners': load_modules_from_directory('partitioners'),
        "transformers": load_modules_from_directory('transformers'),
    }
    return jsonify(config)


# ======================================================
#
#  @generator
#  API relted to data generation.....
#
# ======================================================

DATA_FOLDER = ".data"

def to3d(XX):
    if len(XX[0]) == 3:
        return XX

    pca = PCA(n_components=3)
    XX_3d = pca.fit_transform(XX)
    
    # tnse = TSNE(
    #     n_components=3,
    #     perplexity=30,
    #     n_iter=1000,
    #     random_state=42
    # )
    # XX_3d = tnse.fit_transform(XX)

    # mp = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1) # Adjust n_neighbors and min_dist
    # XX_3d = mp.fit_transform(XX_3d)
    
    return XX_3d
    

def __generate(data_generator, distribution, partitioner, transformers):
    dg = load_module_from_directory('data_generators', data_generator["name"])
    dist = load_module_from_directory('distributions', distribution["name"])
    part = load_module_from_directory('partitioners', partitioner["name"])
    trans = [
        (load_module_from_directory('transformers', t["name"]), t["parameters"])
        for t in transformers
    ]

    XX, YY = dg.run(
        params=data_generator["parameters"],
    )

    n_samples = data_generator["parameters"].get("n_samples", len(XX))
    n_classes = data_generator["parameters"].get("n_classes", len(set(YY)))

    assert len(XX) == n_samples, f"n_samples must be equal to the length of XX but got {len(XX)} wich is not equal to {n_samples}"
    assert len(set(YY)) <= n_classes, "n_classes must be greater than or equal to the number of unique labels in YY"

    distr_map = dist.run(
        params=distribution["parameters"],
        n_samples=n_samples,
        n_classes=n_classes,
    )

    PP = part.run(
        params=partitioner["parameters"],
        X=XX,
        Y=YY,
        distr_map=distr_map,
    )

    n_partitions = distribution["parameters"]["n_partitions"]

    for t, p in trans:
        if "active" in p and not p["active"]:
            continue

        XX, YY, PP, distr_map = t.run(
            params=p,
            XX=XX,
            YY=YY,
            PP=PP,
            distr_map=distr_map,
        )

    return XX, YY, PP, n_classes, n_samples, n_partitions

def __downsample(XX, YY, PP, n_classes, n_samples, n_partitions):
    PXX = to3d(XX)
    CTP = np.zeros((n_classes, n_partitions))
    for i in range(len(YY)):
        c = YY[i]
        p = PP[i]
        CTP[c][p] += 1

    if len(XX) > 4000:
        random_indices = np.random.choice(len(XX), 4000, replace=False)
        XX = XX[random_indices]
        YY = YY[random_indices]
        PXX = PXX[random_indices]
        PP = PP[random_indices]

    return PXX, YY, PP, CTP

@app.route('/api/generate', methods=['POST'])
def generate_data():
    XX, YY, PP, n_classes, n_samples, n_partitions = __generate(
        request.json.get('data_generator'),
        request.json.get('distribution'),
        request.json.get('partitioner'),
        request.json.get('transformers', [])
    )

    PXX, YY, PP, CTP = __downsample(
        XX, YY, PP, n_classes, n_samples, n_partitions
    )
    
    return jsonify({
        'X': PXX.tolist(),
        'Y': YY.tolist(),
        'PP': PP.tolist(),
        'CTP': CTP.tolist(),
        'n_classes': n_classes,
        'n_samples': n_samples,
        "n_partitions": n_partitions,
    })

@app.route('/api/save', methods=['POST'])
def save_data():
    filename = request.json.get('file_name')
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    XX, YY, PP, n_classes, n_samples, n_partitions = __generate(
        request.json.get('data_generator'),
        request.json.get('distribution'),
        request.json.get('partitioner'),
        request.json.get('transformers', [])
    )

    save = {
        "XX": XX,
        "YY": YY,
        "PP": PP,
        "n_classes": n_classes,
        "n_samples": n_samples,
        "n_partitions": n_partitions,
        "generation_params": request.json
    }

    save_path = os.path.join(DATA_FOLDER, filename)
    np.savez_compressed(save_path, **save)
    return jsonify({'status': 'success', 'filename': filename})


@app.route('/api/saved-files', methods=['GET'])
def get_saved_files():
    """
    Endpoint to retrieve a list of saved files in the specified directory.
    """
    files = os.listdir(DATA_FOLDER)
    files = [f for f in files if f.endswith('.npz')]

    # load generation parameters from the files
    for i, file in enumerate(files):
        file_path = os.path.join(DATA_FOLDER, file)
        data = np.load(file_path, allow_pickle=True)
        
        if 'generation_params' in data:
            generation_params = data['generation_params'].item()
        else:
            generation_params = {}
            
        n_samples = data['n_samples'].item()
        n_classes = data['n_classes'].item()
        n_partitions = data['n_partitions'].item()
        n_features = data['XX'].shape[1]

        files[i] = {
            "name": file,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "n_partitions": n_partitions,
            "n_features": n_features,
            "generation_params": generation_params
        }

    return jsonify(files)

@app.route('/api/data/<filename>', methods=['GET'])
def get_data(filename):
    file_path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    data = np.load(file_path, allow_pickle=True)

    XX = data['XX']
    YY = data['YY']
    PP = data['PP']
    n_samples = data['n_samples'].item()
    n_classes = data['n_classes'].item()
    n_partitions = data['n_partitions'].item()
   
    PXX, YY, PP, CTP = __downsample(
        XX, YY, PP, n_classes, n_samples, n_partitions
    )
    
    return jsonify({
        'X': PXX.tolist(),
        'Y': YY.tolist(),
        'PP': PP.tolist(),
        "CTP": CTP.tolist(),
        'n_classes': n_classes,
        'n_samples': n_samples,
        "n_partitions": n_partitions,
    })

@app.route('/api/data/<filename>', methods=['DELETE'])
def delete_data(filename):
    file_path = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"status": "success", "message": "File deleted"}), 200
    else:
        return jsonify({"error": "File not found"}), 404


# ======================================================
#
#  @simulation
#  API relted to simulations.....
#
# ======================================================

SIM_FOLDER = ".simulations"

@app.route('/api/list-simulations', methods=['GET'])
def list_simulations():
    """
    Endpoint to retrieve a list of saved simulations in the specified directory.
    """ 

    # list subfolders in the simulation folder
    folders = os.listdir(SIM_FOLDER)
    folders = [f for f in folders if os.path.isdir(os.path.join(SIM_FOLDER, f))]

    # forach folder list files
    simulations = []
    for folder in folders:
        folder_path = os.path.join(SIM_FOLDER, folder)
        files = os.listdir(folder_path)
        # count .model and .metrics files individually
        model_files = [f for f in files if f.endswith('.model')]
        metrics_files = [f for f in files if f.endswith('.metrics')]

        # check info file
        info_file = [f for f in files if f == 'info.json']
        info = None
        if len(info_file) == 1:
            info_file_path = os.path.join(folder_path, 'info.json')
            with open(info_file_path, 'r') as f:
                data = f.read()
                info = json.loads(data)
        

        simulations.append({
            "name": folder,
            "model_files": len(model_files),
            "metrics_files": len(metrics_files),
            "info": info,
        })
    
    return jsonify(simulations)

import json

@app.route('/api/simulation/<simulation_name>', methods=['DELETE'])
def delete_simulation(simulation_name):
    """
    Endpoint to delete a specific simulation folder.
    """
    folder_path = os.path.join(SIM_FOLDER, simulation_name)
    if os.path.exists(folder_path):
        import shutil
        shutil.rmtree(folder_path)
        return jsonify({"status": "success", "message": "Simulation deleted"}), 200
    else:
        return jsonify({"error": "Simulation not found"}), 404

@app.route('/api/simulation-metrics/<simulation_name>', methods=['GET'])
def get_simulation_metrics(simulation_name):
    """
    Endpoint to retrieve metrics for a specific simulation.
    """
    folder_path = os.path.join(SIM_FOLDER, simulation_name)
    if not os.path.exists(folder_path):
        return jsonify({"error": "Simulation not found"}), 404

    files = os.listdir(folder_path)
    metrics_files = [f for f in files if f.endswith('.metrics')]

    metrics = []
    for file in metrics_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            data = f.read()
            jslist = json.loads(data)
            metrics.append(jslist)

    metrics.sort(key=lambda x: x['version'])


    return jsonify(metrics)



from backend.pp_manager import PipelineManager
PIP_TEMPLATES_FOLDER = ".pipelines_templates"
PIP_MANAGER = PipelineManager(
    log_folder = ".pipelines_logs", status_file = ".pipelines_logs/00000_state.json"
)

@app.route('/api/pipelines', methods=['GET'])
def get_pipeline():
    pipelines = PIP_MANAGER.list_pipelines()
    return jsonify(pipelines)

@app.route('/api/pipelines/<pipeline_id>', methods=['DELETE'])
def delete_pipeline(pipeline_id):
    PIP_MANAGER.cleanup_pipeline(pipeline_id)
    return jsonify({'status': 'success', 'pipeline_id': pipeline_id})

@app.route('/api/pipeline/<pipeline_id>/rerun', methods=['POST'])
def rerun_pipeline(pipeline_id):
    try:
        status = PIP_MANAGER.rerun_pipeline(pipeline_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        status = {
            'status': 'error',
            'message': str(e)
        }

        return jsonify(status), 500
    
    return jsonify({'status': status})

@app.route('/api/pipeline/<pipeline_id>/log', methods=['GET'])
def get_pipeline_log(pipeline_id):
    try:
        log = PIP_MANAGER.get_pipeline_log(pipeline_id)
        log[
            'config'
        ] = PIP_MANAGER.get_pipeline_config(pipeline_id)

    except Exception as e:
        import traceback
        traceback.print_exc()
        log = {
            'status': 'error',
            'message': str(e)
        }

        return jsonify(log), 500
    
    return jsonify(log)


@app.route('/api/pipeline/<temp_name>/run', methods=['POST'])
def run_pipeline(temp_name):
    data = request.json

    try:
        temp_name = os.path.join(
            PIP_TEMPLATES_FOLDER,
            f"{temp_name}.json"
        )

        status = PIP_MANAGER.start_from_template(temp_name, data.get('args', None))
    except Exception as e:
        import traceback
        traceback.print_exc()
        status = {
            'status': 'error',
            'message': str(e)
        }

        return jsonify(status), 500
    
    return jsonify({'status': status})

# ep to get a template
@app.route('/api/pipeline/templates', methods=['GET'])
def get_pipeline_templates():
    """
    Endpoint to retrieve a list of pipeline templates in the specified directory.
    """
    # list subfolders in the simulation folder
    files = os.listdir(PIP_TEMPLATES_FOLDER)
    files = [f for f in files if f.endswith('.json')]

    templates = []
    for file in files:
        file_path = os.path.join(PIP_TEMPLATES_FOLDER, file)
        with open(file_path, 'r') as f:
            data = f.read()
            js = json.loads(data)
            js["id"] = file[:-5]
            templates.append(js)

    return jsonify(templates)


# Execute arbitrary code
@app.route('/api/exec-python', methods=['POST'])
def execute_code():
    """
    Endpoint to execute arbitrary code. Expects a JSON payload with 'code'.
    """
    data = request.json
    code = data.get('code')
    GLOBALS = data.get('globals', {})


    def fn():
        GLOBALS["response"] = None
        exec(code, GLOBALS)
        return GLOBALS["response"]

    if not code:
        return jsonify({"error": "No code provided"}), 400
    try:
        res = fn()
        return jsonify({"status": "success", "result": res}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from io import BytesIO
from PIL import Image

@app.route('/api/serve-static/<path:filename>', methods=['GET'])
def serve_static(filename):
    """
    Endpoint to serve static files from the static folder.
    """

    # get query params
    query_params = request.args.to_dict()
    #w=int,h=int
    # With image it enables to resize the image

    h = query_params.get("h", None)
    w = query_params.get("w", None) 

    static_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../"
    )

    file_path = os.path.join(static_folder, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"File {filename} not found in path {static_folder}"}), 404

    with open(file_path, 'rb') as f:
        content = f.read()

    # Set the appropriate content type based on the file extension
    if filename.endswith('.js'):
        content_type = 'application/javascript'
    elif filename.endswith('.css'):
        content_type = 'text/css'
    elif filename.endswith('.html'):
        content_type = 'text/html'
    elif filename.endswith('.json'):
        content_type = 'application/json'
    elif filename.endswith('.png'):
        content_type = 'image/png'
        img = Image.open(BytesIO(content))
        if h and not w:
            w = img.size[0] * (int(h) / img.size[1])
        
        elif w and not h:
            h = img.size[1] * (int(w) / img.size[0])
         
        if h and w:
            img.thumbnail((int(w), int(h)))
        
        img_io = BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        content = img_io.getvalue()

    elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
        content_type = 'image/jpeg'
    else:
        content_type = 'application/octet-stream'
    
    return content, 200, {
        'Content-Type': content_type,
        'Content-Disposition': f'inline; filename={filename}'
    }


if __name__ == '__main__':
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(SIM_FOLDER, exist_ok=True)
    os.makedirs(PIP_TEMPLATES_FOLDER, exist_ok=True)

    app.run(host='0.0.0.0',
        debug=True, threaded=True, port=5555)