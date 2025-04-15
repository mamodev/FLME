from src.backend.pp_manager import PipelineManager
import os

PIP_TEMPLATES_FOLDER = ".pipelines_templates"
os.makedirs(PIP_TEMPLATES_FOLDER, exist_ok=True)

pp = PipelineManager(
    log_folder = ".pipelines_logs", status_file = ".pipelines_logs/00000_state.json"
)

# import json

# open file TEMPLATES_FOLDER/pipeline.json

# with open(os.path.join(PIP_TEMPLATES_FOLDER, "pipeline.json"), 'r') as f:
#     template = json.load(f)


# print(f"Running pipeline with template: {template['name']}")

# for param, default in template['parameters'].items():
#     print(f"Param: {param}, Default: {default}")
#     val = input(f"Press Enter to continue with {param} = {default}... or type a new value: ")
#     if val != "":
#         template['parameters'][param] = val

# def replace_args(string, args_dict):
#     for key, value in args_dict.items():
#         string = string.replace(f"@@{key}", str(value))
    
#     return string

# def rec_replace(obj, args_dict = {}):
#     if isinstance(obj, dict):
#         obj2 = {}
#         for key, value in obj.items():
#             obj2[rec_replace(key, args_dict)] = rec_replace(value, args_dict)          

#         return obj2

#     elif isinstance(obj, list):
#         obj2 = []
#         for item in obj:
#             obj2.append(rec_replace(item, args_dict))
        
#         return obj2
    
#     elif isinstance(obj, str):
#        return replace_args(obj, args_dict)
#     else:
#         return obj


# pipeline = rec_replace(template["pipeline"], template["parameters"])
# pipeline = json.dumps(pipeline, indent=4)

# pp.start_pipeline(pipeline)
pp.start_from_template(os.path.join(PIP_TEMPLATES_FOLDER, "pipeline.json"))

import time 
while True:
    time.sleep(1)
    tasks = pp.list_pipelines()
    if len(tasks) == 0:
        break

    # print reset ansii
    print("\033[H\033[J", end="")
    for task in tasks:
        print(f"[{task['id']}] {task['status']} ({task['pid']})")
        
    if all(task["status"] != "RUNNING" for task in tasks):
        break

# for task in tasks:
#     pp.cleanup_pipeline(task["id"])