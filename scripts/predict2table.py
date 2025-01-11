import glob
from omegaconf import OmegaConf
data_path = "docs/images/20250111"

# Get all the yaml files in the data path
yaml_files = glob.glob(f"{data_path}/**/*.yaml", recursive=True)

metrics = {}

# parse the yaml file path
for yaml_file in yaml_files:
    cfg = OmegaConf.load(yaml_file)
    elems = yaml_file.split("/")
    run_name = elems[3] + "/" + elems[4]
    model_name = ("_").join(run_name.split("_")[:-2])
    testset_name = elems[-1].split(".")[0].split("_", 1)[1]
    
    metrics[run_name] = metrics.get(run_name, {})
    metrics[run_name][testset_name] = OmegaConf.to_container(cfg, resolve=True)

import json
print(json.dumps(metrics, indent=4))
