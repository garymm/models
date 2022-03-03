import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, List
import yaml
import copy
import warnings
#temporary for now
warnings.filterwarnings(action="once")
warnings.simplefilter("ignore")

@dataclass
class ConfigOptimizer():
    """
    Handles the parsing of relevant parameters for hyperbones optimization
    """
    projectname: str #names on app names defined explicitely in each mech file
    projectpath:str #the exact path of each project file
    variable_to_optimize:str
    num_epochs: int
    num_runs:int
    num_trials:int
    num_parallel:int
    minimize:bool
    go_args:str = "" #args to optionally pass into go
    wandb_key:str = "" #if you decide to do wandb logging, specify an id
    use_onlinelogging:bool = field(init = False)

    def __post_init__(self):
        assert self.num_epochs > 0
        assert self.num_runs > 0
        assert self.num_trials > 0
        assert self.num_parallel > 0 #should also check if this goes beyond available cpuus
        assert os.path.exists(self.projectpath)
        #assert os.path.exists(os.path.join(self.projectpath,self.projectname))

        if len(self.wandb_key) > 0:
            self.use_onlinelogging = True
        else:
            self.use_onlinelogging = False

def keep_relevant_keys(parsed_yaml:Dict[Any,Any]) ->Dict[Any,Any]:
    copy_yaml = copy.deepcopy(parsed_yaml)
    for key in parsed_yaml:
        if (key in ConfigOptimizer.__dataclass_fields__)==False:
            del copy_yaml[key]

            warnings.warn("at least one additional key in yamlfile does not exist in ConfigOptimizer, "
                          "this may or may not be intentional, missing element called: {}".format(key),UserWarning)
    return copy_yaml
def dict_to_configobj(parsed_yaml:Dict[Any,Any])->ConfigOptimizer:
    cleaned_yaml = keep_relevant_keys(parsed_yaml)
    return ConfigOptimizer(**cleaned_yaml)

def file_to_configobj(path:str)->ConfigOptimizer:
    with open(path) as yaml_file:
        yaml_string = yaml_file.read()
        yaml_file = yaml.safe_load(yaml_string)
        return dict_to_configobj(yaml_file)

def test_crash():
    configobj = file_to_configobj("configs/bone_config.yaml")
