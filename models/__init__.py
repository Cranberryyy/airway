from .basemodel import BaseModel
from .basemodel_connectivity import BaseModel_Connectivity
from .basemodel_connectivity_distance import BaseModel_Connectivity_Distance
from .basemodel_distance import BaseModel_Distance
from .basemodel_full import BaseModel_Full

from .stagesmodel import StagesModel
from .stagesmodel_connnectivity import StagesModel_Connectivity
from .stagesmodel_connectivity_distance import StagesModel_Connectivity_Distance
from .stagesmodel_distance import StagesModel_Distance
from .stagesmodel_merged import StagesModel_Merged
from .stagesmodel_pointclouds import StagesModel_PointClouds
from .utils import split_val

def get_model(model_name):
    return {
        'basemodel': BaseModel,
        'basemodel_connectivity' : BaseModel_Connectivity,
        'basemodel_distance' : BaseModel_Distance,
        'basemodel_connectivity_distance' : BaseModel_Connectivity_Distance,
        'basemodel_full': BaseModel_Full,
        'stagesmodel': StagesModel,
        'stagesmodel_distance': StagesModel_Distance,
        'stagesmodel_connectivity': StagesModel_Connectivity,
        'stagesmodel_connectivity_distance': StagesModel_Connectivity_Distance,
        'stagesmodel_merged': StagesModel_Merged,
        'stagesmodel_pointclouds': StagesModel_PointClouds,
    }[model_name]
