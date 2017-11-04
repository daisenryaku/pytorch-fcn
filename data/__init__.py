from utils import get_data_path
from pascal_voc import pascalVOCLoader

def get_loader(name):
    return {
            'pascal': pascalVOCLoader,
            }[name]

