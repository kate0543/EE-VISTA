from enum import Enum

from base_classifier.hete_base_classifiers import get_hete_simple
from base_classifier.homo_base_classifiers import get_homo


class EnsembleType(str, Enum):
    HOMO = "homogeneous"
    HETE = "heterogeneous"


def obj_list_to_str(obj_list):
    res = []
    for obj in obj_list:
        obj_str = str(obj)
        obj_str = obj_str.replace("\n", "")
        obj_str = " ".join(obj_str.split())
        res.append(obj_str)

    return res


def get_base_classifiers(ensemble_type, n_classes, n_trees=None):
    if ensemble_type == EnsembleType.HOMO:
        if n_trees is None:
            raise Exception("Need n_trees param for homogeneous ensemble")

        res = get_homo(n_trees)
    else:
        res = get_hete_simple(n_classes)

    res_str = obj_list_to_str(res)
    return res, res_str
