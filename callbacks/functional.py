import os
import os.path
import uuid
import torch


def make_directory_available(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def save_checkpoint(checkpoint, checkpoint_name, save_dir):
    p = os.path.join(save_dir, checkpoint_name)
    torch.save(checkpoint, p)


def gen_experiment_unique_id_simple():
    return str(uuid.uuid4()).split('-')[0]


def gen_experiment_unique_id_full():
    return str(uuid.uuid4())


def get_experiment_unique_id(full_length=False):
    experiment_id = gen_experiment_unique_id_full() if full_length else gen_experiment_unique_id_simple()
    return experiment_id


# runs folder tree
# runs/
#     experiment.exid.{identifier}/
#         logs.exid.{identifier}/
#             tfevents...
#         save.exid.{identifier}/
#             checkpoint.exid.{identifier}.epoch.{epoch}.pth
#             checkpoint.exid.{identifier}.best.pth
#             ...
