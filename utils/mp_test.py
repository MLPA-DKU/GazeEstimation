import torch
import torch.multiprocessing as mp


class CheckpointModule:

    def __init__(self, checkpoint, f):
        self.file_manager = ...
        self.checkpoint = checkpoint
        self.f = f
        self.p = None

    def __persist__(self, checkpoint, f):
        torch.save(checkpoint, f)
        print('checkpoint completed.')

    def __save__(self):
        self.p = mp.Process(target=self.__persist__, args=(self.checkpoint, self.f,))
        self.p.start()

    def save(self):
        self.__save__()


class CheckpointBlueprint:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return ...
