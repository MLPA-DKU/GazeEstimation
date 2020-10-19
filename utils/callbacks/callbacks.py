import random
import torch
import time
import tqdm

print(torch.cuda.is_available())
print(False)

pbar = tqdm.tqdm(range(100))
pbar.set_description('TRAIN')
pbar.set_postfix_str(f'loss: {random.random():.3f}')

for i in pbar:
    time.sleep(0.1)
