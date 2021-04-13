from PIL import ImageDraw
import gc
import random
import numpy as np
import torch
import torch.backends.cudnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def denorm(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def load_batch(batch, device=None, non_blocking=False):
    batch = [*batch]
    batch = [b.to(device=device, non_blocking=non_blocking) for b in batch] if device is not None else batch
    return batch


def salvage_memory():
    torch.cuda.empty_cache()
    gc.collect()


def enable_easy_debug(enable=False):
    torch.autograd.set_detect_anomaly(enable)


def enable_reproducibility(enable=False, random_seed=42, parallel=False):
    if enable:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
    if parallel:
        torch.cuda.manual_seed_all(random_seed)


def update(batch, model, optimizer, criterion, device=None):
    model.train()
    optimizer.zero_grad()
    inputs, targets = load_batch(batch, device=device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return inputs, targets, outputs, loss


def update_with_amp(batch, model, optimizer, criterion, scaler, device=None):
    model.train()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        inputs, targets = load_batch(batch, device=device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return inputs, targets, outputs, loss


def evaluate(batch, model, criterion, evaluator, device=None):
    model.eval()
    with torch.no_grad():
        inputs, targets = load_batch(batch, device=device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        score = evaluator(outputs, targets)
    return loss, score


def print_result(epoch, epochs, idx, dataloader, losses, scores, header=None):
    print(f'\r[{header}] Epoch[{epoch + 1:>{len(str(epochs))}}/{epochs}] - '
          f'batch[{idx + 1:>{len(str(len(dataloader)))}}/{len(dataloader)}] - '
          f'loss: {np.nanmean(losses):.3f} - angular error: {np.nanmean(scores):.3f}', end='')


def print_result_on_epoch_end(epoch, epochs, scores):
    print(f'\n[ RES ] Epoch[{epoch + 1:>{len(str(epochs))}}/{epochs}] - '
          f'angular error (°) [{np.nanmean(scores):.3f}|{np.nanstd(scores):.3f}|'
          f'{np.min(scores):.3f}|{np.max(scores):.3f}:MEAN|STD|MIN|MAX]')


def visualize_gaze_direction_gaze360(tensor, gaze, prediction=None, save=None, length=200):

    # PyTorch image tensor shape: CHW
    # Input 'tensor' should be 3D tensor (Channel, Height, Width)
    # Input 'gaze' should be 2D vector (theta, phi)

    def transform_coordinate_system(theta, phi, o):
        x = int((-1 * torch.sin(theta) * torch.cos(phi) * length + o[0]).item())
        y = int((-1 * torch.sin(phi) * length + o[1]).item())
        return x, y

    image = transforms.ToPILImage()(tensor.detach())
    image_overlay = ImageDraw.Draw(image)

    endpoint = transform_coordinate_system(gaze[0], gaze[1], (image.width // 2, image.height // 2))
    image_overlay.line([(int(image.width // 2), int(image.height // 2)), endpoint], fill='blue', width=3)

    if prediction:
        predict = transform_coordinate_system(prediction[0], prediction[1], (image.width // 2, image.height // 2))
        image_overlay.line([(int(image.width // 2), int(image.height // 2)), predict], fill='blue', width=3)

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    plt.close()

    if save is not None:
        image.save(save)
