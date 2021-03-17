from PIL import ImageDraw
import gc
import torch
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
