import torch
import utils


def update(batch, model, optimizer, criterion, device=None):
    model.train()
    optimizer.zero_grad()
    inputs, targets = utils.load_batch(batch, device=device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return inputs, targets, outputs, loss


def update_with_amp(batch, model, optimizer, criterion, scaler, device=None):
    model.train()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        inputs, targets = utils.load_batch(batch, device=device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return inputs, targets, outputs, loss
