import torch


def load_batch(batch, device=None, non_blocking=False):
    batch = [*batch]
    batch = [b.to(device=device, non_blocking=non_blocking) for b in batch] if device is not None else batch
    return batch


def update(batch, model, optimizer, criterion, device=None):
    model.train()
    optimizer.zero_grad()
    inputs, targets = load_batch(batch, device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return inputs, targets, outputs, loss


def evaluate(batch, model, criterion, evaluator, device=None):
    model.eval()
    with torch.no_grad():
        inputs, targets = load_batch(batch, device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        score = evaluator(outputs, targets)
    return loss, score


def train(dataloader, model, optimizer, criterion, evaluator=None, device=None):
    for idx, batch in enumerate(dataloader):
        _, targets, outputs, loss = update(batch, model, optimizer, criterion, device)
        score = evaluator(outputs, targets) if evaluator else None


def valid(dataloader, model, criterion, evaluator, device=None):
    for idx, batch in enumerate(dataloader):
        loss, score = evaluate(batch, model, criterion, evaluator, device)
