import torch
import utils.visualization as viz


def train(dataloader, model, criterion, evaluator, optimizer, writer, args):

    alpha = 0.5

    model.train()
    for i, batch in enumerate(dataloader):
        face, _, gaze = batch
        face, gaze = face.to(args.device), gaze.to(args.device)

        outputs, cam_mask, cam_gaze = model(face)
        acc_loss = criterion(outputs, gaze)
        cam_loss = criterion(cam_mask, cam_gaze)
        loss = alpha * acc_loss + (1 - alpha) * cam_loss
        accuracy = evaluator(outputs, gaze)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('training loss', loss.item(), args.epoch * len(dataloader) + i)
        writer.add_scalar('training accuracy', accuracy.item(), args.epoch * len(dataloader) + i)

        print(f'Epoch[{args.epoch + 1:4d}/{args.epochs:4d}] - batch[{i + 1:4d}/{len(dataloader):4d}]'
              f' - loss: {loss.item():7.3f} - accuracy: {accuracy.item():7.3f}')
