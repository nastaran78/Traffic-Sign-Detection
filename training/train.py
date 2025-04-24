# FASTER RCNN DEFINITION
import torch
import torchvision


#TRAIN
def train_fn(model, optimizer, loader, device) -> torch.nn.Module:
    model.to(device)



    model.train()
    for epoch in range(5):
        for i, (imgs, targets) in enumerate(loader):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()


        print(f"Epoch {epoch}: Loss = {losses.item():.4f}")

    return model
