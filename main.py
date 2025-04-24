import logging

import torch
import yaml

from data.dataloader import FasterRCNNDataset, collate_fn
from torch.utils.data import DataLoader
import  argparse

from training.faster_rcnn import create_model
from training.train import train_fn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main(args):
    mode = args.mode
    log.info("Mode: %s", mode)
    # Load the configuration file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if mode == 'train':
        # Initialize the dataset and dataloader
        dataset = FasterRCNNDataset(cfg["dataset"]["train_dir"], img_size=cfg["dataset"]["img_size"])
        dataloader = DataLoader(dataset, batch_size=15, shuffle=False, collate_fn=collate_fn)

        for images, targets in dataloader:
            print(images, targets)

        model = create_model(len(cfg["model"]["classes"]))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        model = train_fn(model, optimizer, dataloader, device)
        torch.save(model.state_dict(), args.model_path)
    elif mode == 'test':
        model = create_model(len(cfg["model"]["classes"]))
        model.load_state_dict(torch.load(args.model_path))

        test_dataset = FasterRCNNDataset(cfg["dataset"]["test_dir"], img_size=cfg["dataset"]["img_size"])
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        total_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i, (img, targets) in enumerate(test_dataloader):
            model.eval()
            with torch.no_grad():
                image = img[0].to(device)
                prediction = model([image])[0] #TODO: extract score

            model.train() # Set the model back to training mode to get the loss for the given image and logit
            with torch.no_grad():
                output = {k: v.to(device) for k, v in targets[0].items()}
                loss_dict = model([image], [output])
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses
                print(f"total_loss {total_loss / i}", end='\r', flush=True)

        print(total_loss / len(test_dataset))




if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Train or test the model")
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "valid"],
        required=True, help="Mode to run the script",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--model_path", type=str, default='model_weights.pth',
        help="Model weights path"
    )
    args = parser.parse_args()

    main(args)