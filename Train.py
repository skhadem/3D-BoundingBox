from torch_lib.Dataset import *
from torch_lib.Model import Model, OrientationLoss


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torch.utils import data


import os
from argparse import ArgumentParser



def evaluate(model, dataloader):
    orient_loss_func = OrientationLoss
    model.eval()
    set_loss = 0
    for local_batch, local_labels in dataloader:
        truth_orient = local_labels["Orientation"].float().cuda()
        truth_conf = local_labels["Confidence"].long().cuda()

        local_batch = local_batch.float().cuda()
        [orient, conf] = model(local_batch)
        orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
        set_loss += orient_loss.item()
    avg_loss = set_loss / len(dataloader)    
    print(f"Average Orientation Loss: {avg_loss}")
    model.train()
    return

def make_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default="/home/thoro-ml/work/ml/3D-BoundingBox/Pallet-dataset/",
        help="Path to data root",
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Initial SGD learning rate",
    )
    parser.add_argument(
        "-m", "--momentum", type=float, default=0.9, help="SGD momentum"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Path to checkpointed weights to continue training from",
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Path to save checkpoints",
        default="weights"
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.6,
        help="L_dims loss coefficient"
    )
    parser.add_argument(
        "-w",
        type=float,
        default=0.4,
        help="L_loc loss coefficient"
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # hyper parameters
    epochs = args.epochs
    batch_size = args.batch_size
    alpha = args.alpha # scale L_dims (see paper for details)
    w = args.w # scale L_loc (see paper for details)

    print("Loading all detected objects in dataset...")
    data_root = args.data
    train_path = os.path.join(data_root, "training")
    test_path = os.path.join(data_root, "testing")
    train_dataset = Dataset(train_path)
    test_dataset = Dataset(test_path)
    ## As well as batch_size and num_workers
    # params = {"batch_size": batch_size, "shuffle": True, "num_workers": 10}
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    my_vgg = vgg.vgg19_bn(weights=vgg.VGG19_BN_Weights.IMAGENET1K_V1)
    model = Model(features=my_vgg.features).cuda()
    opt_SGD = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    orient_loss_func = OrientationLoss

    first_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt_SGD.load_state_dict(checkpoint["optimizer_state_dict"])
        first_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(
            "Found previous checkpoint: %s at epoch %s" % (args.checkpoint, first_epoch)
        )
        print("Resuming training....")

    total_num_batches = int(len(train_dataset) / batch_size)

    for epoch in range(first_epoch + 1, epochs + 1):
        curr_batch = 0
        passes = 0
        for local_batch, local_labels in train_loader:
            truth_orient = local_labels["Orientation"].float().cuda()
            truth_conf = local_labels["Confidence"].long().cuda()

            local_batch = local_batch.float().cuda()
            [orient, conf] = model(local_batch)

            orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)

            loss = conf_loss + w * orient_loss

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()

            if passes % 10 == 0:
                print(
                    "--- epoch %s | batch %s/%s --- [loss: %s]"
                    % (epoch, curr_batch, total_num_batches, loss.item())
                )
                passes = 0

            passes += 1
            curr_batch += 1

        # save after every 10 epochs
        if epoch % 2 == 0:
            name = os.path.join(args.save, f"epoch_{epoch}.pkl")
            print("====================")
            print("Evaluating epoch %s" % epoch)
            evaluate(model, test_loader)
            print("Done with epoch %s!" % epoch)
            print("Saving weights as %s ..." % name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt_SGD.state_dict(),
                    "loss": loss,
                },
                name,
            )
            print("====================")


if __name__ == "__main__":
    main()
