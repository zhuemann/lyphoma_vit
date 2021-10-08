import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#plt.style.use("ggplot")

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import timm

import gc
import os
import time
import random
from datetime import datetime

from PIL import Image, ImageOps

from os import listdir
from os.path import isfile, join
from os.path import exists

from tqdm import tqdm
from sklearn import model_selection, metrics
from skimage import io



def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class MIPSDataset(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, data_path='Z:/Lymphoma_UW_Retrospective/Data/mips/', mode="train", transforms=None):
        super().__init__()
        self.df_data = df.values
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode

        #self.data_dir = "train_images" if mode == "train" else "test_images"

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        img_name, label = self.df_data[index]
        if exists(os.path.join(self.data_path, 'Group_1_2_3_curated', img_name)):
            data_dir = "Group_1_2_3_curated"
        if exists(os.path.join(self.data_path, 'Group_4_5_curated', img_name)):
            data_dir = "Group_4_5_curated"
        img_path = os.path.join(self.data_path, data_dir, img_name)

        #img2 = io.imread(img_path)
        #img2_norm = img2 * (255/65535)

        try:
            #img = Image.open(img_path).convert("RGB")
            #img = Image.open(img_path).convert("L")
            img_raw = io.imread(img_path)
            img_norm = img_raw * (255 / 65535)
            img = Image.fromarray(np.uint8(img_norm)).convert("RGB")

        except:
            print("can't open")
            print(img_path)

        if self.transforms is not None:
            image = self.transforms(img)
            try:
                #image = self.transforms(img)
                image = self.transforms(img)
            except:
                print("can't transform")
                print(img_path)
        else:
            image = img

        return image, label

def vit_train():

    seed_everything(1001)

    # model specific global variables
    IMG_SIZE = 224
    BATCH_SIZE = 16
    LR = 2e-05
    GAMMA = 0.7
    N_EPOCHS = 30


    negative_dir = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated'
    positive_dir = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated'

    # gets all the file names in and puts them in a list
    neg_files = [f for f in listdir(negative_dir) if isfile(join(negative_dir, f))]
    pos_files = [f for f in listdir(positive_dir) if isfile(join(positive_dir, f))]

    if "Thumbs.db" in neg_files: neg_files.remove("Thumbs.db")
    if "Thumbs.db" in pos_files: pos_files.remove("Thumbs.db")

    print(f"num 0 labels: " + str(len(neg_files)))
    print(f"num 1 labels: " + str(len(pos_files)))
    print(f"fraction 0/(0+1): " + str(len(neg_files)/(len(neg_files) + len(pos_files))))

    # creates all the labels for each file
    labels = []
    for i in range(0, len(neg_files)):
        labels.append(0)
        #labels.append([1,0])

    for i in range(0, len(pos_files)):
        neg_files.append(pos_files[i])
        labels.append(1)
        #labels.append([0,1])

    df = pd.DataFrame()

    df['image_id'] = neg_files
    df['label'] = labels

    train_df, valid_df = model_selection.train_test_split(
        df, test_size=0.1, random_state=42, stratify=df.label.values
    )

    # create image augmentations
    transforms_train = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            #transforms.RandomResizedCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    model = ViTBase16(n_classes=2, pretrained=True)

    # Start training processes
    device = torch.device("cuda")

    def _run():
        train_dataset = MIPSDataset(train_df, transforms=transforms_train)
        valid_dataset = MIPSDataset(valid_df, transforms=transforms_valid)

        #train_dataset = MIPSDataset(train_df, transforms=None)
        #valid_dataset = MIPSDataset(valid_df, transforms=None)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            sampler=None,
            drop_last=True,
            num_workers=8,
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=BATCH_SIZE,
            sampler=None,
            drop_last=True,
            num_workers=8,
        )

        criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCEWithLogitsLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        # device = xm.xla_device()
        model.to(device)

        # lr = LR * xm.xrt_world_size()
        lr = LR
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # xm.master_print(f"INITIALIZING TRAINING ON {xm.xrt_world_size()} TPU CORES")
        start_time = datetime.now()
        # xm.master_print(f"Start Time: {start_time}")
        print(f"Start Time: {start_time}")

        logs = fit_tpu(
            model=model,
            epochs=N_EPOCHS,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        print(f"Execution time: {datetime.now() - start_time}")
        # xm.master_print(f"Execution time: {datetime.now() - start_time}")

        # xm.master_print("Saving Model")
        print("Saving Model")
        # xm.save(
        #    model.state_dict(), f'model_5e_{datetime.now().strftime("%Y%m%d-%H%M")}.pth'
        # )



    def _mp_fn(rank, flags):
        torch.set_default_tensor_type("torch.FloatTensor")
        a = _run()

    _run()


class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):

        super(ViTBase16, self).__init__()

        #self.model = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        if pretrained:
            MODEL_PATH = ("C:/Users/zmh001/Documents/vit_model/jx_vit_base_p16_224-80ecf9dd.pth/jx_vit_base_p16_224-80ecf9dd.pth")
            self.model.load_state_dict(torch.load(MODEL_PATH))

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):

        x = self.model(x)

        return x

    def train_one_epoch(self, train_loader, criterion, optimizer, device):
        # keep track of training loss
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        ###################
        # train the model #
        ###################

        self.model.train()

        for i, (data, target) in tqdm(enumerate(train_loader)):


            # move tensors to GPU if CUDA is available
            if device.type == "cuda":
                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.forward(data)
            #output = torch.squeeze(output)
            print("output:")
            print(output)
            print("target")
            print(target)

            print("max of output")
            print(output.argmax(dim=1))

            #target = target.float()
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Calculate Accuracy
            accuracy = (output.argmax(dim=1) == target).float().mean()
            # update training loss and accuracy
            epoch_loss += loss
            epoch_accuracy += accuracy

            optimizer.step()

        return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader)

    def validate_one_epoch(self, valid_loader, criterion, device):
        # keep track of validation loss
        valid_loss = 0.0
        valid_accuracy = 0.0

        ######################
        # validate the model #
        ######################
        self.model.eval()
        for data, target in tqdm(valid_loader):
            # move tensors to GPU if CUDA is available
            if device.type == "cuda":
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                print("output")
                print(output)
                print("target")
                print(target)
                # calculate the batch loss
                loss = criterion(output, target)
                # Calculate Accuracy
                accuracy = (output.argmax(dim=1) == target).float().mean()
                # update average validation loss and accuracy
                valid_loss += loss
                valid_accuracy += accuracy

        return valid_loss / len(valid_loader), valid_accuracy / len(valid_loader)


def fit_tpu(
        model, epochs, device, criterion, optimizer, train_loader, valid_loader=None
):
    valid_loss_min = np.Inf  # track change in validation loss

    # keeping track of losses as it happen
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in range(1, epochs + 1):
        gc.collect()
        # para_train_loader = pl.ParallelLoader(train_loader, [device])

        # xm.master_print(f"{'='*50}")
        print(f"{'=' * 50}")
        # xm.master_print(f"EPOCH {epoch} - TRAINING...")
        print(f"EPOCH {epoch} - TRAINING...")
        # train_loss, train_acc = model.train_one_epoch(
        #    para_train_loader.per_device_loader(device), criterion, optimizer, device
        # )
        train_loss, train_acc = model.train_one_epoch(
            train_loader, criterion, optimizer, device
        )
        print(f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, ACCURACY: {train_acc}\n")
        # xm.master_print(
        #    f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, ACCURACY: {train_acc}\n"
        # )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        gc.collect()

        # moves towards valid_loader
        print("start valid_laoder")

        if valid_loader is not None:
            gc.collect()
            # para_valid_loader = pl.ParallelLoader(valid_loader, [device])
            # xm.master_print(f"EPOCH {epoch} - VALIDATING...")
            print(f"EPOCH {epoch} - VALIDATING...")
            valid_loss, valid_acc = model.validate_one_epoch(
                valid_loader, criterion, device
            )
            print(f"\t[VALID] LOSS: {valid_loss}, ACCURACY: {valid_acc}\n")
            # xm.master_print(f"\t[VALID] LOSS: {valid_loss}, ACCURACY: {valid_acc}\n")
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            gc.collect()

            # save model if validation loss has decreased
            # if valid_loss <= valid_loss_min and epoch != 1:
            # xm.master_print(
            #    "Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(
            #        valid_loss_min, valid_loss
            #    )
            # )
            #                 xm.save(model.state_dict(), 'best_model.pth')

            valid_loss_min = valid_loss

    return {
        "train_loss": train_losses,
        "valid_losses": valid_losses,
        "train_acc": train_accs,
        "valid_acc": valid_accs,
    }


