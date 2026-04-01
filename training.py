import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import cuda, no_grad, load
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from pathlib import Path
from copy import deepcopy

from dataset_class import MetalDefectDataset
from unet import UNet

from loss import TverskyLoss


def dice_score(pred, target, smooth=1.):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()

    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1.):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)


def main():
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    epochs = 100
    batch_size = 10
    learning_rate = 1e-4

    train_model = True
    test_model = True

    unet_best_model_name = 'best_model'

    num_classes = 1  # defect (binary segmentation)

    unet_model = UNet(n_class=num_classes)

    # Pass U-Net to the available device.
    unet_model = unet_model.to(device)

    # Define the optimizer and give the parameters of the CNN model to an optimizer.
    optimizer = Adam(params=unet_model.parameters(), lr=learning_rate)

    # Instantiate the loss function as a class.
    loss_function = TverskyLoss(alpha=0.3, beta=.7)

    parent_dir = "../data"
    defect_types = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Free"]

    all_images = []
    for type in defect_types:
        root = Path(parent_dir, type)
        if type == "MT_Free":
            free_images = sorted(root.glob("*.jpg"))
            # Undersample 80 images from the MT_Free dataset randomly
            np.random.seed(3 * 22)
            free_images_sampled = np.random.choice(free_images, 80, replace=False)
            all_images += list(free_images_sampled)
        else:
            all_images += sorted(root.glob("*.jpg"))

    # split image dataset to train, validation and test sets
    train_imgs, temp_imgs = train_test_split(
        all_images, test_size=0.3, random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.5, random_state=42
    )

    train_dataset = MetalDefectDataset(image_paths=train_imgs)
    val_dataset = MetalDefectDataset(image_paths=val_imgs)
    test_dataset = MetalDefectDataset(image_paths=test_imgs)

    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_data_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0)
    test_data_loader = DataLoader(test_dataset, 1, shuffle=False)

    # Variables for early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 30
    patience_counter = 0

    best_model = None

    if train_model:
        # Start training.
        print("Starting training...")
        for epoch in range(epochs):
            # Lists to hold the corresponding losses of each epoch.
            epoch_loss_training = []
            epoch_loss_validation = []

            # Indicate that we are in training mode, so (e.g.) dropout
            # will function
            unet_model.train()

            for inputs, labels, _ in train_data_loader:

                # Zero the gradient of the optimizer.
                optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = unet_model(inputs)
                outputs = F.sigmoid(logits)

                loss = loss_function(outputs, labels)

                # Do the backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                # Append the loss of the batch
                epoch_loss_training.append(loss.item())

            #print(logits.min(), logits.max())
            # Indicate that we are in evaluation mode
            unet_model.eval()

            # Say to PyTorch not to calculate gradients, so everything will
            # be faster.
            with no_grad():

                sample_img, sample_mask = val_dataset[7]

                input_tensor = torch.tensor(sample_img, dtype=torch.float32).unsqueeze(0).to(device)

                output = unet_model(input_tensor)
                pred = F.sigmoid(output)

                pred_mask = (pred > 0.5).float()

                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(sample_img.squeeze(), cmap='gray')
                plt.title("Input")

                plt.subplot(1, 3, 2)
                plt.imshow(sample_mask.squeeze(), cmap='gray')
                plt.title("Ground Truth")

                plt.subplot(1, 3, 3)
                #plt.imshow(pred.cpu().squeeze(), cmap='gray')
                plt.imshow(pred_mask.cpu().squeeze(), cmap='gray')
                plt.title("Prediction")
                if epoch % 5 == 0:
                    plt.savefig(f"epoch_{epoch}.png")

                plt.show(block=False)
                plt.pause(4)
                plt.close()

                for inputs, labels, _ in val_data_loader:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = unet_model(inputs)
                    outputs = F.sigmoid(outputs)

                    loss = loss_function(outputs, labels)

                    # Append the validation loss.
                    epoch_loss_validation.append(loss.item())

            # Calculate mean losses.
            epoch_loss_validation = np.array(epoch_loss_validation).mean()
            epoch_loss_training = np.array(epoch_loss_training).mean()

            print(f'Epoch: {epoch:03d} | '
                  f'Mean training loss: {epoch_loss_training:7.4f} | '
                  f'Mean validation loss {epoch_loss_validation:7.4f}')

            # Check early stopping conditions.
            if epoch_loss_validation < lowest_validation_loss:
                lowest_validation_loss = epoch_loss_validation
                patience_counter = 0
                best_model = deepcopy(unet_model.state_dict())
                best_validation_epoch = epoch
                torch.save(unet_model.state_dict(), "./best_model")
            else:
                patience_counter += 1

            if (patience_counter >= patience) or (epoch == epochs - 1):

                print('\nExiting due to early stopping', end='\n\n')
                print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')

    if test_model:
        print('Starting testing', end=' | ')
        testing_loss = []
        clean_dice_scores = []
        defect_dice_scores = []
        defect_iou_scores = []
        results = []

        # Load best model
        try:
            unet_model.load_state_dict(load(unet_best_model_name, map_location=device))
        except (FileNotFoundError, RuntimeError):
            unet_model.load_state_dict(best_model)

        unet_model.eval()

        with no_grad():
            for inputs, masks, paths in test_data_loader:
                inputs = inputs.to(device)
                masks = masks.to(device)

                logits = unet_model(inputs)
                outputs = F.sigmoid(logits)

                preds = (outputs > 0.5).float()

                dice = dice_score(preds, masks)
                iou = iou_score(preds, masks)

                loss = loss_function(outputs, masks)

                # Append the validation loss, dice score, and IoU score
                testing_loss.append(loss.item())
                if "MT_Free" in paths[0]:
                    clean_dice_scores.append(dice.item())
                else:
                    defect_dice_scores.append(dice.item())
                    results.append((dice.item(), iou.item(), inputs, masks, preds))  # only include images with defects in results
                    defect_iou_scores.append(iou.item())

        testing_loss = np.array(testing_loss).mean()
        clean_dice_scores_mean = np.array(clean_dice_scores).mean()
        defect_dice_scores_mean = np.array(defect_dice_scores).mean()
        iou_scores_mean = np.array(defect_iou_scores).mean()
        results.sort(key=lambda x: x[0])

        print(f'Mean testing loss: {testing_loss:7.4f}')
        print(f'Mean dice score: {defect_dice_scores_mean:7.4f}')
        print(f'Mean IoU score: {iou_scores_mean:7.4f}')

        best = results[-1]
        typical = results[len(results)//2]
        failure = results[0]
        examples = [best, typical, failure]

        plt.figure(figsize=(12, 8))
        for i, ex in enumerate(examples):
            plt.subplot(3, 4, 1 + i * 4)
            plt.imshow(ex[2].squeeze(), cmap='gray')
            plt.title("Input")

            plt.subplot(3, 4, 2 + i * 4)
            plt.imshow(ex[3].squeeze(), cmap='gray')
            plt.title("Ground Truth")

            plt.subplot(3, 4, 3 + i * 4)
            plt.imshow(ex[4].cpu().squeeze(), cmap='gray')
            plt.title("Prediction")

            plt.subplot(3, 4, 4 + i * 4)
            plt.text(0.1, 0.4, f"Dice: {ex[0]:.3f}", fontsize=12)
            plt.text(0.1, 0.6, f"IoU: {ex[1]:.3f}", fontsize=12)
            plt.axis("off")

        plt.tight_layout()
        plt.show()
        #plt.savefig("segmentation_results.png")


if __name__ == '__main__':
    main()
