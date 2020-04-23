import sys
import numpy as np
import torch
from adversarial_model import UNet, GANLoss, Discriminator
from argparser import parse_adv_train_arguments
from datasets import PascalVOCDataset
from model import VGGBase, SSD300
from utils import create_data_lists, process_boxes_and_labels, save_adversarial_checkpoint, AverageMeter, \
    create_image_with_boxes, one_hot_embedding, make_dot
import torchviz

keep_difficult = True
workers = 4
max_boxes = 37
num_classes = 21
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(batch_size, continue_training, exp_name, learning_rate, num_epochs, print_freq, run_colab):
    # Data
    data_folder = create_data_lists(run_colab)
    train_dataset = PascalVOCDataset(data_folder,
                                     split='test',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Networks
    checkpoint = torch.load(exp_name / "checkpoint_ssd300.pth.tar", map_location=device)
    print(f"Number of training epochs for detection network: {checkpoint['epoch']}")
    detection_network = checkpoint['model']
    detection_network = SSD300(n_classes=num_classes)

    if continue_training:
        adversarial_checkpoint = torch.load(exp_name / checkpoint, map_location=device)
        discriminator = adversarial_checkpoint['adversarial_model']
        optimizer = adversarial_checkpoint['optimizer']
        start_epoch = adversarial_checkpoint['epoch']
        print(f"Continue training of adversarial network from epoch {start_epoch}")
    else:
        start_epoch = 0
        image_encoder = VGGBase()
        discriminator = Discriminator(num_classes)
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)
    discriminator, image_encoder = discriminator.to(device), image_encoder.to(device)
    loss_function = GANLoss('vanilla').to(device)
    losses = AverageMeter()  # loss


    for epoch in range(start_epoch, num_epochs):
        for j, (images, boxes, labels, _) in enumerate(train_loader):
            images = images.to(device)
            with torch.no_grad():
                _, image_embedding = image_encoder(images)
            random_box_indices = [np.random.randint(len(box)) for box in boxes]
            random_boxes = torch.stack([box[random_box_indices[i]] for i, box in enumerate(boxes)]).to(device)
            random_labels = torch.stack([one_hot_embedding(label[random_box_indices[i]], num_classes) for i, label in enumerate(labels)]).to(device)
            pred_real = discriminator(random_boxes, random_labels, image_embedding)
            loss_real = loss_function(pred_real, 1)

            with torch.no_grad():
                predicted_locs, predicted_scores = detection_network.forward(images)
                pred_boxes, pred_labels, _ = detection_network.detect_objects(predicted_locs,
                                                                                                       predicted_scores,
                                                                                                       min_score=0.2,
                                                                                                       max_overlap=0.45,
                                                                                                       top_k=200)
            random_box_indices = [np.random.randint(len(box)) for box in pred_boxes]
            random_fake_boxes = torch.stack([box[random_box_indices[i]] for i, box in enumerate(pred_boxes)]).to(device)
            random_fake_labels = torch.stack([one_hot_embedding(label[random_box_indices[i]], num_classes) for i, label in enumerate(pred_labels)]).to(device)
            pred_fake = discriminator(random_fake_boxes, random_fake_labels, image_embedding)
            loss_fake = loss_function(pred_fake, 0)

            total_loss = loss_fake + loss_real
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.update(total_loss.item(), images.size(0))
            if j % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, j, len(train_loader), loss=losses))
        save_adversarial_checkpoint(epoch, discriminator, image_encoder, optimizer, exp_name)



if __name__ == "__main__":
    arguments = parse_adv_train_arguments(sys.argv[1:])
    main(**arguments)