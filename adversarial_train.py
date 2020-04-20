import sys
from pathlib import Path

import torch
from adversarial_model import UNet, Encoder, GANLoss
from argparser import parse_adv_train_arguments
from datasets import PascalVOCDataset
from utils import create_data_lists, process_boxes_and_labels, save_adversarial_checkpoint, AverageMeter

keep_difficult = True
batch_size = 8
workers = 4
max_boxes = 37
num_classes = 21
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(batch_size, continue_training, exp_name, learning_rate, num_epochs, print_freq, run_colab):
    # Data
    data_folder = create_data_lists(run_colab)
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Networks
    checkpoint = torch.load(exp_name / "checkpoint_ssd300.pth.tar", map_location=device)
    print(f"Number of training epochs for detection network: {checkpoint['epoch']}")
    detection_network = checkpoint['model']
    if continue_training:
        adversarial_checkpoint = torch.load(exp_name / checkpoint, map_location=device)
        adversarial_model = adversarial_checkpoint['adversarial_model']
        box_encoder = adversarial_checkpoint['box_encoder']
        label_encoder = adversarial_checkpoint['label_encoder']
        optimizer = adversarial_checkpoint['optimizer']
        start_epoch = adversarial_checkpoint['epoch']
        print(f"Continue training of adversarial network from epoch {start_epoch}")
    else:
        start_epoch = 0
        adversarial_model = UNet(3, 1)
        box_encoder = Encoder(4 * max_boxes)
        label_encoder = Encoder(num_classes * max_boxes)
        optimizer = torch.optim.Adam(list(adversarial_model.parameters()) + list(box_encoder.parameters())
                                    + list(label_encoder.parameters()), learning_rate)
    box_encoder, label_encoder, adversarial_model = box_encoder.to(device), \
                                                    label_encoder.to(device), adversarial_model.to(device)
    loss_function = GANLoss('vanilla').to(device)
    losses = AverageMeter()  # loss


    for epoch in range(start_epoch, num_epochs):
        for j, (images, boxes, labels, _) in enumerate(train_loader):
            images = images.to(device
                               )
            boxes_real, labels_real = process_boxes_and_labels(boxes, labels, num_classes, max_boxes, device)
            box_embedding_real = box_encoder(boxes_real)
            label_embedding_real = label_encoder(labels_real)
            pred_real = adversarial_model(images, box_embedding_real, label_embedding_real)
            loss_real = loss_function(pred_real, 1)

            with torch.no_grad():
                predicted_locs, predicted_scores = detection_network.forward(images)
                pred_boxes, pred_labels, _ = detection_network.detect_objects(predicted_locs,
                                                                                                       predicted_scores,
                                                                                                       min_score=0.2,
                                                                                                       max_overlap=0.45,
                                                                                                       top_k=200)
            boxes_fake, labels_fake = process_boxes_and_labels(pred_boxes, pred_labels, num_classes, max_boxes, device)
            box_embedding_fake = box_encoder(boxes_fake)
            label_embedding_fake = label_encoder(labels_fake)
            pred_fake = adversarial_model(images, box_embedding_fake, label_embedding_fake)
            loss_fake = loss_function(pred_fake, 1)

            total_loss = loss_fake + loss_real
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.update(total_loss.item(), images.size(0))
            if j % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, j, len(train_loader), loss=losses))
                print(pred_fake)
                print("-" * 50)
                print(pred_real)
                print("-" * 50)
        save_adversarial_checkpoint(epoch, adversarial_model, box_encoder, label_encoder, optimizer, exp_name)



if __name__ == "__main__":
    arguments = parse_adv_train_arguments(sys.argv[1:])
    main(**arguments)