from pathlib import Path

import torch
from adversarial_model import UNet, Encoder, GANLoss
from datasets import PascalVOCDataset
from utils import create_data_lists, process_boxes_and_labels, save_adversarial_checkpoint

run_colab = False
keep_difficult = True
batch_size = 8
workers = 4
max_boxes = 37
num_classes = 21
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Data
    data_folder = create_data_lists(run_colab)
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Networks
    exp_name = Path(".")
    checkpoint = exp_name / "checkpoint_ssd300.pth.tar"
    checkpoint = torch.load(checkpoint, map_location=device)
    print(f"Number of training epochs: {checkpoint['epoch']}")
    detection_network = checkpoint['model']
    adversarial_model = UNet(3, 1)
    box_encoder = Encoder(4 * max_boxes)
    label_encoder = Encoder(num_classes * max_boxes)

    # Training
    num_epochs = 10
    loss_function = GANLoss('vanilla')
    learning_rate = 0.01
    optimizer = torch.optim.SGD(list(adversarial_model.parameters()) + list(box_encoder.parameters())
                               + list(label_encoder.parameters()), learning_rate, momentum=0.99)

    for i in range(num_epochs):
        for _, (images, boxes, labels, _) in enumerate(train_loader):
            images = images.to(device)
            boxes_real, labels_real = process_boxes_and_labels(boxes, labels, num_classes, max_boxes)
            box_embedding_real = box_encoder(boxes_real)
            label_embedding_real = label_encoder(labels_real)
            pred_real = adversarial_model(images, box_embedding_real, label_embedding_real)
            loss_real = loss_function(pred_real, 1)

            with torch.no_grad():
                predicted_locs, predicted_scores = detection_network.forward(images)
                det_boxes_batch, det_labels_batch, _ = detection_network.detect_objects(predicted_locs,
                                                                                                       predicted_scores,
                                                                                                       min_score=0.2,
                                                                                                       max_overlap=0.45,
                                                                                                       top_k=200)
            boxes_fake, labels_fake = process_boxes_and_labels(det_boxes_batch, det_labels_batch, num_classes, max_boxes)
            box_embedding_fake = box_encoder(boxes_fake)
            label_embedding_fake = label_encoder(labels_fake)
            pred_fake = adversarial_model(images, box_embedding_fake, label_embedding_fake)
            loss_fake = loss_function(pred_fake, 0)
            total_loss = loss_fake + loss_real
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            break
        save_adversarial_checkpoint(i, adversarial_model, box_encoder, label_encoder, optimizer, exp_name)



if __name__ == "__main__":
    main()