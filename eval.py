from pathlib import Path
from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 32
workers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.cpu() for b in boxes]
            labels = [l.cpu()for l in labels]
            difficulties = [d.cpu() for d in difficulties]

            det_boxes.extend([box.cpu() for box in det_boxes_batch])
            det_labels.extend([label.cpu() for label in det_labels_batch])
            det_scores.extend([score.cpu() for score in det_scores_batch])
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)
    model.train()

if __name__ == '__main__':
    run_local = False
    if run_local:
        voc_2007 = Path("/Users/laurenssamson/Documents/Projects/Chess_notation/pascal_VOC/VOCdevkit/VOC2007")
        voc_test = Path("/Users/laurenssamson/Documents/Projects/Chess_notation/pascal_VOC/VOCdevkit_test/VOC2007")

        voc_2012 = Path("/Users/laurenssamson/Documents/Projects/Chess_notation/pascal_VOC/VOCdevkit2012/VOC2012")
        out_path = Path("/Users/laurenssamson/Documents/Projects/Chess_notation/object_detecion/json")
        data_folder = './json'  # folder with data files
        checkpoint = 'checkpoint_ssd300.pth.tar'

    else:
        voc_2007 = Path("/content/data/VOCdevkit/VOC2007")
        voc_test = Path("/content/data/VOCdevkit/VOC2007")

        voc_2012 = Path("/content/data/VOCdevkit/VOC2012")
        out_path = Path("/content/data/VOCdevkit")
        data_folder = "/content/data/VOCdevkit"  # folder with data files
        checkpoint = '/content/gdrive/My Drive/checkpoint_ssd300.pth.tar'

    out_path.mkdir(exist_ok=True)
    create_data_lists(voc_2007, voc_2012, voc_test, out_path)
    test_dataset = PascalVOCDataset(data_folder,
                                    split='train',
                                    keep_difficult=keep_difficult)
    test_dataset.images = test_dataset.images[:500]
    test_dataset.objects = test_dataset.objects[:500]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(checkpoint, map_location=device)
    model = checkpoint['model']
    print(f"Number of epoch trained: {checkpoint['epoch']}")
    model = model.to(device)

    # Switch to eval mode
    model.eval()
    evaluate(test_loader, model)

