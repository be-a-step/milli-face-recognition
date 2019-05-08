import sys
import sampling_image
import triming_face
import train_model
from models.ir_model import IRModel
import torch.nn as nn
import torchvision.models as models


def print_usage():
    print("print usage")


if len(sys.argv) < 2:
    print_usage()

option = sys.argv[1]
is_all = option == "all"
is_create_dataset = option == "create_dataset"

cascade_conf_path = "./resources/conf/cv2-cascade-conf/lbpcascade_animeface.xml"
source_directory = './resources/sources/movies'
images_dist = './resources/sources/images'
dataset_path = './resources/dataset/faces'

batch_size = 8
validation_split = .2
epoch = 20
base_lr = 1e-4
name = "IRModel"
criterion = nn.CrossEntropyLoss()

if option == "sampling" or is_all or is_create_dataset:
    sampling_rate = 10
    margin = (20, 20, 20, 0)

    sampling_image.sampling_from_movies(
        source_directory, images_dist, sampling_rate)

if option == "triming" or is_all or is_create_dataset:
    triming_face.triming_images(
        images_dist, dataset_path, margin, cascade_conf_path)

if option == "train" or is_all:
    is_resnet = True
    trainer = train_model.ModelTrainer(
        dataset_path, batch_size, validation_split, is_resnet)
    model = IRModel()
    if is_resnet:
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, trainer.num_classes)
        name = "Resnet50"
    trainer.train(
        epoch=epoch,
        model=model,
        base_lr=base_lr,
        loss_fn=criterion,
        name=name)
