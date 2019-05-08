import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import numpy as np
from pathlib import Path


DATA_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

DATA_TRANSFORM_RES = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

SHUFFLE_DATASET = True
RANDOM_SEED = 42


class ModelTrainer(object):
    def __init__(self, dataset_path, batch_size, validation_split, is_resnet):
        self.create_dataset_loader(
            dataset_path,
            batch_size,
            validation_split,
            is_resnet)
        self.cuda_avail = torch.cuda.is_available()
        if self.cuda_avail:
            print("cuda available")

    def create_dataset_loader(
            self,
            dataset_path,
            batch_size,
            validation_split,
            is_resnet):
        # データセットを読み込む
        data_transform = DATA_TRANSFORM
        if is_resnet:
            data_transform = DATA_TRANSFORM_RES

        dataset = torchvision.datasets.ImageFolder(
            root=dataset_path, transform=data_transform)
        self.num_classes = len(dataset.classes)
        # データセットをトレーニングとテストに分割するindecesを作る
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        self.train_dataset_size = dataset_size - split
        self.test_dataset_size = split
        if SHUFFLE_DATASET:
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # sampler と loader を作る
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler)
        self.test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=valid_sampler)

    def adjust_learning_rate(epoch, base_lr, optimizer):
        if epoch > 180:
            base_lr = base_lr / 1000000
        elif epoch > 150:
            base_lr = base_lr / 100000
        elif epoch > 120:
            base_lr = base_lr / 10000
        elif epoch > 90:
            base_lr = base_lr / 1000
        elif epoch > 60:
            base_lr = base_lr / 100
        elif epoch > 30:
            base_lr = base_lr / 10

        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr

    def train(self, epoch, model, base_lr, loss_fn, name):
        if self.cuda_avail:
            model = model.cuda()
            loss_fn = loss_fn.cuda()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=base_lr,
            weight_decay=1e-4)
        test_loss = np.array([])
        test_acc = np.array([])
        best_acc = 0

        print("start train")

        for i in range(epoch):
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            iterations = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = Variable(images)
                labels = Variable(labels)
                if self.cuda_avail:
                    images = images.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()

                # images = images.float()
                # labels = labels.long()
                estimated = model(images)
                loss = loss_fn(estimated, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, prediction = torch.max(estimated, 1)
                train_acc += torch.sum(prediction ==
                                       labels.data)
                iterations += 1
            train_loss = train_loss / iterations
            train_acc = train_acc.item() / self.train_dataset_size
            print("epoch: {}, train_loss: {}".format(i, train_loss))
            print("train_acc: {:.3f}".format(train_acc))

            ModelTrainer.adjust_learning_rate(epoch, base_lr, optimizer)

            loss, acc = self.test(model, loss_fn)

            test_loss = np.append(test_loss, loss)
            test_acc = np.append(test_acc, acc)
            print("epoch: {}, test_loss: {}".format(i, test_loss[-1]))
            print("test_acc: {}".format(test_acc[-1]))
            if test_acc[-1] > best_acc:
                ModelTrainer.save_model(i, model, name)
                best_acc = test_acc[-1]

    def test(self, model, criterion):
        model.eval()
        test_loss = 0
        test_acc = 0
        iterations = 0
        for batch_idx, (images, labels) in enumerate(self.test_loader):
            if self.cuda_avail:
                images = images.cuda()
                labels = labels.cuda()
            images = Variable(images)
            labels = Variable(labels)
            with torch.no_grad():
                estimated = model(images)
            loss = criterion(estimated, labels)
            _, prediction = torch.max(estimated, 1)
            test_loss += loss.item()
            test_acc += torch.sum(prediction == labels.data)
            iterations += 1
        return test_loss / iterations, test_acc.item() / self.test_dataset_size

    def save_model(epoch, model, name):
        path = Path("./resources/models/")
        torch.save(model.state_dict(),
                   path.joinpath("{}_{}.model".format(name, epoch)))
