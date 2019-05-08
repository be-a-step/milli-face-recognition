import torch.nn as nn


class IRModel(nn.Module):
    def __init__(self):
        super(IRModel, self).__init__()
        self.output_num = 10
        self.filer_num = 64
        self.act = nn.LeakyReLU(0.2)
        self.conv1 = self.conv_block(
            1,
            self.filer_num * 1,
            kernel_size=15,
            stride=3,
            padding=0,
            act_fn=self.act)
        self.conv2 = self.conv_block(
            self.filer_num * 1,
            self.filer_num * 2,
            kernel_size=3,
            stride=1,
            padding=0,
            act_fn=self.act)
        self.conv3 = self.conv_block(
            self.filer_num * 2,
            self.filer_num * 4,
            kernel_size=3,
            stride=1,
            padding=1,
            act_fn=self.act)
        self.conv4 = self.conv_block(
            self.filer_num * 4,
            self.filer_num * 4,
            kernel_size=3,
            stride=1,
            padding=1,
            act_fn=self.act)
        self.conv5 = self.conv_block(
            self.filer_num * 4,
            self.filer_num * 4,
            kernel_size=3,
            stride=1,
            padding=1,
            act_fn=self.act)
        self.conv6 = self.conv_block(
            self.filer_num * 4,
            self.filer_num * 8,
            kernel_size=7,
            stride=1,
            padding=0,
            act_fn=self.act)
        self.conv7 = self.conv_block(
            self.filer_num * 8,
            self.filer_num * 8,
            kernel_size=1,
            stride=1,
            padding=0,
            act_fn=self.act)
        self.conv8 = self.conv_block(
            self.filer_num * 8,
            50,
            kernel_size=1,
            stride=1,
            padding=0,
            act_fn=self.act)
        self.out = nn.Sequential(
            nn.Linear(50, self.output_num),
            nn.Sigmoid(),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.drop = nn.Dropout2d(p=0.25)

        self.res_block = nn.Sequential(
            self.conv_block(
                self.filer_num * 2,
                self.filer_num * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                act_fn=self.act),
            self.conv_block(
                self.filer_num * 4,
                self.filer_num * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                act_fn=self.act),
            nn.Conv2d(
                self.filer_num * 4,
                self.filer_num * 4,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(
                self.filer_num * 4),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(
                self.filer_num * 2,
                self.filer_num * 4,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(
                self.filer_num * 4),
        )

    def forward(self, input):
        h = self.pool(self.conv1(input))
        h = self.pool(self.conv2(h))
        h1 = self.residual(h)
        h = self.pool(self.act(self.res_block(h) + h1))
        h = self.drop(self.conv6(h))
        h = self.conv8(self.drop(self.conv7(h)))
        h = self.out(self.flatten(h))
        return h

    def conv_block(
            self,
            in_dim,
            out_dim,
            kernel_size,
            stride,
            padding,
            act_fn):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )

    def flatten(self, x):
        bs = x.size()[0]
        return x.view(bs, -1)
