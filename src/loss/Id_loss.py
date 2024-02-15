import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class classifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(classifier, self).__init__()

        self.block = nn.Linear(input_dim, output_dim)
        self.block.apply(weights_init_classifier)

    def forward(self, x):
        x = self.block(x)
        return x

class Id_Loss(nn.Module):

    def __init__(self, opt, feature_length):
        super(Id_Loss, self).__init__()

        self.opt = opt
        self.classifier = classifier(feature_length, opt.class_num)

    def calculate_IdLoss(self, image_embedding, text_embedding, label):

        label = label.view(label.size(0))

        criterion = nn.CrossEntropyLoss(reduction='mean')

        score_i2t_local_i = self.classifier(image_embedding)
        score_t2i_local_i = self.classifier(text_embedding)

        Lipt_local = criterion(score_i2t_local_i, label)
        Ltpi_local = criterion(score_t2i_local_i, label)

        loss = Lipt_local + Ltpi_local

        return loss

    def forward(self, image_embedding, text_embedding, label):

        loss = self.calculate_IdLoss(image_embedding, text_embedding, label)

        return loss