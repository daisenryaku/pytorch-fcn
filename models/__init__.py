import torchvision.models as models
from fcn import FCN

def get_model(n_classes=21):
    model = FCN(n_classes=n_classes)
    vgg16 = models.vgg16(pretrained=True)
    model.init_vgg16_params(vgg16)
    return model
