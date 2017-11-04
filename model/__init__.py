from loss import cross_entropy2d
from fcn import fcn8s
import torchvision.models as models

def get_model(name, n_classes):
    if name == 'fcn8s':
        model = fcn8s(n_classes = n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    else:
        raise 'Model {} noe available'.format(name)
    return model
