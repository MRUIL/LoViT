import torch.nn as nn
import timm
class FrameLevelModel(nn.Module):
    """Runs a frame level model on all the frames."""
    def __init__(self, num_classes: int, model: nn.Module = None):
        del num_classes
        super().__init__()
        self.model = model
class TIMMModel(FrameLevelModel):
    def __init__(self,
                 num_classes,
                 model_type='vit_base_patch16_224',
                 drop_cls=True):
        super().__init__(num_classes)
        model = timm.create_model(model_type,
                                  num_classes=0 if drop_cls else num_classes)
        self.model = model
