from Utils.Registry import Registry

augmentations = Registry("Augmentations")


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for _cfg in self.cfg:
            self.transforms.append(augmentations.build(_cfg))

    def __call__(self, data, labels=None):
        if labels is not None:
            for t in self.transforms:
                data, labels = t(data, labels)
            return data, labels
        else:
            for t in self.transforms:
                data = t(data)
            return data


def build_augmentation(cfgs_augmentation):
    """
        Build an augmentation
        Args:
            cfgs_augmentation (eDICT):
        Returns:
            Augment: a constructed augmentation specified by augmentations.
    """
    return Compose(cfgs_augmentation)

