from Utils.Registry import Registry

models = Registry('Models')


def build_model(cfgs, default_args=None):
    """
        Build a model, defined by `model_name`.
        Args:
            cfgs :
            default_args :
        Returns:
            Model: a constructed model specified by model_name.
    """
    return models.build(cfgs, default_args=default_args)
