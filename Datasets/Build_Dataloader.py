from Utils.Registry import Registry
from Utils.Misc import worker_init_fn
from torch.utils.data import DataLoader

datasets = Registry("Datasets")


def build_dataset(cfgs, default_args=None):
    """
        Build a dataset, defined by `dataset_name`.
        Args:
            cfgs :
            default_args :
        Returns:
            Dataset: a constructed dataset specified by dataset_name.
    """
    return datasets.build(cfgs, default_args=default_args)


def build_dataloader(cfgs):

    return DataLoader(
        build_dataset(cfgs.dataset),
        batch_size=cfgs.batch_size,
        num_workers=int(cfgs.num_workers),
        shuffle=cfgs.dataset.mode != "test",
        drop_last=cfgs.dataset.mode != "test",
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        # multiprocessing_context='spawn',
        # persistent_workers=True
    )
