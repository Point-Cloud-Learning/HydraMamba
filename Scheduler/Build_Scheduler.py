from Utils.Registry import Registry

schedulers = Registry("Schedulers")


def build_scheduler(cfg, optimizer):
    cfg.optimizer = optimizer

    return schedulers.build(cfg)
