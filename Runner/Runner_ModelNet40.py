import sys
import torch

from tqdm import tqdm
from Criteria.Builid_Criteria import build_criteria
from Datasets.Build_Dataloader import build_dataloader
from Models.Build_Model import build_model
from Optimizer.Build_Optimizer import build_optimizer
from Scheduler.Build_Scheduler import build_scheduler
from Utils.Logger import get_logger, print_log
from Utils.Misc import summary_parameters


def train_net(cfgs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = get_logger(cfgs.common.log_name)

    # build dataset
    train_dataloader, test_dataloader = build_dataloader(cfgs.dataloader.train), build_dataloader(cfgs.dataloader.test)

    # build model
    base_model = build_model(cfgs.model).to(device)
    summary_parameters(base_model, logger=logger)
    base_model.zero_grad()

    # build optimizer and scheduler
    optimizer = build_optimizer(cfgs.optimizer, base_model, cfgs.optimizer.pop('param_dicts'))
    if cfgs.scheduler.NAME == 'OneCycleLR':
        cfgs.scheduler.total_steps = len(train_dataloader) * cfgs.common.epoch
    scheduler = build_scheduler(cfgs.scheduler, optimizer)

    # build loss function
    loss_function = build_criteria(cfgs.criteria)

    # record setting
    start_epoch = 0
    best_acc = 0.0
    best_mAcc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, cfgs.common.epoch):
        num_iter = 0

        # set model to training mode
        base_model.train()
        optimizer.zero_grad()

        # accumulating loss, correct sample number, and total sample number
        accu_loss, accu_num, sample_num = 0.0, 0, 0

        train_dataloader = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(train_dataloader):
            num_iter += 1
            points, label = data[0].to(device), data[1].to(device)
            logits = base_model(points)

            sample_num += points.shape[0]
            ret_cls = torch.max(logits, dim=1)[1]
            accu_num += torch.eq(ret_cls, label).sum()

            loss = loss_function(logits, label.long())
            loss.backward()
            accu_loss += loss.detach()

            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss)
                sys.exit(1)

            train_dataloader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch, accu_loss / (step + 1), accu_num / float(sample_num))

            if num_iter == cfgs.common.step_per_update:
                if cfgs.common.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), cfgs.common.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

        train_loss, train_acc = accu_loss / (step + 1), accu_num / sample_num

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        if epoch % cfgs.common.val_freq == 0 and epoch != 0:
            # Validate the current model
            val_loss, val_acc, val_mACC = val_net(base_model, test_dataloader, epoch, cfgs, loss_function, device, logger=logger)

            print_log("epoch %d train_loss: %.4f, train_acc: %.4f, val_loss: %.4f, val_acc: %.4f, val_mACC: %.4f, lr: %f" % (
                epoch, train_loss, train_acc, val_loss, val_acc, val_mACC, optimizer.param_groups[0]['lr']), logger=logger)

            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
                print_log("Saving ModelNet40...", logger=logger)
                # Save checkpoints
                torch.save(base_model.state_dict(), cfgs.common.experiment_dir + "/best_model.pth")
            if val_mACC >= best_mAcc:
                best_mAcc = val_mACC

    print_log(f"best_acc: %.4f, best_mAcc: %.4f, best_epoch: %d" % (best_acc, best_mAcc, best_epoch), logger=logger)
    print_log("End of training...", logger=logger)


def val_net(base_model, test_dataloader, epoch, cfgs, loss_function, device, logger=None):
    num_category = cfgs.model.num_classes

    # set model to eval mode
    base_model.eval()

    accu_loss, accu_num, sample_num = 0.0, 0, 0

    # accumulate the number of samples for each category
    num_per_class = torch.as_tensor([0 for _ in range(num_category)], dtype=torch.float32).to(device)
    # accumulate the number of correctly classified samples for each category
    right_num_per_class = torch.as_tensor([0 for _ in range(num_category)], dtype=torch.float32).to(device)

    test_dataloader = tqdm(test_dataloader, file=sys.stdout)
    with torch.no_grad():
        for step, data in enumerate(test_dataloader):
            points, label = data[0].cuda(), data[1].cuda()
            logits = base_model(points)

            sample_num += points.shape[0]
            ret_cls = torch.max(logits, dim=1)[1]
            accu_num += torch.eq(ret_cls, label).sum()

            num_per_class += torch.bincount(label, minlength=num_category)
            right_num_per_class += torch.bincount(label, weights=label == ret_cls, minlength=num_category)

            loss = loss_function(logits, label.long())
            accu_loss += loss

            test_dataloader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch, accu_loss / (step + 1), accu_num / float(sample_num))

        mAcc = right_num_per_class / num_per_class
        right_rate_per_class = '  '.join([str(x) for x in torch.round(mAcc, decimals=4).cpu().numpy()])
        num_per_class = '  '.join([str(x) for x in torch.round(num_per_class, decimals=4).cpu().numpy()])
        right_num_per_class = '  '.join([str(x) for x in torch.round(right_num_per_class, decimals=4).cpu().numpy()])

        print_log(f"epoch {epoch}\n  每个类的参与总数：{num_per_class}\n  每个类正确分类数：{right_num_per_class}\n  每个类的正确比例：{right_rate_per_class}", logger=logger)

    return accu_loss / (step + 1), accu_num / float(sample_num), torch.round(torch.mean(mAcc), decimals=4).cpu().numpy()
