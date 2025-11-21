import math
import sys
from typing import Iterable

import torch
import util.misc as misc
import util.lr_sched as lr_sched
from llama.llama_adapter import LLaMA_adapter  # ✅ correct import

def train_one_epoch(model: LLaMA_adapter,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir:', log_writer.log_dir)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # unpack batch (Radar + Image + Text)
        imgs = batch["image"].to(device, non_blocking=True)
        radars = batch["radar"].to(device, non_blocking=True)
        examples = batch["tokens"].to(device)   # input text tokens
        labels = batch["labels"].to(device)     # shifted text targets

        # adjust learning rate per iteration
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        # forward + loss
        with torch.cuda.amp.autocast():
            c_loss, m_loss = model(examples, labels, imgs, radar=radars)
            loss = c_loss + m_loss * 0.0  # m_loss unused but kept for structure

        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = m_loss.item() if torch.is_tensor(m_loss) else 0.0

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # tensorboard logging
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value, epoch_1000x)
            log_writer.add_scalar('m_train_loss', m_loss_value, epoch_1000x)
            log_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch_1000x)

    # gather stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
