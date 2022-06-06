import argparse
import gc
import os
import pickle
import random
import time
from os import cpu_count

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import metrics as module_metric
from torch.optim import Adam

from base import BaseModel, BaseTrainer
from data_prepare import get_dataloaders
from decoder import MLPDecoder
from dual_loss import RegLoss
from encoder_temporal import MultiChannelWave
from parse_config import ConfigParser
from pretrain import GraphDataset, GAT
from utils import inf_loop, MetricTracker, to_device, data_folder, get_device


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, metric_name, optimizer, config, data_loader,
                 test_data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, metric_name, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader

        self.lr_scheduler = lr_scheduler

        self.time_slice = time_slice
        self.train_metrics = MetricTracker('loss', *[f"{self.time_slice}_" + m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[f"{self.time_slice}_" + m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)
        metrics_weights = config["metrics_weights"]
        self.mix_weights = {f"{self.time_slice}_MSE": metrics_weights[0], f"{self.time_slice}_MAE": metrics_weights[1]}

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        total_batch_num = len(self.data_loader)
        p_data_loader = tqdm(self.data_loader, total=total_batch_num)
        for batch_idx, batch in enumerate(p_data_loader):
            uid, x, _, y = to_device(batch)
            self.optimizer.zero_grad()
            step_idx = (epoch - 1) * self.len_epoch + batch_idx
            y_hat, dual_loss = self.model(uid, x, step_idx)
            loss_dict = self.criterion(y_hat, y, self.model, dual_loss, self.config)
            loss = loss_dict["total"]
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            self.writer.set_step(step_idx)
            self._log_para_update(batch_idx, step_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(f"{self.time_slice}_" + met.__name__, met(y_hat, y))
            for n, v in loss_dict.items():
                self.writer.add_scalar(f"train_loss/{n}", v.item(), step_idx)
            p_data_loader.set_description(f"Train epoch {epoch}: loss: {loss.item()}")
            del loss
            gc.collect()
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result(self.mix_weights, f"{self.time_slice}_mix")

        if self.do_validation:
            logger.info("Start evaluation...")
            val_log = self._valid_epoch(epoch, self.valid_data_loader, "val")
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        logger.info("Start test...")
        test_log = self._valid_epoch(epoch, self.test_data_loader, "test")
        log.update(**{'test_' + k: v for k, v in test_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_log[f"{self.time_slice}_MSE"])
        return log

    def _valid_epoch(self, epoch, data_loader, val_type):
        self.model.eval()
        self.valid_metrics.reset()
        uids = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            total_batch_num = len(data_loader)
            p_data_loader = tqdm(data_loader, total=total_batch_num)
            for batch_idx, batch in enumerate(p_data_loader):
                uid, x, _, y = to_device(batch)
                step_idx = (epoch - 1) * self.len_epoch + batch_idx
                y_hat, dual_loss = self.model(uid, x, None)
                loss_dict = self.criterion(y_hat, y, self.model, dual_loss, self.config)
                loss = loss_dict["total"]
                if "test" == val_type:
                    uids.extend(uid.cpu().detach().numpy())
                    y_pred.extend(y_hat.view(1, -1).cpu().detach().numpy())
                    y_true.extend(y.view(1, -1).cpu().detach().numpy())
                self.writer.set_step(step_idx, val_type)
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(f"{self.time_slice}_" + met.__name__, met(y_hat, y))
                for n, v in loss_dict.items():
                    self.writer.add_scalar(f"{val_type}_loss/{n}", v.item(), step_idx)
                p_data_loader.set_description(f"{val_type} epoch {epoch}: {val_type}_loss: {loss.item()}")
                del loss
                gc.collect()

        return self.valid_metrics.result(self.mix_weights, f"{self.time_slice}_mix")

    def _log_para_update(self, batch_idx, step_idx):
        if batch_idx % 10 == 0:
            param_updates = {name: param.detach().cpu().clone() for name, param in self.model.named_parameters()}
            self.optimizer.step()
            # for name, param in self.model.named_parameters():
            #     if "encoder_spatial" in name:
            #         print(name, param.grad)
            for name, param in self.model.named_parameters():
                param_updates[name].sub_(param.detach().cpu())
                update_norm = torch.norm(param_updates[name].view(-1, ))
                param_norm = torch.norm(param.view(-1, )).cpu()
                self.writer.add_scalar("gradient_update/" + name, update_norm / (param_norm + 1e-7), step_idx)
        else:
            self.optimizer.step()


def criterion(y_hat, y, model, dual_loss, config):
    # wavelet bases constrains
    wave_loss = config["wavelet_loss_weight"] * model.encoder_temporal.wavelets_losses()
    ltv_loss = F.smooth_l1_loss(y_hat, y.squeeze())
    l1_regularization, l2_regularization = 0, 0
    for name, param in model.named_parameters():
        if "bias" not in name:
            l1_regularization += torch.norm(param, 1)
            l2_regularization += torch.norm(param, 2)
    reg_loss = config["l1_weight"] * l1_regularization + config["l2_weight"] * l2_regularization
    total = wave_loss + ltv_loss
    # print(loss)
    # cluster alignment loss
    d_loss = torch.tensor([0]).to(get_device())
    if dual_loss is not None:
        d_loss = config["dual_loss_weight"] * dual_loss
        total += d_loss
    # print(loss)
    # print("____________________________________________________________________________")
    return {
        "wave_loss": wave_loss,
        "ltv_loss": ltv_loss,
        "d_loss": d_loss,
        "total": total
    }


class STPModel(BaseModel):
    def __init__(self, config):
        super(STPModel, self).__init__()
        self.time_slice = config["time_slice"]
        self.use_gm = config["use_gm"]
        self.use_dual_loss = config["use_dual_loss"]

        wave_levels = 3
        channels = 50
        hid_dim = 100
        self.encoder_temporal = MultiChannelWave(channels=channels, filter_len=10, levels=wave_levels, seq_len=30,
                                                 hid_dim=hid_dim, out_dim=config["se_dim"], rnn_type=config["rnn_type"])

        if self.use_gm:
            graph = GraphDataset(data_name=config["data_name"],
                                 limit_num=-1,
                                 k=11,
                                 save_dir=os.path.join(data_folder, "graphs"))[0]
            graph = graph.to(get_device())
            num_layers = 3
            heads = ([4] * num_layers) + [4]
            self.node_features = graph.ndata['feat']
            self.encoder_spatial = GAT(graph, num_layers=num_layers,
                                       in_dim=self.node_features.shape[1],
                                       num_hidden=hid_dim,
                                       out_dim=config["se_dim"],
                                       heads=heads,
                                       activation=None,
                                       feat_drop=0.1,
                                       attn_drop=0.1,
                                       negative_slope=0.2,
                                       residual=False)
        if self.use_dual_loss:
            self.dual_loss = RegLoss(te_dim=config["te_dim"], se_dim=config["se_dim"], k=config["cluster_k"], k_dim=50)

        self.decoder = MLPDecoder(hid_dim, te_dim=config["te_dim"], se_dim=config["se_dim"])

    def forward(self, uid, x, step_idx):
        t1 = time.time()
        if self.use_gm:
            xnext = self.encoder_spatial(self.node_features)
        t2 = time.time()
        coeffs, rnn_outs, hiddens = self.encoder_temporal(x)
        t3 = time.time()
        # print(f"low-f: {coeffs[-1].shape}, rnn_hid: {hiddens[-1].shape}")
        dual_loss = None
        if self.use_dual_loss and self.use_gm:
            dual_loss = self.dual_loss(hiddens[-1], xnext[uid]) if step_idx and (step_idx + 1) % config[
                "update_interval"] == 0 else None
        t4 = time.time()
        y = self.decoder(torch.sum(x, -1), hiddens, xnext[uid] if self.use_gm else None)
        t5 = time.time()
        # print(f"s time: {t2 - t1}, t time: {t3 - t2}, dec time: {t5 - t4}")
        return y, dual_loss


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='LTV')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')

    torch.autograd.set_detect_anomaly(True)

    # fix random seeds for reproducibility
    SEED = 2021
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    config = ConfigParser.from_args(args)
    logger = config.get_logger('train')

    debug = False
    time_slice = config["time_slice"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"] if not debug else 2
    num_workers = cpu_count() if not debug else 1

    model = STPModel(config)
    logger.info(model)

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True,
    #                                      profile_memory=False) as prof:
    #     inp = torch.randn(batch_size, 30)
    #     outputs = model(None, inp, None)
    # print(prof.table())
    # prof.export_chrome_trace('./resnet_profile.json')

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=learning_rate)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    data_name = config["data_name"]
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_name, batch_size=batch_size,
                                                                        num_workers=num_workers,
                                                                        pin_memory=True if not debug else False,
                                                                        future_len=time_slice,
                                                                        limit_num=config["limit_num"],
                                                                        train_ratio=config["train_ratio"],
                                                                        active_days=config["active_days"])

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainer = Trainer(model, criterion, metrics, f"val_{time_slice}_mix", optimizer, config, train_dataloader,
                      test_data_loader=test_dataloader,
                      valid_data_loader=val_dataloader, lr_scheduler=lr_scheduler, len_epoch=None)
    trainer.train()
