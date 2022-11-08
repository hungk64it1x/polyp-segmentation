import torch
import multiprocessing as mp
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP 
from utilizes.utils import setup_ddp
from utilizes.utils import clip_gradient, AvgMeter
import timeit
import os
import numpy as np
from auxiliary.metrics.metrics import *
import datetime

class Trainer:
    def __init__(self, model, model_name, optimizer, loss, scheduler, save_dir, save_from, logger, device, use_amp=False, use_ddp=False, multi_loss=False, name_writer=None):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler= scheduler
        self.save_from = save_from
        self.save_dir = save_dir
        self.logger = logger
        self.device = device
        self.use_amp = use_amp
        self.use_ddp = use_ddp
        self.name_writer = name_writer
        if self.name_writer == None:
            self.writer = SummaryWriter()
        else:
            save_wr = f'./runs/{self.name_writer}'
            self.writer = SummaryWriter(save_wr)
        
        self.scaler = GradScaler(enabled=self.use_amp)
        self.multi_loss = multi_loss
    
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def train_loop(self, train_loader, val_loader, num_epochs, img_size=352, size_rates=[1], clip_grad=0.5, is_val=False):
        start = timeit.default_timer()
        best_score = 0
        for epoch in range(num_epochs):
            self.model.train()
            if self.use_ddp:
                self.model = DDP(self.model, device_ids=[setup_ddp()])
            train_loss = 0.0
            
            total_iters = len(train_loader)
            pbar = tqdm(enumerate(train_loader), total=total_iters, 
                        desc=f"Epoch: [{epoch + 1}/{num_epochs}] | Iter: [{0}/{total_iters}] | LR: {self.get_lr() :.8f}")
            for iter, (images, gts) in pbar:
                for rate in size_rates:
                    self.optimizer.zero_grad()
                    images = images.to(self.device)
                    gts = gts.to(self.device)
                    
                    trainsize = int(round(img_size * rate / 32) * 32)
                    
                    if rate != 1:
                        images = torch.nn.functional.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        gts = torch.nn.functional.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)    

                    else:
                        images = torch.nn.functional.upsample(images, size=(img_size, img_size), mode='bilinear', align_corners=True)
                        gts = torch.nn.functional.upsample(gts, size=(img_size, img_size), mode='bilinear', align_corners=True)                           
                    # print(images.shape)
                    if self.multi_loss == False:
                        logits = self.model(images)
                        loss = self.loss(logits, gts)
                        
                        loss.backward()
                        clip_gradient(self.optimizer, clip_grad)
                        self.optimizer.step()  
                    elif self.multi_loss == True:
                        logits1, logits2, logits3, logits4 = self.model(images)
                        loss = self.loss(logits1, gts) + self.loss(logits2, gts) + self.loss(logits3, gts) + self.loss(logits4, gts)
                        
                        loss.backward()
                        clip_gradient(self.optimizer, clip_grad)
                        self.optimizer.step()

                    if self.use_amp: 
                        
                        with autocast(enabled=self.use_amp):
                            logits = self.model(images)
                            loss = self.loss(logits, gts)
                        self.scaler.scale(loss).backward()
                        clip_gradient(self.optimizer, clip_grad)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        torch.cuda.synchronize()
                        
                    if rate == 1:
                        train_loss += loss.item()
                        self.writer.add_scalar('train/loss', train_loss, epoch)
                    
                    pbar.set_description(f'Epoch: [{epoch + 1}/ {num_epochs}] | Iter: [{0}/{total_iters}] | LR: {self.get_lr():.6f}')
            train_loss /= iter + 1
                
            if is_val == True:
                score = self.val_loop(val_loader, epoch)
                if score > best_score:
                    best_score = score
                    torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    os.path.join(
                        self.save_dir, self.model_name + "_best.pth"
                    ),
                    )
                    self.logger.info(
                        "[Saving Snapshot:]"
                        + os.path.join(
                            self.save_dir, self.model_name + "_best.pth"
                        )
                    )    

            
            os.makedirs(self.save_dir, exist_ok=True)
            if is_val == False:
                self.logger.info(f'Epoch: [{epoch + 1}/ {num_epochs}] | Train loss: [{train_loss}]')
            # if epoch >= self.save_from and (epoch + 1) % 1 == 0 or epoch == 2:
            if epoch >= self.save_from and (epoch + 1) % 1 == 0 or epoch == 2:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    os.path.join(
                        self.save_dir, self.model_name + "_%d.pth" % (epoch + 1)
                    ),
                )
                self.logger.info(
                    "[Saving Snapshot:]"
                    + os.path.join(
                        self.save_dir, self.model_name + "_%d.pth" % (epoch + 1)
                    )
                )    
            self.scheduler.step(epoch)
        end = timeit.default_timer()

        self.logger.info("Training cost: " + str(end - start) + "seconds")
            
            
    def val_loop(self, val_loader, epoch):
        len_val = len(val_loader)

        mean_iou = 0
        mean_dice = 0
        mean_dice_neo = 0
        mean_dice_non = 0
        mean_iou_neo = 0
        mean_iou_non = 0


        val_loss = AvgMeter()
        polyp_metric = MetricForNeoplasm("polyp")
        neo_metric = MetricForNeoplasm("neo")
        non_metric = MetricForNeoplasm("non")
        for i, pack in enumerate(val_loader):
            
            image, gt, gt_resize = pack
            self.model.eval()

            gt_ = gt.cuda()

            gt = np.asarray(gt, np.float32).round()
            neo_gt = gt[:, 0,:,:]
            non_gt = gt[:, 1,:,:]
            polyp_gt = neo_gt + non_gt

            image = image.cuda()
            gt_resize = gt_resize.cuda()

            with torch.no_grad():
                result = self.model(image)
            res = torch.nn.functional.upsample(
                result, size=gt[0][0].shape, mode="bilinear", align_corners=False
            )

            loss2 = self.loss(res, gt_)
            res = res.softmax(dim=1)
            res = torch.argmax(
                res, dim=1, keepdims=True).squeeze().data.cpu().numpy()
            
            neo_pr = (res == 0).astype(np.float)
            non_pr = (res == 1).astype(np.float)

            # print(np.unique(non_pr))
            polyp_pr = neo_pr + non_pr
            # print(polyp_pr.shape)
            
            val_loss.update(loss2.data, 1)
            self.writer.add_scalar(
                "Val_loss", val_loss.show(), epoch * len(val_loader) + i
            )
            if i == len_val - 1:
                
                self.logger.info(
                    "Valid | Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}],\
                    [val_loss: {:.4f}]".format(
                        epoch,
                        epoch,
                        self.optimizer.param_groups[0]["lr"],
                        i,
                        val_loss.show(),
                    )
                )
            polyp_metric.cal(polyp_pr, polyp_gt)
            neo_metric.cal(neo_pr, neo_gt)
            non_metric.cal(non_pr, non_gt)

        #     mean_dice_neo += dice_m(neo_gt, neo_pr)
        #     mean_dice_non += dice_m(non_gt, non_pr)
            
        #     mean_iou_neo += jaccard_m(neo_gt, neo_pr)
        #     mean_iou_non += jaccard_m(non_gt, non_pr)
            
        #     mean_dice += dice_m(polyp_gt, polyp_pr)
        #     mean_iou += jaccard_m(polyp_gt, polyp_pr)

        # mean_iou /= len_val
        # mean_dice /= len_val
        # mean_dice_neo /= len_val
        # mean_dice_non /= len_val
        # mean_iou_neo /= len_val
        # mean_iou_non /= len_val

    #     self.logger.info(
    #     "Macro scores: Dice all: {:.3f} | IOU all: {:.3f} | Dice neo: {:.3f} | IOU neo: {:.3f} | Dice non: {:.3f} | IOU non: {:.3f}".format(
    #         mean_dice,
    #         mean_iou,
    #         mean_dice_neo,
    #         mean_iou_neo,
    #         mean_dice_non,
    #         mean_iou_non
    #     )
    # )
        dice_polyp, iou_polyp = polyp_metric.show()
        dice_neo, iou_neo = neo_metric.show()
        dice_non, iou_non = non_metric.show()

        score = (dice_polyp + dice_neo + dice_non) / 3
        return score

        
