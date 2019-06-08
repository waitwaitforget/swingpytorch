from __future__ import absolute_import

import os
import sys
sys.path.append('../../..')
import json
import time
import numpy as np
import torch as t
from easydict import EasyDict as edict
from logging import getLogger

from torch.nn import functional as F
#from logging import getLogger
from misc import Bar
import tools as tools
from tools import AverageMeter
# from utils import AverageMeter, TargetMapping, metrics, get_optimizer
import copy

"""
TODO list
- add log module [Done]
- add tensorboard visualization

"""
logger = getLogger()

class Engine(object):
    def __init__(self, model, config, metrics=None, **kwargs):
        """
        Engine initialization.
        Support self-defined optimizer/ scheduler.
        Args:
            model: models that need to be optmized
            config: config obj
            metrics: name and eval func of measures which is organized in a dict
            kwargs: future extension (callbacks)
        """
        self.config = config

        # use cuda
        self.model = model.cuda() if config.cuda else model
        
        # set default classification criterion
        if not hasattr(model, 'loss_function'):
            print('Register cross entropy loss as default loss function.')
            model.loss_function = nn.CrossEntropyLoss()

        # optimizer / scheduler
        # TODO add customizatio for optimizer and scheduler
        self.optimizer = t.optim.SGD(model.parameters(), lr=config.learning_rate,
                                         momentum=config.momentum, 
                                         nesterov=config.nesterov, 
                                         weight_decay=config.weight_decay)
        self.scheduler = t.optim.lr_scheduler.MultiStepLR(
                                        self.optimizer, 
                                        milestones=config.milestones, 
                                        gamma=config.gamma)


        logger.info('%i parameters in the model.' % sum([p.nelement() for p in self.model.parameters()]))

        # training statistics
        self.stats = edict()
        self.stats.train_epoch_loss = []
        self.stats.eval_epoch_loss = []
        self.stats.batch_loss = []
        self.measures = edict()
        self.measures.train = edict()
        
        if metrics is None:
            metrics = {'top1_acc': accuracy}
            self.measures.train.top1_acc = []
            self.measures.val.top1_acc = []
        else:
            for key in metrics.keys():
                self.measures.train[key] = []
                self.measures.val[key] = []
        
        self.metrics_func = metrics
        # best accuracy
        self.best_measures = [0.] if metrics is None else [0.] * len(metrics)
        self.cur_epoch = 0

        # kwargs
        state_dict = self._state_dict()
        for k,v in kwargs.items():
            if k not in state_dict:
                setattr(self, k, v)

    def _state_dict(self):
        return {k: getattr(self,k) for k,_ in self.__dict__.items() if not k.startswith('__')}

    def train_step(self, epoch, trainloader):
        """Training step for each epoch.
        Args:
         epoch: current epoch
         trainloader: dataloader for train set
        Return:
         None
        """
        self.model.train()
        epoch_loss = AverageMeter()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        metrics_meter = dict()
        for k in self.measures.train.keys():
            metrics_meter[k] = AverageMeter()

        bar = Bar('Processing', max=len(trainloader))
        end = time.time()

        for batch_idx, (data, targets) in enumerate(trainloader):
            
            data_time.update(time.time() - end)
            if self.config.cuda:
                data, targets = data.cuda(), targets.cuda()

            preds = self.model(data)

            loss = self.model.loss_function(preds, targets)
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # calculate measure statistics
            batch_measure = dict()
            for k, func in self.metrics_func.items():
                if k.startswith('top'):
                    batch_measure[k] = func(preds, targets)[0]
                else:
                    batch_measure[k] = func(preds, targets)
                if isinstance(batch_measure[k], t.autograd.Variable):
                    batch_measure[k] = batch_measure[k].item()
                metrics_meter[k].update(batch_measure[k], data.size(0))

            # record statistics
            self.stats.batch_loss.append(loss.item())
            epoch_loss.update(loss.item())
 
            batch_time.update(time.time() - end)
            end = time.time()

            #plot progress
            measure_bar = ' | '.join(['%s : %.4f'%(k, v.avg) for k,v in metrics_meter.items()])
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=epoch_loss.avg) + measure_bar
            bar.next()
           
        bar.finish()
        for k in metrics_meter.keys():
            self.measures.val[k] = metrics_meter[k].avg
        # plot on tensorboard
        '''
        for k, v in metrics_meter.items():
            self.metrics[k].append(v.avg)
            log_value('train %s' % k, v.avg, epoch)
        '''
        self.stats.train_epoch_loss.append(epoch_loss.avg)
        # log_value('epoch_loss', epoch_loss.avg, epoch)

        logger.info(('%02i - ' % (epoch + 1)) + ' / '.join(['train loss %.5f' % epoch_loss.avg] + [k + ' %.5f' % v.avg for k, v in metrics_meter.items()]))

    def validate(self, epoch, valdataloader):
        self.model.eval()
        
        epoch_loss = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        metrics_meter = dict()
        for k in self.measures.val.keys():
            metrics_meter[k] = AverageMeter()

        end = time.time()
        bar = Bar('Processing', max=len(valdataloader))
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(valdataloader):
                data_time.update(time.time() - end)
                if self.config.cuda:
                    data, targets = data.cuda(), targets.cuda()
                    
                preds = self.model(data)
                loss = self.model.loss_function(preds, targets)

                # calculate measure statistics
                batch_measure = dict()
                for k, func in self.metrics_func.items():
                    if k.startswith('top'):
                        batch_measure[k] = func(preds, targets)[0]
                    else:
                        batch_measure[k] = func(preds, targets)
                    if isinstance(batch_measure[k], t.autograd.Variable):
                        batch_measure[k] = batch_measure[k].item()
                    metrics_meter[k].update(batch_measure[k], data.size(0))
                self.stats.batch_loss.append(loss.item())
                epoch_loss.update(loss.item())
    

                batch_time.update(time.time() - end)
                end = time.time()

                #plot progress
                measure_bar = ' | '.join(['%s : %.4f'%(k,v.avg) for k,v in metrics_meter.items()])
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | '.format(
                        batch=batch_idx + 1,
                        size=len(valdataloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=epoch_loss.avg) + measure_bar
                bar.next()
           
        bar.finish()
        for k in metrics_meter.keys():
            self.measures.val[k] = metrics_meter[k]

        self.stats.eval_epoch_loss.append(epoch_loss.avg)
        #log_value('val loss', epoch_loss.avg, epoch)
        #for k, v in metrics_meter.items():
        #    self.metrics[k].append(v.avg)
        #    log_value('val %s' % k, v.avg, epoch)

       
        to_log = dict([('epoch', epoch)] + [(k, v.avg) for k, v in metrics_meter.items()])

        logger.debug("__log__:%s" % json.dumps(to_log))

        return to_log

    def save_best_checkpoint(self, to_log):
        """
        Save the best models / periodically save the models.
        """
        if to_log['top1_acc'] > self.best_measures[0]:
            self.best_measures[0] = to_log['top1_acc']
            logger.info('Best top1 acc : %.5f' % self.best_measures[0])
            self.save_model('best_top1')

    def save_model(self, name):
        """
        Save the model.
        """
        def save(model, filename):
            path = os.path.join(self.config.checkpoint, '%s_%s.pth' % (name, filename))
            logger.info('Saving %s to %s ...' % (filename, path))
            t.save(model, path)
        save(self.model, 'model')

    def start(self, trainloader, valloader):
        print('start training %s on %s'%(self.model.__class__, self.config.dataset))
        for epoch in range(self.config.max_epoch):
            logger.info('Starting epoch %i...' % (epoch+1))
            print('\n[Epoch %d/%d]'%(epoch+1, self.config.max_epoch))
            self.scheduler.step()
            self.train_step(epoch, trainloader)
            to_log = self.validate(epoch, valloader)

            self.save_best_checkpoint(to_log)
            # save checkpoints
            if (epoch + 1) % self.config.ckpt_interval == 0:
                self.save_model('ckpt_%i' % (epoch))
        print('finish training')
        print('best accuracy: %.4f'%self.best_measures[0])