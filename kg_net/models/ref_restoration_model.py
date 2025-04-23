import importlib
import logging
import os.path as osp
from collections import OrderedDict
import scipy.io as sio

import mmcv
import torch

import kg_net.models.networks as networks
import kg_net.utils.metrics as metrics
import kg_net.utils.PSNRSSIMloss as PSNRSSIMloss
from kg_net.utils import ProgressBar, tensor2img, img2tensor
from kg_net.data.util import complex_abs_eval

from .sr_model import SRModel
import pdb

loss_module = importlib.import_module('kgmgt.models.losses')
logger = logging.getLogger('base')
psnr_list = []


class RefRestorationModel(SRModel):

    def __init__(self, opt):
        super(RefRestorationModel, self).__init__(opt)

        # net_map does not have any trainable parameters.
        self.net_map = networks.define_net_map(opt)
        self.net_map = self.model_to_device(self.net_map)

        # define network for feature extraction/
        self.net_extractor = networks.define_net_extractor(opt)
        self.net_extractor = self.model_to_device(self.net_extractor)
        self.print_network(self.net_extractor)

        # load pretrained feature extractor
        load_path = self.opt['path'].get('pretrain_model_feature_extractor',
                                         None)
        if load_path is not None:
            self.load_network(self.net_extractor, load_path,
                              self.opt['path']['strict_load'])

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])
        if self.is_train:
            self.net_g.train()

            # optimizers
            train_opt = self.opt['train']
            weight_decay_g = train_opt.get('weight_decay_g', 0)
            optim_params_g = []
            optim_params_offset = []
            optim_params_relu2_offset = []
            optim_params_relu3_offset = []
            if train_opt.get('lr_relu3_offset', None):
                optim_params_relu3_offset = []
            for name, v in self.net_g.named_parameters():
                if v.requires_grad:
                    if 'offset' in name:
                        if 'small' in name:
                            logger.info(name)
                            optim_params_relu3_offset.append(v)
                        elif 'medium' in name:
                            logger.info(name)
                            optim_params_relu2_offset.append(v)
                        else:
                            optim_params_offset.append(v)
                    else:
                        optim_params_g.append(v)

            self.optimizer_g = torch.optim.Adam(
                [{
                    'params': optim_params_g
                }, {
                    'params': optim_params_offset,
                    'lr': train_opt['lr_offset']
                }, {
                    'params': optim_params_relu3_offset,
                    'lr': train_opt['lr_relu3_offset']
                }, {
                    'params': optim_params_relu2_offset,
                    'lr': train_opt['lr_relu2_offset']
                }],
                lr=train_opt['lr_g'],
                weight_decay=weight_decay_g,
                betas=train_opt['beta_g'])

            self.optimizers.append(self.optimizer_g)

    def init_training_settings(self):
        train_opt = self.opt['train']

        if self.opt.get('network_d', None):
            # define network net_d
            self.net_d = networks.define_net_d(self.opt)
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
            # load pretrained models
            load_path = self.opt['path'].get('pretrain_model_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path,
                                  self.opt['path']['strict_load'])
        else:
            logger.info('No discriminator.')
            self.net_d = None

        if self.net_d:
            self.net_d.train()

        # define losses
        if train_opt['pixel_weight'] > 0:
            cri_pix_cls = getattr(loss_module, train_opt['pixel_criterion'])
            self.cri_pix = cri_pix_cls(
                loss_weight=train_opt['pixel_weight'],
                reduction='mean').to(self.device)
        else:
            logger.info('Remove pixel loss.')
            self.cri_pix = None

        if train_opt.get('perceptual_opt', None):
            cri_perceptual_cls = getattr(loss_module, 'PerceptualLoss')
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            logger.info('Remove perceptual loss.')
            self.cri_perceptual = None

        if train_opt.get('style_opt', None):
            cri_style_cls = getattr(loss_module, 'PerceptualLoss')
            self.cri_style = cri_style_cls(**train_opt['style_opt']).to(
                self.device)
        else:
            logger.info('Remove style loss.')
            self.cri_style = None

        if train_opt.get('texture_opt', None):
            cri_texture_cls = getattr(loss_module, 'TextureLoss')
            self.cri_texture = cri_texture_cls(**train_opt['texture_opt']).to(
                self.device)
        else:
            logger.info('Remove texture loss.')
            self.cri_texture = None

        if train_opt.get('gan_type', None):
            cri_gan_cls = getattr(loss_module, 'GANLoss')
            self.cri_gan = cri_gan_cls(
                train_opt['gan_type'],
                real_label_val=1.0,
                fake_label_val=0.0,
                loss_weight=train_opt['gan_weight']).to(self.device)

            if train_opt['grad_penalty_weight'] > 0:
                cri_grad_penalty_cls = getattr(loss_module,
                                               'GradientPenaltyLoss')
                self.cri_grad_penalty = cri_grad_penalty_cls(
                    loss_weight=train_opt['grad_penalty_weight']).to(
                        self.device)
            else:
                logger.info('Remove gradient penalty.')
                self.cri_grad_penalty = None
        else:
            logger.info('Remove GAN loss.')
            self.cri_gan = None

        # we need to train the net_g with only pixel loss for several steps
        self.net_g_pretrain_steps = train_opt['net_g_pretrain_steps']
        self.net_d_steps = train_opt['net_d_steps'] if train_opt[
            'net_d_steps'] else 1
        self.net_d_init_steps = train_opt['net_d_init_steps'] if train_opt[
            'net_d_init_steps'] else 0

        # optimizers
        if self.net_d:
            weight_decay_d = train_opt.get('weight_decay_d', 0)
            self.optimizer_d = torch.optim.Adam(
                self.net_d.parameters(),
                lr=train_opt['lr_d'],
                weight_decay=weight_decay_d,
                betas=train_opt['beta_d'])
            self.optimizers.append(self.optimizer_d)

        # check the schedulers
        self.setup_schedulers()

        self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.img_in_lq = data['img_in_lq'].to(self.device)
        self.img_ref1 = data['img_ref1'].to(self.device)
        self.img_ref2 = data['img_ref2'].to(self.device)
        self.gt = data['img_ref_gt'].to(self.device)
        self.match_img_in = data['img_in_up'].to(self.device)
        self.dc_mask256 = data['dc_mask256'].to(self.device)
        if 'img_in_ori' in data:
            self.gt_ori = data['img_in_ori'].to(self.device)


    def optimize_parameters(self, step):

        self.features1 = self.net_extractor(self.match_img_in, self.img_ref1)
        self.pre_offset1, self.img_ref_feat1 = self.net_map(
            self.features1, self.img_ref1)
        self.output1, self.k_d = self.net_g(self.gt, self.dc_mask256, self.img_in_lq, self.img_ref_feat1, self.pre_offset1, self.img_ref_feat1)

        self.features2 = self.net_extractor(self.output1, self.img_ref2)
        self.pre_offset2, self.img_ref_feat2 = self.net_map(
            self.features2, self.img_ref2)
        self.output, _ = self.net_g(self.gt, self.dc_mask256, self.output1, self.k_d, self.pre_offset2,
                                  self.img_ref_feat2)

        if step <= self.net_g_pretrain_steps:
            # pretrain the net_g with pixel Loss
            self.optimizer_g.zero_grad()
            l_pix = self.cri_pix(self.output, self.gt)
            l_pix.backward()
            self.optimizer_g.step()

            # set log
            self.log_dict['l_pix'] = l_pix.item()
        else:
            if self.net_d:
                # train net_d
                self.optimizer_d.zero_grad()
                for p in self.net_d.parameters():
                    p.requires_grad = True

                # compute WGAN loss
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                self.log_dict['l_d_real'] = l_d_real.item()
                self.log_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                self.log_dict['l_d_fake'] = l_d_fake.item()
                self.log_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_total = l_d_real + l_d_fake
                if self.cri_grad_penalty:
                    l_grad_penalty = self.cri_grad_penalty(
                        self.net_d, self.gt, self.output)
                    self.log_dict['l_grad_penalty'] = l_grad_penalty.item()
                    l_d_total += l_grad_penalty
                l_d_total.backward()
                self.optimizer_d.step()

            # train net_g
            self.optimizer_g.zero_grad()
            if self.net_d:
                for p in self.net_d.parameters():
                    p.requires_grad = False

            l_g_total = 0
            if (step - self.net_g_pretrain_steps) % self.net_d_steps == 0 and (
                    step - self.net_g_pretrain_steps) > self.net_d_init_steps:
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    self.log_dict['l_g_pix'] = l_g_pix.item()
                if self.cri_perceptual:
                    l_g_percep, _ = self.cri_perceptual(self.output, self.gt)
                    l_g_total += l_g_percep
                    self.log_dict['l_g_percep'] = l_g_percep.item()
                if self.cri_style:
                    _, l_g_style = self.cri_style(self.output, self.gt)
                    l_g_total += l_g_style
                    self.log_dict['l_g_style'] = l_g_style.item()
                if self.cri_texture:
                    l_g_texture = self.cri_texture(self.output, self.maps,
                                                   self.weights)
                    l_g_total += l_g_texture
                    self.log_dict['l_g_texture'] = l_g_texture.item()

                if self.net_d:
                    # gan loss
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    self.log_dict['l_g_gan'] = l_g_gan.item()

                l_g_total.backward()
                self.optimizer_g.step()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.features1 = self.net_extractor(self.match_img_in, self.img_ref1)
            self.pre_offset1, self.img_ref_feat1 = self.net_map(
                self.features1, self.img_ref1)
            self.output1, self.k_d = self.net_g(self.gt, self.dc_mask256, self.img_in_lq, self.img_ref_feat1, self.pre_offset1,
                                     self.img_ref_feat1)

            self.features2 = self.net_extractor(self.output1, self.img_ref2)
            self.pre_offset2, self.img_ref_feat2 = self.net_map(
                self.features2, self.img_ref2)
            self.output, _ = self.net_g(self.gt, self.dc_mask256, self.output1, self.k_d, self.pre_offset2,
                                     self.img_ref_feat2)

        self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_in_lq'] = self.img_in_lq.detach().cpu()
        out_dict['rlt'] = self.output.detach().cpu()
        # out_dict['sr_c'] = self.output_complex.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()

        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.net_d:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        pbar = ProgressBar(len(dataloader))
        avg_psnr = 0.
        avg_psnr_y = 0.
        avg_ssim_y = 0.
        avg_lpips = 0.
        dataset_name = dataloader.dataset.opt['name']
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path2'][0]))[0]
            img_name = str(idx) +'_' +img_name
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            pre_sr = visuals['rlt'].permute(0, 2, 3, 1).contiguous()
            pre_gt = visuals['gt'].permute(0, 2, 3, 1).contiguous()

            sr_img, gt_img = tensor2img([abs(torch.view_as_complex(pre_sr)), abs(torch.view_as_complex(pre_gt))])

            if 'multi' in dataset_name:  
                _, h, w, _ = self.gt_ori.shape
                sr_img = sr_img[:h, :w, :]
                gt_img = gt_img[:h, :w, :]

            if 'padding' in val_data.keys():
                padding = val_data['padding']
                original_size = val_data['original_size']
                if padding:
                    sr_img = sr_img[:original_size[0], :original_size[1]]

            if save_img:
                # out_c = torch.zeros(visuals['sr_c'].size())
                # for i in range(out_c.size(0)):
                #     out1 = visuals['sr_c'][:, i, :]
                #     out2 = abs(out1)
                #     out_c[:, i, :] = out2
                #     img_c = tensor2img(out_c)
                # img_c = tensor2img(abs(visuals['sr_c']))
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             f'{img_name}_{current_iter}.mat')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'],
                        f"{img_name}_{self.opt['name']}.mat")
                    if self.opt['suffix']:
                        save_img_path = save_img_path.replace(
                            '.mat', f'_{self.opt["suffix"]}.mat')
                # mmcv.imwrite(sr_img, save_img_path)
                sio.savemat(save_img_path,
                             {'rec': abs(torch.view_as_complex(pre_sr)).squeeze(0).detach().cpu().numpy(),
                              'gt': abs(torch.view_as_complex(pre_gt)).squeeze(0).detach().cpu().numpy()})

                sio.savemat(save_img_path,
                             {'rec': (torch.view_as_complex(pre_sr)).squeeze(0).detach().cpu().numpy(),
                              'gt': (torch.view_as_complex(pre_gt)).squeeze(0).detach().cpu().numpy()})
            # tentative for out of GPU memory
            del self.img_in_lq
            del self.output
            del self.gt
            torch.cuda.empty_cache()

            # calculate PSNR
            psnr = PSNRSSIMloss.nmse(gt_img / gt_img.max(), sr_img / sr_img.max())
            psnr_list.append(psnr)
            avg_psnr += psnr
            psnr_y = PSNRSSIMloss.psnr(gt_img / gt_img.max(), sr_img / sr_img.max())
            avg_psnr_y += psnr_y
            ssim_y = PSNRSSIMloss.ssim(gt_img / gt_img.max(), sr_img / sr_img.max())
            avg_ssim_y += ssim_y

            if not self.is_train:
                logger.info(f'# img {img_name} # PSNR: {psnr:.4e} '
                            f'# PSNR_Y: {psnr_y:.4e} # SSIM_Y: {ssim_y:.4e}.')

            pbar.update(f'Test {img_name}')

        avg_psnr = avg_psnr / (idx + 1)
        avg_psnr_y = avg_psnr_y / (idx + 1)
        avg_ssim_y = avg_ssim_y / (idx + 1)
        
        # log
        logger.info(f'# Validation {dataset_name} # MSE: {avg_psnr:.4e} '
                    f'# PSNR_Y: {avg_psnr_y:.4e} # SSIM_Y: {avg_ssim_y:.4e}.')
                   
        if tb_logger:
            tb_logger.add_scalar('mse', avg_psnr, current_iter)
            tb_logger.add_scalar('psnr_y', avg_psnr_y, current_iter)
            tb_logger.add_scalar('ssim_y', avg_ssim_y, current_iter)
