import torch
from collections import Counter
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm
import torch.nn.functional as F
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class VideoRecurrenttestModel(VideoBaseModel):

    def __init__(self, opt):
        super(VideoRecurrentModel, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for differnet lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'raftupnet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        super(VideoRecurrentModel, self).optimize_parameters(current_iter)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics and not hasattr(self, 'metric_results'):
            self.metric_results = {}
            num_frame_each_folder = Counter(dataset.data_info['folder'])
            for folder, num_frame in num_frame_each_folder.items():
                self.metric_results[folder] = torch.zeros(
                    num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')

        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first
        # num_folders results will be recorded. (To avoid wait-dead)
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)
            # print("[val_data['lq']]",val_data['lq'].shape)
            # print("[val_data['gt']]",val_data['gt'].shape)

            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            if self.center_frame_only:
                # visuals['result'] = visuals['result'].unsqueeze(1)
                visuals['lq'] = visuals['lq'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            # evaluate
            if i < num_folders:
                # for idx in range(visuals['result'].size(1)):
                for idx in range(visuals['lq'].size(1)):
                    # result = visuals['result'][0, idx, :, :, :]
                    lq_img = visuals['lq'][0, idx, :, :, :]
                    # print("result",result.shape)
                    lq_img = tensor2img([lq_img])  # uint8, bgr
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        # print("[result_img]".result_img.shape)
                        # print("[gt]",gt.shape)
                        if result.shape != gt.shape:
                            gt = gt[:,:result.shape[1],:result.shape[2]]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        # if self.center_frame_only:
                        #     gt_img = tensor2img([gt])  # uint8, bgr


                    if save_img:
                        if self.opt['is_train']:
                            raise NotImplementedError('saving image is not supported during training.')
                        else:
                            if self.center_frame_only:  # vimeo-90k
                                # print("{clip_}",clip_)
                                clip_ = val_data['lq_path'].split('/')[-3]
                                
                                seq_ = val_data['lq_path'].split('/')[-2]
                                name_ = f'{clip_}_{seq_}'
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{name_}_{self.opt['name']}.png")
                            else:  # others
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{idx:08d}_{self.opt['name']}.png")
                            # image name only for REDS dataset
                        imwrite(result_img, img_path)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            metric_data = dict(img1=lq_img, img2=gt_img)
                            # metric_data = dict(img1=result_img, img2=gt_img)
                            # print("metric_data",metric_data[result_img],"metric_data",metric_data[gt_img])
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result

                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self):
        n = self.lq.size(1)
        # print("[n]",n)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        # print("[flip_seq]",flip_seq)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)
        # print("[self.lq  111]",self.lq.shape)
        batch, _, _, h, w = self.lq.shape
        # print("[self.lq  111]",self.lq.shape)
        
        if (h % 8 != 0 ) or (w % 8 != 0 ):
            # print("[ before self.lq]",self.lq.shape)
            self.lq = self.lq.squeeze(0)
            # print("selflqqqq",type(self.lq))
            padder = InputPadder(self.lq.shape)   #  , mode='sintel'
            self.lq = padder.pad(self.lq)
            self.lq = self.lq[0]  #  torch.Tensor(
            # print("[self.lq]",type(self.lq))
            self.lq = self.lq.unsqueeze(0)
            # print("[self.lq padder]",self.lq.shape)
#         # ref, supp = padder.pad(ref, supp)
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            
            # self.lq = self.lq[:,0:9,:,:,:]
            # self.lq = self.lq.unsqueeze(0)
            # print("[self.lq 222]",self.lq.shape)   #  torch.Size([1, 9, 3, 288, 360])  torch.Size([1, 32, 3, 264, 636])
            self.output = self.net_g(self.lq)
            # print("[self.output ]",self.output.shape)
            if (h % 8 != 0 ) or (w % 8 != 0 ):
                # self.output = padder.unpadx2(self.output)
                self.output = padder.unpadx4(self.output)

                # self.output = self.output[:,:,:,:h*self.opt['scale'],:w*self.opt['scale']]
            # print("[self.output padder]",self.output.shape)
            torch.cuda.empty_cache()

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()




class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
            # self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    def unpadx2(self,x):
        ht, wd = x.shape[-2:]
        c = [2*self._pad[2], ht-2*self._pad[3], 2*self._pad[0], wd-2*self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    def unpadx3(self,x):
        ht, wd = x.shape[-2:]
        c = [3*self._pad[2], ht-3*self._pad[3], 3*self._pad[0], wd-3*self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

    def unpadx4(self,x):
        ht, wd = x.shape[-2:]
        c = [4*self._pad[2], ht-4*self._pad[3], 4*self._pad[0], wd-4*self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
