# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import matplotlib.pyplot as plt

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import AttentionPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, VisImage

from adet.utils.visualizer import TextVisualizer
from torchvision.transforms import Resize


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, attention_mode=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
            attention_mode (bool): if True, only draw attention mask prediction. If False, it's
                equal to normal sytle.
        """
        self.attention_mode = attention_mode
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cfg = cfg
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = cfg.MODEL.ROI_HEADS.NAME == "TextHead" or cfg.MODEL.TRANSFORMER.ENABLED

        self.predictor = AttentionPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        img_ori, pred_attentions = self.predictor(image)
        img = img_ori[0]      # 3, H, W
        print(f'in visual_tools, img.shape = {img.shape}')
        print(f'type of image:{type(image)}, type of img:{type(img)}')
        img = np.array(img.cpu()).transpose(1, 2, 0)   # 3, H, W -> H, W, 3   3: R, G, B

        atten_images = []
        for i, item in enumerate(pred_attentions):
            print(f'pred_attention.shape = {item.shape}')
            pred_shape = item.size()[-2:]
            img_tmp = np.array(Resize(pred_shape)(img_ori[0]).cpu())
            item = item.sigmoid()
            # if i == 3:
            #     print(item[0][0])
            thr = 0.5
            img_tmp[2, :, :] = np.where(np.array(item.cpu())[0][0] > thr, 0, img_tmp[2, :, :])
            img_tmp[1, :, :] = np.where(np.array(item.cpu())[0][0] > thr, 0, img_tmp[2, :, :])
            img_tmp[0, :, :] = np.where(np.array(item.cpu())[0][0] > thr, 255, img_tmp[2, :, :])   # R
            atten_images.append(VisImage(img_tmp.transpose(1, 2, 0)))
        return atten_images, VisImage(img)

    def run_on_image_w_gt(self, img_ori, pred_attentions, gt_attention, use_maxpool=True):
        """
        此时通过一个自定义的trainer给出原始的图像信息和预测的gt信息
        :param img_ori:
        :param pred_attentions:
        :param gt_attention: 1, H, W
        :return:
        """
        img = img_ori[0]      # 3, H, W
        print(f'in visual_tools, img.shape = {img.shape}')
        img = np.array(img.cpu()).transpose(1, 2, 0)   # 3, H, W -> H, W, 3   3: R, G, B

        atten_images = []
        for i, item in enumerate(pred_attentions):
            print(f'pred_attention.shape = {item.shape}')
            pred_shape = item.size()[-2:]
            img_tmp = np.array(Resize(pred_shape)(img_ori[0]).cpu())
            item = item.sigmoid().detach()
            # if i == 3:
            #     print(item[0][0])
            thr = 0.5
            img_tmp[2, :, :] = np.where(np.array(item.cpu())[0][0] > thr, 0, img_tmp[2, :, :])
            img_tmp[1, :, :] = np.where(np.array(item.cpu())[0][0] > thr, 0, img_tmp[1, :, :])
            img_tmp[0, :, :] = np.where(np.array(item.cpu())[0][0] > thr, 255, img_tmp[0, :, :])   # R
            atten_images.append(VisImage(img_tmp.transpose(1, 2, 0)))
        gt_images = []
        for i, item in enumerate(pred_attentions):
            """
            可视化真值标签
            """
            pred_shape = item.size()[-2:]
            if use_maxpool:
                import torch.nn.functional as F
                kernel_size = gt_attention.size(1) // pred_shape[0]
                gt = F.max_pool2d(gt_attention, kernel_size=kernel_size)
                gt = Resize(pred_shape)(gt.cpu())
            else:
                gt = Resize(pred_shape)(gt_attention.cpu())
            img_tmp = np.array(Resize(pred_shape)(img_ori[0]).cpu())
            gt = np.array(gt.sigmoid().detach())
            # if i == 3:
            #     print(item[0][0])
            thr = 0.5
            img_tmp[2, :, :] = np.where(np.array(gt)[0] > thr, 0, img_tmp[2, :, :])
            img_tmp[1, :, :] = np.where(np.array(gt)[0] > thr, 0, img_tmp[1, :, :])
            img_tmp[0, :, :] = np.where(np.array(gt)[0] > thr, 255, img_tmp[0, :, :])   # R
            gt_images.append(VisImage(img_tmp.transpose(1, 2, 0)))
        return atten_images, gt_images, VisImage(img)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def vis_bases(self, bases):
        basis_colors = [[2, 200, 255], [107, 220, 255], [30, 200, 255], [60, 220, 255]]
        bases = bases[0].squeeze()
        bases = (bases / 8).tanh().cpu().numpy()
        num_bases = len(bases)
        fig, axes = plt.subplots(nrows=num_bases // 2, ncols=2)
        for i, basis in enumerate(bases):
            basis = (basis + 1) / 2
            basis = basis / basis.max()
            basis_viz = np.zeros((basis.shape[0], basis.shape[1], 3), dtype=np.uint8)
            basis_viz[:, :, 0] = basis_colors[i][0]
            basis_viz[:, :, 1] = basis_colors[i][1]
            basis_viz[:, :, 2] = np.uint8(basis * 255)
            basis_viz = cv2.cvtColor(basis_viz, cv2.COLOR_HSV2RGB)
            axes[i // 2][i % 2].imshow(basis_viz)
        plt.show()

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
