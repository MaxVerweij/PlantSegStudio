# Copyright (c) OpenMMLab. All rights reserved.
import csv
import json
import os.path as osp

import numpy as np
import torch
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist
from mmengine.registry import METRICS

from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from mmdet3d.evaluation.functional.oneformer3d_utils.instance_seg_eval import instance_seg_eval
from mmdet3d.evaluation.functional.softgroup_utils.instance_seg_eval import \
    instance_seg_eval_softgroup
from mmdet3d.evaluation.metrics import InstanceSegMetric, SegMetric


@METRICS.register_module()
class UnifiedSegMetric(SegMetric):
    """Metric for instance, semantic, and panoptic evaluation. The order of
    classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(self,
                 thing_class_inds,
                 stuff_class_inds,
                 min_num_points,
                 id_offset,
                 sem_mapping,
                 inst_mapping,
                 metric_meta,
                 per_file_metric_prefix=None,
                 logger_keys=[('miou', ), ('all_ap', 'all_ap_50%', 'all_ap_25%'), ('pq', )],
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        self.per_file_metric_prefix = per_file_metric_prefix
        self.logger_keys = logger_keys
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        super().__init__(**kwargs)

    def process(self, data_batch, data_samples):
        """Process one batch of data samples and keep sample identifiers."""
        for data_sample in data_samples:
            pred_3d = data_sample['pred_pts_seg']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v

            sample_name = None
            if 'lidar_path' in data_sample:
                sample_name = osp.basename(data_sample.lidar_path)
            elif 'lidar_idx' in eval_ann_info:
                sample_name = str(eval_ann_info['lidar_idx'])

            self.results.append((eval_ann_info, cpu_pred_3d, sample_name))

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        label2cat = self.metric_meta['label2cat']
        ignore_index = self.metric_meta['ignore_index']
        classes = self.metric_meta['classes']
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)

        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        normalized_results = []
        for result in results:
            if len(result) == 3:
                eval_ann, single_pred_results, sample_name = result
            else:
                eval_ann, single_pred_results = result
                sample_name = str(eval_ann.get('lidar_idx', len(normalized_results)))
            normalized_results.append((eval_ann, single_pred_results, sample_name))

        for eval_ann, single_pred_results, _ in normalized_results:

            if self.metric_meta['dataset_name'] in ['S3DIS', 'Plant']:
                pan_gt = {}
                pan_gt['pts_semantic_mask'] = eval_ann['pts_semantic_mask']
                pan_gt['pts_instance_mask'] = \
                    eval_ann['pts_instance_mask'].copy()

                for stuff_cls in self.stuff_class_inds:
                    pan_gt['pts_instance_mask'][
                        pan_gt['pts_semantic_mask'] == stuff_cls] = \
                        np.max(pan_gt['pts_instance_mask']) + 1

                pan_gt['pts_instance_mask'] = np.unique(
                    pan_gt['pts_instance_mask'], return_inverse=True)[1]
                gt_masks_pan.append(pan_gt)
            else:
                gt_masks_pan.append(eval_ann)

            pred_masks_pan.append({
                'pts_instance_mask': single_pred_results['pts_instance_mask'][1],
                'pts_semantic_mask': single_pred_results['pts_semantic_mask'][1]
            })

            gt_semantic_masks_sem_task.append(eval_ann['pts_semantic_mask'])
            pred_semantic_masks_sem_task.append(single_pred_results['pts_semantic_mask'][0])

            if self.metric_meta['dataset_name'] in ['S3DIS', 'Plant']:
                gt_semantic_masks_inst_task.append(eval_ann['pts_semantic_mask'])
                gt_instance_masks_inst_task.append(eval_ann['pts_instance_mask'])
            else:
                sem_mask, inst_mask = self.map_inst_markup(eval_ann['pts_semantic_mask'].copy(),
                                                           eval_ann['pts_instance_mask'].copy(),
                                                           self.valid_class_ids[num_stuff_cls:],
                                                           num_stuff_cls)
                gt_semantic_masks_inst_task.append(sem_mask)
                gt_instance_masks_inst_task.append(inst_mask)

            pred_instance_masks_inst_task.append(
                torch.tensor(single_pred_results['pts_instance_mask'][0]))
            pred_instance_labels.append(torch.tensor(single_pred_results['instance_labels']))
            pred_instance_scores.append(torch.tensor(single_pred_results['instance_scores']))

        ret_pan = panoptic_seg_eval(
            gt_masks_pan, pred_masks_pan, classes, thing_classes,
            stuff_classes, self.min_num_points, self.id_offset, label2cat,
            ignore_index, logger)

        ret_sem = seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index[0],
            logger=logger)

        if self.metric_meta['dataset_name'] in ['S3DIS', 'Plant']:
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids,
                class_labels=classes[:-1],
                logger=logger)
        else:
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1],
                logger=logger)

        metrics = dict()
        for ret, keys in zip((ret_sem, ret_inst, ret_pan), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]

        if self.per_file_metric_prefix:
            per_file_metrics = self._compute_per_file_metrics(
                normalized_results, classes, thing_classes, stuff_classes,
                label2cat, ignore_index, logger)
            self._dump_per_file_metrics(per_file_metrics, logger)
        return metrics

    def _compute_per_file_metrics(self, results, classes, thing_classes,
                                  stuff_classes, label2cat, ignore_index,
                                  logger):
        """Compute metrics for each individual sample."""
        per_file_metrics = []
        valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        num_stuff_cls = len(stuff_classes)

        for eval_ann, single_pred_results, sample_name in results:
            ret_sem = seg_eval(
                [eval_ann['pts_semantic_mask']],
                [single_pred_results['pts_semantic_mask'][0]],
                label2cat,
                ignore_index[0],
                logger='silent')

            if self.metric_meta['dataset_name'] in ['S3DIS', 'Plant']:
                inst_gt_sem = [eval_ann['pts_semantic_mask']]
                inst_gt_mask = [eval_ann['pts_instance_mask']]
                inst_valid_class_ids = valid_class_ids
                inst_class_labels = classes[:-1]
            else:
                sem_mask, inst_mask = self.map_inst_markup(
                    eval_ann['pts_semantic_mask'].copy(),
                    eval_ann['pts_instance_mask'].copy(),
                    valid_class_ids[num_stuff_cls:],
                    num_stuff_cls)
                inst_gt_sem = [sem_mask]
                inst_gt_mask = [inst_mask]
                inst_valid_class_ids = valid_class_ids[num_stuff_cls:]
                inst_class_labels = classes[num_stuff_cls:-1]

            ret_inst = instance_seg_eval(
                inst_gt_sem,
                inst_gt_mask,
                [torch.tensor(single_pred_results['pts_instance_mask'][0])],
                [torch.tensor(single_pred_results['instance_labels'])],
                [torch.tensor(single_pred_results['instance_scores'])],
                valid_class_ids=inst_valid_class_ids,
                class_labels=inst_class_labels,
                logger='silent')

            if self.metric_meta['dataset_name'] in ['S3DIS', 'Plant']:
                pan_gt = {}
                pan_gt['pts_semantic_mask'] = eval_ann['pts_semantic_mask']
                pan_gt['pts_instance_mask'] = eval_ann['pts_instance_mask'].copy()
                for stuff_cls in self.stuff_class_inds:
                    pan_gt['pts_instance_mask'][
                        pan_gt['pts_semantic_mask'] == stuff_cls] = \
                        np.max(pan_gt['pts_instance_mask']) + 1
                pan_gt['pts_instance_mask'] = np.unique(
                    pan_gt['pts_instance_mask'], return_inverse=True)[1]
            else:
                pan_gt = eval_ann

            ret_pan = panoptic_seg_eval(
                [pan_gt],
                [{
                    'pts_instance_mask': single_pred_results['pts_instance_mask'][1],
                    'pts_semantic_mask': single_pred_results['pts_semantic_mask'][1]
                }],
                classes,
                thing_classes,
                stuff_classes,
                self.min_num_points,
                self.id_offset,
                label2cat,
                ignore_index,
                logger='silent')

            per_file_metrics.append(
                dict(
                    file=sample_name,
                    miou=float(ret_sem['miou']),
                    all_ap=float(ret_inst['all_ap']),
                    all_ap_50=float(ret_inst['all_ap_50%']),
                    all_ap_25=float(ret_inst['all_ap_25%']),
                    pq=float(ret_pan['pq']),
                    pq_things=float(ret_pan['pq_things']),
                    pq_stuff=float(ret_pan['pq_stuff']),
                    sq_mean=float(ret_pan['sq_mean']),
                    rq_mean=float(ret_pan['rq_mean'])))

        if per_file_metrics:
            preview_rows = min(10, len(per_file_metrics))
            logger.info('Per-file metrics preview (first %d samples):', preview_rows)
            for row in per_file_metrics[:preview_rows]:
                logger.info(
                    '%s | miou=%.4f all_ap=%.4f all_ap_50=%.4f pq=%.4f',
                    row['file'], row['miou'], row['all_ap'],
                    row['all_ap_50'], row['pq'])

        return per_file_metrics

    def _dump_per_file_metrics(self, per_file_metrics, logger):
        """Write per-file metrics to CSV and JSON."""
        output_dir = osp.dirname(self.per_file_metric_prefix)
        if output_dir:
            mkdir_or_exist(output_dir)

        csv_path = f'{self.per_file_metric_prefix}.csv'
        json_path = f'{self.per_file_metric_prefix}.json'
        fieldnames = [
            'file', 'miou', 'all_ap', 'all_ap_50', 'all_ap_25', 'pq',
            'pq_things', 'pq_stuff', 'sq_mean', 'rq_mean'
        ]

        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_file_metrics)

        with open(json_path, 'w') as json_file:
            json.dump(per_file_metrics, json_file, indent=2)

        logger.info('Per-file metrics written to %s and %s', csv_path, json_path)

    def map_inst_markup(self, pts_semantic_mask, pts_instance_mask, valid_class_ids, num_stuff_cls):
        """Map gt instance and semantic classes back from panoptic annotations.

        Args:
            pts_semantic_mask (np.array): of shape (n_raw_points,)
            pts_instance_mask (np.array): of shape (n_raw_points.)
            valid_class_ids (Tuple): of len n_instance_classes
            num_stuff_cls (int): number of stuff classes

        Returns:
            Tuple:
                np.array: pts_semantic_mask of shape (n_raw_points,)
                np.array: pts_instance_mask of shape (n_raw_points,)
        """
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]

        return pts_semantic_mask, pts_instance_mask


@METRICS.register_module()
class InstanceSegMetric_(InstanceSegMetric):
    """The only difference with InstanceSegMetric is that following ScanNet
    evaluator we accept instance prediction as a boolean tensor of shape
    (n_points, n_instances) instead of integer tensor of shape (n_points, ).

    For this purpose we only replace instance_seg_eval call.
    """

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.classes = self.dataset_meta['classes']
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']

        gt_semantic_masks = []
        gt_instance_masks = []
        pred_instance_masks = []
        pred_instance_labels = []
        pred_instance_scores = []

        for eval_ann, single_pred_results in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            gt_instance_masks.append(eval_ann['pts_instance_mask'])
            pred_instance_masks.append(single_pred_results['pts_instance_mask'])
            pred_instance_labels.append(single_pred_results['instance_labels'])
            pred_instance_scores.append(single_pred_results['instance_scores'])

        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes,
            logger=logger)

        return ret_dict


@METRICS.register_module()
class SoftGroupSegMetric(SegMetric):
    """Metric for instance, semantic. Panoptic eval is not refactored for
    softgroup in PSS. The order of classes must be [stuff classes, thing
    classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(self,
                 thing_class_inds,
                 stuff_class_inds,
                 min_num_points,
                 id_offset,
                 sem_mapping,
                 inst_mapping,
                 metric_meta,
                 logger_keys=[('miou', ), ('all_ap', 'all_ap_50%', 'all_ap_25%')],
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        self.logger_keys = logger_keys
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        super().__init__(**kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        valid_class_ids_inst = np.arange(len(self.valid_class_ids)) + 1
        label2cat = self.metric_meta['label2cat']
        ignore_index = self.metric_meta['ignore_index']
        classes = self.metric_meta['classes']

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        pred_instances_softgroup = []
        gt_instances_softgroup = []

        for eval_ann, single_pred_results in results:
            gt_semantic_masks_sem_task.append(eval_ann['pts_semantic_mask'])
            pred_semantic_masks_sem_task.append(
                single_pred_results['point_wise_results']['semantic_preds'])

            pred_instances_softgroup.append((single_pred_results['pred_instances']))
            gt_instances_softgroup.append((single_pred_results['gt_instances']))

        ret_sem = seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index[0],
            logger=logger)

        ret_inst = instance_seg_eval_softgroup(
            pred_instances_softgroup,
            gt_instances_softgroup,
            valid_class_ids=valid_class_ids_inst,
            class_labels=classes[:-1],
            logger=logger)

        metrics = dict()
        for ret, keys in zip((ret_sem, ret_inst), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        return metrics

    def map_inst_markup(self, pts_semantic_mask, pts_instance_mask, valid_class_ids, num_stuff_cls):
        """Map gt instance and semantic classes back from panoptic annotations.

        Args:
            pts_semantic_mask (np.array): of shape (n_raw_points,)
            pts_instance_mask (np.array): of shape (n_raw_points.)
            valid_class_ids (Tuple): of len n_instance_classes
            num_stuff_cls (int): number of stuff classes

        Returns:
            Tuple:
                np.array: pts_semantic_mask of shape (n_raw_points,)
                np.array: pts_instance_mask of shape (n_raw_points,)
        """
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]

        return pts_semantic_mask, pts_instance_mask
