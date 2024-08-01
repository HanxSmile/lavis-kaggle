from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
from vigc.common.dist_utils import main_process
import os.path as osp
import os
import json
import numpy as np
from tqdm.auto import tqdm
from vigc.datasets.datasets.vton_datasets.eval_dataset import VtonFolderDataset
from cleanfid import fid
from torch.utils.data import DataLoader


@registry.register_task("vton_train_eval")
class VtonTrain(BaseTask):

    def __init__(
            self,
            evaluate,
            report_metric=True,
            save_dir="generated_images",
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
            clip_skip=None,
            num_inference_steps=50,
            guidance_scale=7.5,
            eta=0.0,
            use_png=False,
            num_workers=4,
            batch_size=32
    ):
        super(VtonTrain, self).__init__()
        self.save_dir = save_dir
        self.evaluate = evaluate

        self.report_metric = report_metric
        self.use_png = use_png

        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.num_images_per_prompt = num_images_per_prompt
        self.clip_skip = clip_skip
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.eta = eta

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.epoch = 0
        self.model = None

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.generate_cfg

        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)
        save_dir = run_cfg.get("save_dir", "generated_images")

        do_classifier_free_guidance = generate_cfg.get("do_classifier_free_guidance", False)
        num_images_per_prompt = generate_cfg.get("num_images_per_prompt", 1)
        clip_skip = generate_cfg.get("clip_skip", None)
        num_inference_steps = generate_cfg.get("num_inference_steps", 50)
        guidance_scale = generate_cfg.get("guidance_scale", 7.5)

        use_png = generate_cfg.get("use_png", False)
        num_workers = generate_cfg.get("num_workers", 4)
        batch_size = generate_cfg.get("batch_size", 32)

        return cls(
            save_dir=save_dir,
            evaluate=evaluate,
            report_metric=report_metric,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_png=use_png,
            num_workers=num_workers,
            batch_size=batch_size
        )

    @property
    def ssim_scorer(self):
        return self.model.ssim_scorer

    @property
    def lpips_scorer(self):
        return self.model.lpips_scorer

    def valid_step(self, model, samples):
        self.model = model
        results = []

        generate_images = model.generate(
            samples,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_images_per_prompt=self.num_images_per_prompt,
            clip_skip=self.clip_skip,
            num_inference_steps=self.num_inference_steps,
            generator=None,
            guidance_scale=self.guidance_scale,
            eta=self.eta,
        )

        ids = samples["id"]
        orders = samples["paired"]
        categories = samples["category"]
        dataset_names = samples["dataset_name"]
        image_names = samples["image_name"]
        image_paths = samples["image_path"]
        save_root = registry.get_path("result_dir")
        for image, id_, order, category, dataset_name, image_name, image_path in zip(generate_images,
                                                                                     ids,
                                                                                     orders,
                                                                                     categories,
                                                                                     dataset_names,
                                                                                     image_names,
                                                                                     image_paths):
            save_dir = osp.join(save_root, f"{self.save_dir}-epoch-{self.epoch}", order, dataset_name, category)

            if not osp.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            if self.use_png:
                image_name = image_name.replace(".jpg", ".png")
                image.save(osp.join(save_dir, image_name))
            else:
                image.save(osp.join(save_dir, image_name), quality=95)

            this_item = {
                "image_name": image_name,
                "image_path": osp.join(save_dir, image_name),
                "ori_image_path": image_path,
                "order": order,
                "category": category,
                "dataset_name": dataset_name,
                "id": id_
            }
            results.append(this_item)
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}
        self.epoch += 1
        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        edit_dists = []

        save_root = registry.get_path("result_dir")

        with open(eval_result_file) as f:
            results = json.load(f)

        all_dataset_info = dict()
        for item in results:
            dataset_name = item["dataset_name"]
            order = item["order"]
            category = item["category"]
            ori_image_path = item["ori_image_path"]
            gen_image_path = item["image_path"]

            this_order_info = all_dataset_info.setdefault(order, dict())
            this_dataset_info = this_order_info.setdefault(dataset_name, dict())
            this_category_info = this_dataset_info.setdefault(category, list())

            this_category_info.append(dict(image_path=gen_image_path, ori_image_path=ori_image_path))

        unpaired_metric_dic = dict()
        unpaired_order_dic = all_dataset_info["unpaired"]
        fids, kids, ssims, lipipses = [], [], [], []
        for dataset_name, dataset_info in unpaired_order_dic.items():
            for category, category_info in dataset_info.items():
                save_dir = osp.join(save_root, f"{self.save_dir}-epoch-{self.epoch}", "unpaired", dataset_name,
                                    category)

                fid_score = fid.compute_fid(
                    save_dir, dataset_name=f"{dataset_name}_{category}",
                    mode='clean', verbose=True, dataset_split="custom", use_dataparallel=False
                )
                kid_score = fid.compute_kid(
                    save_dir, dataset_name=f"{dataset_name}_{category}", mode='clean', verbose=True,
                    dataset_split="custom", use_dataparallel=False)
                unpaired_metric_dic[f"{dataset_name}_{category}_fid_score"] = fid_score
                unpaired_metric_dic[f"{dataset_name}_{category}_kid_score"] = kid_score

                fids.append(-fid_score)
                kids.append(-10 * kid_score)

        paired_metric_dic = dict()
        paired_order_dic = all_dataset_info["paired"]
        for dataset_name, dataset_info in paired_order_dic.items():
            for category, category_info in dataset_info.items():
                this_category_ds = VtonFolderDataset(category_info)
                this_category_dl = DataLoader(this_category_ds, batch_size=self.batch_size, shuffle=False,
                                              num_workers=self.num_workers)
                for (gen_images, gt_images) in tqdm(this_category_dl, total=len(this_category_dl)):
                    gen_images = gen_images.to(self.model.device)
                    gt_images = gt_images.to(self.model.device)

                    self.ssim_scorer.update(gen_images, gt_images)
                    self.lpips_scorer.update(gen_images, gt_images)

                ssim_score = self.ssim_scorer.compute().item()
                lpips_score = self.lpips_scorer.compute().item()
                paired_metric_dic[f"{dataset_name}_{category}_ssim_score"] = ssim_score
                paired_metric_dic[f"{dataset_name}_{category}_lpips_score"] = lpips_score
                self.ssim_scorer.reset()
                self.lpips_scorer.reset()
                ssims.append(10 * ssim_score)
                lipipses.append(-100 * lpips_score)

        eval_ret = dict()
        eval_ret.update(unpaired_metric_dic)
        eval_ret.update(paired_metric_dic)
        log_stats = {split_name: {k: v for k, v in eval_ret.items()}}

        with open(
                osp.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in eval_ret.items()}

        coco_res["agg_metrics"] = np.mean(kids) + np.mean(fids) + np.mean(ssims) + np.mean(lipipses) + 100

        return coco_res
