from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os.path as osp
import os


@registry.register_task("qformer_train")
class QFormerTrain(BaseTask):

    def __init__(
            self,
            evaluate,
            report_metric=True,
            condition_image="garm",
            target_image="viton",
            save_dir="generated_images",
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=None,
            use_png=True,
            eta=0.0,
            save_imgs_per_epoch=False,
            switch_generate=False,
            vae_encode_method="mode"
    ):
        super().__init__()
        self.save_dir = save_dir
        self.evaluate = evaluate

        self.report_metric = report_metric
        self.condition_image = condition_image
        self.target_image = target_image
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.epoch = 0
        self.use_png = use_png
        self.save_imgs_per_epoch = save_imgs_per_epoch
        self.switch_generate = switch_generate
        self.vae_encode_method = vae_encode_method

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.generate_cfg

        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)

        num_inference_steps = generate_cfg.get("num_inference_steps", 50)
        guidance_scale = generate_cfg.get("guidance_scale", 7.5)
        condition_image = generate_cfg.get("condition_image", "garm")
        target_image = generate_cfg.get("target_image", "viton")
        save_dir = generate_cfg.get("save_dir", "generated_images")
        seed = generate_cfg.get("seed", run_cfg.get("seed", None))
        eta = generate_cfg.get("eta", 0.0)
        use_png = generate_cfg.get("use_png", True)
        save_imgs_per_epoch = generate_cfg.get("save_imgs_per_epoch", False)
        switch_generate = generate_cfg.get("switch_generate", False)
        vae_encode_method = generate_cfg.get("vae_encode_method", "mode")

        return cls(
            save_dir=save_dir,
            evaluate=evaluate,
            report_metric=report_metric,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            condition_image=condition_image,
            target_image=target_image,
            seed=seed,
            eta=eta,
            use_png=use_png,
            save_imgs_per_epoch=save_imgs_per_epoch,
            switch_generate=switch_generate,
            vae_encode_method=vae_encode_method
        )

    def valid_step(self, model, samples):
        results = []

        generate_images = model.generate(
            samples,
            condition_image=self.condition_image,
            target_image=self.target_image,
            seed=self.seed,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            negative_prompt=None,
            eta=self.eta,
            vae_encode_method=self.vae_encode_method
        )

        if self.switch_generate:
            switch_generate_images = model.generate(
                samples,
                condition_image=self.target_image,
                target_image=self.condition_image,
                seed=self.seed,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=None,
                eta=self.eta,
                vae_encode_method=self.vae_encode_method
            )
        else:
            switch_generate_images = [None] * len(generate_images)

        ids = samples["id"]
        orders = samples["order"]
        categories = samples["category"]
        dataset_names = samples["dataset_name"]
        image_names = samples["image_name"]
        image_paths = samples["image_path"]
        save_root = registry.get_path("result_dir")
        for image, switch_image, id_, order, category, dataset_name, image_name, image_path in zip(
                generate_images,
                switch_generate_images,
                ids,
                orders,
                categories,
                dataset_names,
                image_names,
                image_paths):
            if self.save_imgs_per_epoch:
                save_dir = osp.join(save_root, f"{self.save_dir}-epoch-{self.epoch}", order, dataset_name, category)
            else:
                save_dir = osp.join(save_root, f"{self.save_dir}", order, dataset_name, category)

            if not osp.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            if self.use_png:
                image_name = image_name.replace(".jpg", ".png")
                image.save(osp.join(save_dir, image_name))
                if switch_image is not None:
                    switch_image_name = image_name.replace(".png", "_switch.png")
                    switch_image.save(osp.join(save_dir, switch_image_name))
            else:
                image.save(osp.join(save_dir, image_name), quality=95)
                if switch_image is not None:
                    switch_image_name = image_name.replace(".jpg", "_switch.jpg")
                    switch_image.save(osp.join(save_dir, switch_image_name))

            this_item = {
                "image_name": image_name,
                "image_path": osp.join(save_dir, image_name),
                "ori_image_path": image_path,
                "order": order,
                "category": category,
                "dataset_name": dataset_name,
                "id": id_
            }
            if switch_image is not None:
                this_item["switch_image_name"] = switch_image_name
                this_item["switch_image_path"] = osp.join(save_dir, switch_image_name)
            results.append(this_item)
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="id",
        )

        metrics = {"agg_metrics": 0.0}
        self.epoch += 1
        return metrics
