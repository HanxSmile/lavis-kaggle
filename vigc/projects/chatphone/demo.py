from vigc.common.registry import registry
from vigc.common.config import Config
from vigc.datasets.builders.chatphone.chatphone_processor.phone_number_detection import PhoneNumberDetection
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import os.path as osp
from tqdm.auto import tqdm
import json

Preprocessor_obj = PhoneNumberDetection()

DATA_ROOT = r"/home/work/hanxiao_train/input/chatphone/evaluation"
extra_gt_data = [
    r"[DS Model] Chat Phone Number Cases - v1.1_recall_check_extra_data_0118 (1).csv",
    # "[DS Model] Chat Phone Number Cases - v1.2_not_detected_data_0123.csv"
]
extra_gt_csv = [pd.read_csv(osp.join(DATA_ROOT, _)) for _ in extra_gt_data]
extra_gt_mapping = {}
for csv_ in extra_gt_csv:
    if "Phone Number?" in csv_.columns:
        gt_column = "Phone Number?"
    else:
        gt_column = "Contain Phone Number?"
    extra_gt_key = csv_["msg_content"].values
    extra_gt_value = csv_[gt_column].values
    extra_gt_mapping.update({k.strip(): v.lower() == "yes" for k, v in zip(extra_gt_key, extra_gt_value)})

ALL_FILES = [
    r"ID Scene Annotation - 9.18 ID Scene Coverage - 10k, Sep data.csv",
    r"ID Scene Annotation - 10.23 ID Scene Coverage - 10k, Oct data.csv",
    r"ID Scene Annotation - 11.20 ID Scene Coverage - 10k, Nov data.csv",
    r"ID Scene Annotation - 12.18 ID Scene Coverage - 10k, Dec data.csv",
    r"ID Scene Annotation - 1.22 ID Scene Coverage - 10k, Jan data.csv"
]

RESULT_NAMES = [
    "Sep_res.csv",
    "Oct_res.csv",
    "Nov_res.csv",
    "Dec_res.csv",
    "Jan_res.csv"
]

DATE = [
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "Jan",
]

ILLEGAL_COLUMN = "Redirect to private contact"
MSG_TYPES = ['MSG_TYPE_TEXT', 'MSG_TYPE_IMAGE', 'MSG_TYPE_IMAGE_WITH_TEXT', 'MSG_TYPE_VIDEO']


class PhoneNumberDetection:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def process(self, text):
        text = Preprocessor_obj.preprocess_text(text)
        with self.model.maybe_autocast():
            input_ids, attention_mask = self.model.preprocess_inputs({"text": [text]})
            out = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            out = self.model.pooler(out, attention_mask)
            logits = self.model.head(out).squeeze()
            probs = F.sigmoid(logits).item()
        return probs > self.threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--checkpoint", required=True, help="path to the checkpoint file")
    parser.add_argument("--threshold", type=float, default=0.9955)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def init_model(model_config, threshold, checkpoint):
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    model.load_checkpoint(checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
    return PhoneNumberDetection(model, threshold)


def run(IDX, not_detected):
    message_type = ""
    path = osp.join(DATA_ROOT, ALL_FILES[IDX])
    csv = pd.read_csv(path)
    print(csv["msg_type"].unique())
    print(csv.columns)
    print(len(csv))
    gt_nums = 0
    hit_nums = 0
    detected_nums = 0
    total_nums = 0
    new_detected_msgs = []
    new_detected_msgs_type = []
    new_detected_result = []
    new_detected_msgs_date = []
    for i, row in tqdm(csv.iterrows(), total=len(csv)):
        msg_type = row.msg_type
        if message_type and msg_type != message_type:
            continue
        total_nums += 1
        msg_content = row.msg_content

        content = json.loads(msg_content)
        if "text" in content:
            msg = content["text"]["text"]
        else:
            continue

        is_illegal = row.illegal_scene == ILLEGAL_COLUMN
        if msg_content.strip() in extra_gt_mapping:
            is_illegal = extra_gt_mapping[msg_content.strip()]
        gt_nums += is_illegal
        detect_res = pnd_obj.process(msg)
        hit_nums += (detect_res and is_illegal)

        if is_illegal and not detect_res:
            not_detected.append(msg)
        if detect_res:
            detected_nums += 1
            if not is_illegal:
                new_detected_msgs.append(msg_content)
                new_detected_msgs_date.append(DATE[IDX])
                new_detected_result.append(detect_res)
                new_detected_msgs_type.append(msg_type)

    print(total_nums, gt_nums, hit_nums, detected_nums)
    if total_nums == 10000:
        print(gt_nums / total_nums * (1 - hit_nums / gt_nums))
        print(hit_nums / detected_nums)

    res_df = pd.DataFrame({
        "Month": new_detected_msgs_date,
        "msg_type": new_detected_msgs_type,
        "msg_content": new_detected_msgs,
        "detected_result": [json.dumps(_) for _ in new_detected_result],
    })

    return res_df


def run_not_detected(IDX):
    message_type = ""
    path = osp.join(DATA_ROOT, ALL_FILES[IDX])
    csv = pd.read_csv(path)

    gt_nums = 0
    hit_nums = 0
    total_nums = 0
    not_detected_msgs = []
    not_detected_msgs_type = []
    not_detected_result = []
    not_detected_msgs_date = []
    for i, row in tqdm(csv.iterrows(), total=len(csv)):
        msg_type = row.msg_type
        if message_type and msg_type != message_type:
            continue
        total_nums += 1
        msg_content = row.msg_content

        content = json.loads(msg_content)
        if "text" in content:
            msg = content["text"]["text"]
        else:
            continue
        is_illegal = row.illegal_scene == ILLEGAL_COLUMN
        if msg_content.strip() in extra_gt_mapping:
            is_illegal = extra_gt_mapping[msg_content.strip()]
        gt_nums += is_illegal
        detect_res = pnd_obj.process(msg)
        hit_nums += (detect_res and is_illegal)

        if is_illegal and not detect_res:
            not_detected_msgs.append(msg_content)
            not_detected_msgs_date.append(DATE[IDX])
            not_detected_result.append(detect_res)
            not_detected_msgs_type.append(msg_type)

    res_df = pd.DataFrame({
        "Month": not_detected_msgs_date,
        "msg_type": not_detected_msgs_type,
        "msg_content": not_detected_msgs,
        "detected_result": [json.dumps(_) for _ in not_detected_result],
    })

    return res_df


if __name__ == '__main__':
    args = parse_args()
    cfg = Config(args)
    model_config = cfg.model_cfg
    threshold = args.threshold
    pnd_obj = init_model(model_config, threshold, args.checkpoint)

    all_result = []
    not_detected_result = []
    for id_ in range(5):
        this_result = run(id_, [])
        all_result.append(this_result)
        not_detected_result.append(run_not_detected(id_))

    result = pd.concat(all_result)
    not_detected_result = pd.concat(not_detected_result)

    print(f"there are {len(result)} new detected results")
    result.to_csv("new_detected_results.csv")
    print(result.head())

    print(f"there are {len(not_detected_result)} not detected result.")
    not_detected_result.to_csv("not_detected_results.csv")
    print(not_detected_result.head())
