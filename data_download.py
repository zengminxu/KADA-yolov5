import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    name = "coco-2017",
    split="train",  # 下载训练集
    label_types=["detections"],  # 下载目标检测标注文件
    classes=['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket'],
    max_samples=100,  # 下载图片数目
    only_matching=True,  # 只下载匹配到类别的图片
    dataset_dir=".",  # 下载到当前目录
)

export_view = dataset.exclude_labels(tags="extra")

export_view.export(
    export_dir="./train",
    dataset_type=fo.types.YOLOv5Dataset,
    split="train",
    classes=['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket'],
)

dataset.untag_labels("extra")


