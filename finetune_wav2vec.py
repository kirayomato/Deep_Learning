from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import torch
from datasets import Dataset
from datasets import load_dataset, load_metric, Audio
from datasets import ClassLabel
import random
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import numpy as np
import evaluate


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {self.feature_extractor_input_name: feature[self.feature_extractor_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(
                torch.long)

        return batch


@dataclass
class ModelConfig:
    feat_proj_dropout: float = field(default=0.0, metadata={
                                     "help": "The dropout ratio for the projected features."})
    attention_dropout: float = field(default=0.0, metadata={
                                     "help": "The dropout ratio for the attention probabilities."})
    hidden_dropout: float = field(default=0.0, metadata={
                                  "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."})
    final_dropout: float = field(default=0.0, metadata={
                                 "help": "The dropout probability for the final projection layer."})
    gradient_checkpointing: bool = field(default=True, metadata={
                                         "help": "Whether to use gradient checkpointing to save memory at the expense of slower backward pass."})
    layerdrop: float = field(default=0.0, metadata={
                             "help": "The LayerDrop probability."})
    ctc_loss_reduction: str = field(default="mean", metadata={
                                    "help": "The way the CTC loss should be reduced. Should be one of 'mean' or 'sum'."})
    ctc_zero_infinity: bool = field(default=True, metadata={
                                    "help": "Whether to zero infinite losses and the associated gradients of torch.nn.CTCLoss."})
    activation_dropout: float = field(default=0.0, metadata={
                                      "help": "The dropout ratio for activations inside the fully connected layer."})


@dataclass
class TrainingConfig:
    model: str = field(
        default="facebook/wav2vec2-xls-r-300m",
        metadata={"help": "模型名称"}
    )
    repo_name: str = field(
        default="wav2vec2-large-xls-r-300m-Chinese-colab",
        metadata={"help": "仓库名称"}
    )
    num_train_epochs: int = field(
        default=5,
        metadata={"help": "训练的总轮数"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "每个设备的训练批量大小"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "初始学习率"}
    )
    gradient_accumulation_steps: int = field(
        default=32,
        metadata={"help": "梯度累积步数"}
    )
    eval_strategy: str = field(
        default="epoch",
        metadata={"help": "评估策略"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "保存模型的步数"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "评估的步数"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "日志记录的步数"}
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "学习率预热的步数"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "保存的最大检查点数量"}
    )
    max_duration_in_seconds: float = field(
        default=10.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    train_data: str = field(
        default="E:/Download/CV16/zh-CN_train_0/",
        metadata={"help": "训练数据路径"}
    )
    test_data: str = field(
        default="E:/Download/CV16/zh-CN_test_0/",
        metadata={"help": "测试数据路径"}
    )
    train_data_config: str = field(
        default="transcript_zh-CN_train.tsv",
        metadata={"help": "训练数据配置文件"}
    )
    test_data_config: str = field(
        default="transcript_zh-CN_test.tsv",
        metadata={"help": "测试数据配置文件"}
    )
    train_split: float = field(
        default=0.1,
        metadata={"help": "训练集占比"}
    )
    test_split: float = field(
        default=0.1,
        metadata={"help": "测试集占比"}
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "学习率调度器类型"}
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "是否推送到Hub"}
    )
    train_samples: int = field(
        default=None,
        metadata={"help": "训练样本数"}
    )
    test_samples: int = field(
        default=None,
        metadata={"help": "测试样本数"}
    )
    hub_token: str = field(
        default=None,
        metadata={"help": "Hub Token"})


def remove_special_characters(batch):
    chars_to_remove_regex = r'[^a-zA-Z\u4e00-\u9fff]'
    # remove special characters
    batch["sentence"] = re.sub(
        chars_to_remove_regex, ' ', batch["sentence"]).lower()

    return batch


def prepare_dataset(batch, feature_name, feature_extractor, tokenizer):
    # load audio
    sample = batch["path"]

    inputs = feature_extractor(
        sample["array"], sampling_rate=sample["sampling_rate"])

    batch[feature_name] = getattr(inputs, feature_name)[0]
    # take length of raw audio waveform
    batch["input_length"] = len(sample["array"].squeeze())

    batch["labels"] = tokenizer(batch["sentence"], ).input_ids
    return batch


def is_audio_in_length_range(length):
    return length > min_input_length and length < max_input_length


def process_dataset(_datasets, feature_name, feature_extractor, tokenizer):
    _datasets = _datasets.remove_columns(
        ["accents", "variant", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    _datasets = _datasets.map(remove_special_characters)
    _datasets = _datasets.cast_column(
        'path', Audio(sampling_rate=16_000))
    _datasets = _datasets.map(
        lambda batch: prepare_dataset(
            batch, feature_name=feature_name, feature_extractor=feature_extractor, tokenizer=tokenizer),
        num_proc=8,
        remove_columns=_datasets.column_names
    )
    # filter data that is outside expected length
    _datasets = _datasets.filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )
    return _datasets


def compute_metrics(pred):

    pred_ids = pred.predictions[0]
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

    metrics = {k: v.compute(predictions=pred_str, references=label_str)
               for k, v in eval_metrics.items()}

    return metrics


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


if __name__ == '__main__':
    parser = HfArgumentParser((TrainingConfig, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()
    auth = args.hub_token
    repo_name = args.repo_name
    model = args.model
    processor = AutoProcessor.from_pretrained("urarik/"+repo_name)
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor
    feature_name = "input_values"
    if 'bert' in model:
        feature_name = "input_features"
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "urarik/"+repo_name, trust_remote_code=True)
    # feature_extractor = AutoFeatureExtractor.from_pretrained("urarik/"+repo_name,
    #                                                          trust_remote_code=True,
    #                                                          )

    max_input_length = args.max_duration_in_seconds * 16_000
    min_input_length = args.min_duration_in_seconds * 16_000

    train_data = args.train_data
    if train_data[-1] != "/":
        train_data += "/"
    test_data = args.test_data
    if test_data[-1] != "/":
        test_data += "/"

    train_df = pd.read_csv(
        train_data+args.train_data_config, sep="\t")
    test_df = pd.read_csv(
        test_data+args.test_data_config, sep="\t")

    train_df, _ = train_test_split(train_df, train_size=args.train_split)
    test_df, _ = train_test_split(test_df, train_size=args.test_split)

    common_voice_train = Dataset.from_pandas(train_df)
    common_voice_test = Dataset.from_pandas(test_df)

    common_voice_train = common_voice_train.map(
        lambda batch: {"path": train_data + batch["path"]})
    common_voice_test = common_voice_test.map(
        lambda batch: {"path": test_data + batch["path"]})

    common_voice_train = process_dataset(
        common_voice_train, feature_name, feature_extractor, tokenizer)
    common_voice_test = process_dataset(
        common_voice_test, feature_name, feature_extractor, tokenizer)

    # common_voice_train = load_dataset(
    # "mozilla-foundation/common_voice_16_0", "zh-CN", split="train", token=auth, trust_remote_code=True)
    # common_voice_test = load_dataset("mozilla-foundation/common_voice_16_0",
    #  "zh-CN", split="test", token=auth, trust_remote_code=True)

    print(len(common_voice_train), len(common_voice_test))

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, feature_extractor_input_name=feature_name
    )

    metrics = ["wer"]
    eval_metrics = {metric: evaluate.load(metric) for metric in metrics}

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)

    # Update the config dictionary
    config.update(vars(model_config))

    # If you need to initialize the values dynamically
    config.update({
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "add_adapter": "bert" in model
    })

    model = AutoModelForCTC.from_pretrained(
        model,
        config=config,
        trust_remote_code=True,
    )

    # freeze encoder
    if "freeze_feature_extractor" in dir(model):
        model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=repo_name,
        group_by_length=True,
        gradient_checkpointing=True,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_token=auth
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    try:
        trainer.train()
    except BaseException as e:
        print(e)
    trainer.save_model(repo_name)
    if args.push_to_hub:
        trainer.push_to_hub()
