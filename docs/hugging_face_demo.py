from transformers import \
    pipeline, \
    AutoTokenizer, \
    AutoModelForSequenceClassification, \
    TrainingArguments,  \
    Trainer
import evaluate
from datasets import load_dataset
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention.")


# 1. 加载数据集
dataset = load_dataset("imdb")
print(dataset["train"][0])


# 2. 数据预处理
# 选择预训练模型和tokenizer（与模型匹配
model_name = "bert-base-uncased"
# model_name = "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 定义预训练函数：分词，截断，填充
def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,  # 适配模型最大输入长度
        return_tensors="pt"
    )

# 批量处理数据集
tokenized_dataset = dataset.map(
    function=preprocess,
    batched=True,
    remove_columns=["text"] # 移除原始文本，只保留tokenizer返回值
)

# 重命名标签列，当前预训练模型要求标签列为labels而非label
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# imdb没有验证集，从训练集划分10％
tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]
test_dataset = dataset.map(
    function=preprocess,
    batched=True,
    remove_columns=["text"] # 移除原始文本，只保留tokenizer返回值
).rename_column("label", "labels")

# todo 转换成tensor？


# 3. 加载预训练模型
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=model_name,
    num_labels=2, # imdb情感2分类
    # ignore_mismatched_size=True # 忽略预训练模型与任务头尺寸不匹配
)
print(model)


# 4. 配置训练参数
training_args = TrainingArguments(
    # 基础路径配置
    output_dir="./imdb_bert_finetuned/model",  # 模型保存目录
    overwrite_output_dir=True,  # 覆盖已有目录

    # 训练
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,

    # 评估与日志
    eval_strategy="epoch",  # 每轮epoch评估一次，可选steps
    logging_strategy="epoch",   # 每轮epoch打印日志
    logging_dir="./imdb_bert_finetuned/logs",

    # 模型保存
    save_strategy="epoch",  # 每轮epoch保存一次模型
    save_total_limit=2, # 最多保存2个模型，节省外存

    # 其他优化
    fp16=True,  # 混合精度训练，加速训练+节省内存
    load_best_model_at_end=True,    # 训练结束后加载基于验证集指标的最优模型
    metric_for_best_model="accuracy"    # 以准确率判断最优模型
)


# 5. 定义评估指标
metric = evaluate.load("accuracy")

# 定义指标计算函数
def compute_metric(eval_pred):
    logits, labels = eval_pred
    # 从logits中取最大概率对应的类别
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(
        predictions=predictions,
        labels=labels
    )


# 6. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metric,
    tokenizer=tokenizer
)
trainer.train()

# 7. 评估模型
test_results = trainer.evaluate(test_dataset)
print(f"测试集准确率：{test_results['eval_accuracy']:.4f}")


# 8. 保存模型后加载推理
trainer.save_model("./imdb_bert_finetuned/final_model")
tokenizer.save_pretrained("./imdb_bert_finetuned/final_model")

# 加载预训练模型
classifier = pipeline(
    task="sentiment_analysis",
    model="./imdb_bert_finetuned/final_model",
    tokenizer="./imdb_bert_finetuned/final_model"
)

# 测试示例
test_text = "This movie is amazing! The acting is top-notch."
print(classifier(test_text))
