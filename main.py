# This is an example of using parameter efficient fine-tuning (LoRA) on
# a sequence classification model using a downstream dataset.
#
# Low ranked adaptation (LoRA) reduces the number of parameters to train making it
# possible to run fine-tuning on commodity hardware with a smaller compute and
# memory footprint.
#
# Check out: https://github.com/huggingface/peft

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import numpy as np
from evaluate import evaluator
from peft import LoraConfig, AutoPeftModelForSequenceClassification, TaskType, get_peft_model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

if __name__ == '__main__':
    splits = [ "train", "test" ]

    # Load 'emotion' dataset from HuggingFace hub.
    # https://huggingface.co/datasets/dair-ai/emotion
    # We will select the train and test splits.
    dataset = { split: ds for split, ds in zip(splits, load_dataset("emotion", split=splits)) }

    # Tokenizer to convert text into tokens that model understands.
    tokenizer=AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = {}

    for split in splits:
        tokenized_dataset[split] = dataset[split].map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, return_tensors="pt"),
                                                      batched=True)
        tokenized_dataset[split].set_format("torch")
        tokenized_dataset[split] = tokenized_dataset[split].shuffle().select(range(500))

    # Load model
    # Add classifier with 6 labels at the end.
    model = AutoModelForSequenceClassification.from_pretrained('gpt2',
                                                               torch_dtype="auto",
                                                               num_labels=6,
                                                               id2label={
                                                                   0: "sadness",
                                                                   1: "joy",
                                                                   2: "love",
                                                                   3: "anger",
                                                                   4: "fear",
                                                                   5: "surprise",
                                                               },
                                                               label2id={
                                                                   "sadness": 0,
                                                                   "joy": 1,
                                                                   "love": 2,
                                                                   "anger": 3,
                                                                   "fear": 4,
                                                                   "surprise": 5,
                                                               })

    model.config.pad_token_id = model.config.eos_token_id
    print(model)

    task_evaluator = evaluator("text-classification")

    # Test model on dataset without any adaptation
    eval_results = task_evaluator.compute(model_or_pipeline=model, data=dataset["test"],
                           tokenizer=tokenizer,
                           label_mapping={
                               "sadness": 0,
                               "joy": 1,
                               "love": 2,
                               "anger": 3,
                               "fear": 4,
                               "surprise": 5,
                           })

    print("Evaluation result for model on dataset without any adaptation")
    print(eval_results)

    # Inference mode is set to false, since we will train the model.
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32,
                             lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    print("Reduction in model trainable parameters after low ranked adaptation:")
    print(model.print_trainable_parameters())

    # Use higher batch size (say 8 instead of 2) to speed up training, when running with at least 16 GB RAM
    training_args = TrainingArguments(
        output_dir="./data",
        learning_rate=1e-3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_results = trainer.evaluate()

    # Load saved PEFT model
    peft_model = AutoPeftModelForSequenceClassification.from_pretrained("./data/checkpoint-500",
                                        is_trainable=False,
                                        num_labels=6,
                                        id2label={
                                            0: "sadness",
                                            1: "joy",
                                            2: "love",
                                            3: "anger",
                                            4: "fear",
                                            5: "surprise",
                                        },
                                        label2id={
                                            "sadness": 0,
                                            "joy": 1,
                                            "love": 2,
                                            "anger": 3,
                                            "fear": 4,
                                            "surprise": 5,
                                        }
                                    )
    peft_model.config.pad_token_id = model.config.eos_token_id

    print(peft_model)

    # Wrap PEFT model with trainer for running evaluation. We will not actually run training again.
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    eval_results = trainer.evaluate()

    print("Evaluation result for model on dataset after low ranked adaptation:")
    print(eval_results)

    # Manually review predictions done by fine-tuned model.
    items_for_manual_review = tokenized_dataset["test"].select(
        [0, 1, 22, 31, 43, 292, 448, 487]
    )

    results = trainer.predict(items_for_manual_review)
    df = pd.DataFrame({
        "text": [item for item in items_for_manual_review['text']],
        "predictions": np.argmax(results.predictions, axis=1),
        "labels": [label for label in items_for_manual_review['label']],
    })
    df['text'] = df['text'].str.wrap(40)
    df.assign(text=df['text'].str.split(' ')).explode('text')
    print("Sampled predictions for manual review:")
    print(df)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
