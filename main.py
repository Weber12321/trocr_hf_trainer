import os

import pytesseract

import wandb
import torch
import pandas as pd
from datasets import load_metric
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, default_data_collator



def scoring(n_gt_words, n_detected_words, n_match_words):
    if n_detected_words == 0:
        precision = 0
    else:
        precision = n_match_words / n_detected_words
    if n_gt_words == 0:
        recall = 0
    else:
        recall = n_match_words / n_gt_words

    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def calculate_match_words(pred_str, label_str):
    n_match_words = 0

    pred_words = list(pred_str.split())
    ref_words = list(label_str.split())
    n_gt_words = len(ref_words)
    n_detected_words = len(pred_words)

    for pred_w in pred_words:
        if pred_w in ref_words:
            n_match_words += 1
            ref_words.remove(pred_w)

    return n_gt_words, n_detected_words, n_match_words


def get_decode_str(pred_ids, label_ids, processor):
    pred_list = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_list = processor.batch_decode(label_ids, skip_special_tokens=True)

    return pred_list, label_list

class ZhPrintedDataset(Dataset):
    def __init__(self, root_dir, df, processor, tokenizer, max_target_length=20):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['image_path'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, tokenizer, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


def run_OCR():
    df = pd.read_csv('one_data.csv', encoding='utf-8')

    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-small-printed"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base-chinese")

    train_dataset = ZhPrintedDataset(root_dir='data/',
                                     df=df,
                                     processor=processor,
                                     tokenizer=tokenizer
                                     )
    eval_dataset = ZhPrintedDataset(root_dir='data/',
                                    df=df,
                                    processor=processor,
                                    tokenizer=tokenizer
                                    )

    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-small-printed")

    # set special tokens used for creating the decoder_input_ids from the labels
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = tokenizer.vocab_size

    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 15
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    os.environ['WANDB_PROJECT'] = 'trocr_trainer_tiny'

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        fp16=True,
        output_dir="./",
        num_train_epochs=20,
        logging_steps=10,
        save_steps=100,
        report_to="wandb",
        run_name="weber_one_data"
    )


    def prf_metric(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        # https://github.com/microsoft/unilm/blob/master/trocr/scoring.py
        pred_list, label_list = \
            get_decode_str(pred_ids, label_ids, processor)

        precision = 0.0
        recall = 0.0
        f1 = 0.0
        for pred, label in zip(pred_list, label_list):
            n_gt_words, n_detected_words, n_match_words = calculate_match_words(
                pred, label
            )
            p, r, f = scoring(n_gt_words, n_detected_words, n_match_words)
            precision += p
            recall += r
            f1 += f
        length = len(pred_list)

        return {"precision": precision / length, "recall": recall / length, "f1": f1 / length}

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=prf_metric,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    wandb.finish()
    trainer.save_model(output_dir='model')


def run_IAM():

    df = pd.read_fwf('data/IAM/gt_test.txt', header=None)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    del df[2]
    # some file names end with jp instead of jpg, let's fix this
    df['file_name'] = df['file_name'].apply(
        lambda x: x + 'g' if x.endswith('jp') else x)
    train_df, test_df = train_test_split(df, test_size=0.2)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    train_dataset = IAMDataset(root_dir='data/IAM/image/',
                               df=train_df,
                               processor=processor)
    eval_dataset = IAMDataset(root_dir='data/IAM/image/',
                              df=test_df,
                              processor=processor)

    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-stage1")

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    os.environ['WANDB_PROJECT'] = 'trocr_IAM'

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        fp16=True,
        output_dir="./",
        logging_steps=2,
        save_steps=1000,
        eval_steps=200,
        report_to = "wandb",
        run_name = "trocr_IAM_fine_tuning"
    )


    cer_metric = load_metric("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids,
                                           skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    wandb.finish()


def inference(img_path, ckpt):
    # config = VisionEncoderDecoderConfig.from_json_file(ckpt+"/config.json")
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-small-printed"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base-chinese")

    model = VisionEncoderDecoderModel.from_pretrained(
       "microsoft/trocr-small-printed"
    )

    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = \
        tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    print(len(generated_text))

if __name__ == '__main__':
    # run_OCR()
    img_path= "data/weber_2.jpg"
    ckpt = "model"
    inference(img_path, ckpt)