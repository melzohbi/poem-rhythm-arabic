# This code has been adapted from the GitHub repository 'https://github.com/potamides/uniformers'
# Portions of the original code have been modified to fit the specific requirements
# of this project. Credit goes to the original authors for their contributions.
from uniformer_utils.utils import AbstractPoetryLMTrainer
from uniformer_utils.metrics import load_metric
import random
# import wandb
from functools import partial
from random import randrange
import torch
from numpy import where
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from transformers.utils import logging
from transformers import TrainerCallback
import numpy as np
from datasets import Dataset
from uniformer_utils.datasets import load_dataset
from tqdm import tqdm

from uniformer_utils.arud import some_tashkeel, convert_to_phones, convert_to_beat
import rapidfuzz.distance.Levenshtein as _Levenshtein
from statistics import mean

logger = logging.get_logger("transformers")


class MultiEvalCallback(TrainerCallback):
    def __init__(self, second_dataset):
        super().__init__()
        self.second_dataset = second_dataset

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs["trainer"]
        second_metrics = trainer.evaluate(eval_dataset=self.second_dataset)
        print("Evaluation on second dataset:", second_metrics)


def _add_special_tokens(tokenizer, texts):
    bos, eos = tokenizer.bos_token, tokenizer.eos_token
    return [bos + text + eos for text in texts]


# sample spans according to a geometric distribution
def sample_spans(tokens, budget_percentage, p, CorVs, avoid_multi_zeros=False):
    # Total number of tokens
    n = len(tokens)

    # Calculate the total masking budget in terms of tokens
    budget = int(n * (budget_percentage / 100.0))

    # Keep track of indices to be masked
    masked_indices = set()

    # Sample a span length from a geometric distribution
    span_length = np.random.geometric(p)

    # Ensure the span length doesn't exceed the remaining budget
    span_length = min(span_length, budget)

    if avoid_multi_zeros:
        found_valid = False
        for _ in range(100):
            candidate_start = random.randint(0, n - 1)
            candidate_end = min(candidate_start + span_length, n)
            if "00" not in CorVs[candidate_start:candidate_end]:
                start_index = candidate_start
                end_index = candidate_end
                found_valid = True
                break
        # If no valid span is found, randomly select a starting point even if it contains zeros
        if not found_valid:
            candidate_start = random.randint(0, n - 1)
            candidate_end = min(candidate_start + span_length, n)

    else:

        # Randomly select a starting point
        start_index = random.randint(0, n - 1)

        # Compute the span indices
        # Ensure span doesn't exceed sequence length
        end_index = min(start_index + span_length, n)

    # Add the indices to the masked set
    for i in range(start_index, end_index):
        masked_indices.add(i)

    # Create the subset of tokens based on the masked indices
    selected_tokens = [tokens[i] for i in sorted(masked_indices)]

    del tokens[start_index:start_index + span_length]

    chosen_words_corv = "".join(CorVs[start_index:start_index + span_length])
    tokens.insert(start_index, f"<extra_id_0>{chosen_words_corv}<extra_id_1>")

    tokens = " ".join(tokens)
    selected_tokens = " ".join(selected_tokens)

    return {
        "chosen_words": selected_tokens,
        "text_masked": tokens
    }

# Update the attention mask for ablation study


def update_attention_mask(input_ids, attention_mask):
    for i, seq in enumerate(input_ids):
        try:
            # Find start and end indices
            start_index = seq.index(259)+1
            end_index = seq.index(260)
            # Set values between start_index and end_index to 0
            if start_index < end_index:
                attention_mask[i][start_index:end_index +
                                  1] = [0] * (end_index - start_index + 1)
        except ValueError:
            # Handle the case where start_token_id or end_token_id are not in the list
            continue

    return attention_mask


def _tokenizer_arabic(examples, tokenizer, is_encoder_decoder=False, multiple=False, ablation=False):
    inputs, labels = list(), list()

    for id, verse in enumerate(examples['lines']):
        list_of_words = verse.split(" ")

        # select for one word
        try:
            if not multiple:
                chosen_word = random.choice(list_of_words)
                index = list_of_words.index(chosen_word)

                chosen_word_corv = examples['binary'][id].split(",")[index]

                list_of_words[index] = f"<extra_id_0>{chosen_word_corv}<extra_id_1>"

                text_masked = " ".join(some_tashkeel(list_of_words, p=0.2))
                chosen_word_text = f"{chosen_word}"
            else:
                # select for multiple words
                budget_percentage = 25
                p = 0.2
                result = sample_spans(
                    list_of_words, budget_percentage, p, examples['binary'][id].split(","), avoid_multi_zeros=True)

                text_masked = some_tashkeel(result["text_masked"], p=0.2)
                chosen_word_text = result["chosen_words"]

        except:
            logger.info(
                f"Error with verse: {verse} and word: {chosen_word_text}")
            text_masked = "هَكَذَا <extra_id_0>1010<extra_id_1> لَهُم"
            chosen_word_text = "قُلْنَاْ"

        inputs.append(text_masked + "<extra_id_2>")
        labels.append(chosen_word_text)

    if is_encoder_decoder:
        model_inputs = tokenizer(inputs, add_special_tokens=False)

        # added to update the attention mask for ablation study
        if ablation:
            model_inputs["attention_mask"] = update_attention_mask(
                model_inputs["input_ids"], model_inputs["attention_mask"])

        labels = tokenizer(labels)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    else:
        return tokenizer(_add_special_tokens(tokenizer, [i + l for i, l in zip(inputs, labels)]))


# This is a custom trainer class for our purpose
class PoetryArabicTrainer(AbstractPoetryLMTrainer):
    def __init__(
        self,
        model,
        batch_size=128,
        test_run=False,
        multiple_words=False,
        ablation=False,
        train_on='tash',
        num_train_epochs=3,
        **kwargs,
    ):

        super().__init__(
            model=model,
            batch_size=batch_size,
            test_run=test_run,
            eval_multiplier=5 if test_run else 75,
            num_train_epochs=num_train_epochs,
            **kwargs,
        )

        if model.config.is_encoder_decoder:
            data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        else:
            data_collator = DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        tash_train_data, tash_eval_data, apcd_train_data, apcd_eval_data = self.load_dataset(
            test_run,
            multiple_words,
            ablation
        )

        super(AbstractPoetryLMTrainer, self).__init__(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=data_collator,
            train_dataset=tash_train_data if train_on == 'tash' else apcd_train_data,
            eval_dataset={"eval_tash": tash_eval_data,
                          "eval_poetry": apcd_eval_data},
            # callbacks=[MultiEvalCallback(eval_data_2)],
            compute_metrics=partial(
                self.compute_metrics,
                batch_size,
            ),
            **self.trainer_args,
        )

    def save_state(self):
        super().save_state()

    def compute_metrics(self, bs, p):
        labels = p.label_ids[0] if isinstance(
            p.label_ids, tuple) else p.label_ids
        labels = self.decode(
            where(labels == -100, self.tokenizer.pad_token_id, labels), batch=True)
        preds = super().compute_metrics(p)

        # if using wandb
        # eval_table = wandb.Table(columns=["ex_id", "original word", "corv_original", "predicted word", "masked sentence",
        #                                   "corv_score", "lev_score", "t5-prp-fluency"])

        corv_string_list, preds_, masked_sentences_ = list(), list(), list()

        for idx, pred in tqdm(enumerate(preds)):
            tokenized = self.tokenizer.tokenize(pred)
            # mask_token = self.tokenizer.tokenize(" [MASK] ")
            corv_list = tokenized[tokenized.index(
                '<extra_id_0>')+1:tokenized.index('<extra_id_1>')]
            corv_string_list.append(
                self.tokenizer.convert_tokens_to_string(corv_list))
            preds[idx] = self.tokenizer.convert_tokens_to_string(
                tokenized[tokenized.index('<extra_id_2>')+1:])

            # adding these for the purpose of t5 loss evaluation
            preds_.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(
                "<extra_id_0> ") + tokenized[tokenized.index('<extra_id_2>')+1:]).replace("<pad>", ""))
            masked_sentence_ = tokenized[1:tokenized.index('<extra_id_0>')] + self.tokenizer.tokenize(
                " <extra_id_0> ") + tokenized[tokenized.index('<extra_id_1>')+1:tokenized.index('<extra_id_2>')]
            masked_sentences_.append(
                self.tokenizer.convert_tokens_to_string(masked_sentence_).replace("<pad>", ""))

        logger.info(
            f"Computing metrics with the arabic align metric")

        # convowel_score = load_metric("arabicalign", batch_size=bs, diac_model=self.diac_model).compute(
        #     predicted_words=preds, corv=corv_string_list)

        # modified with mt5 version to accept arabic text
        mt5coh_score = load_metric("t5coh", batch_size=bs).compute(
            texts=masked_sentences_, predicted_words=preds_)

        convowel_score = self.calculate_metric(preds, corv_string_list)

        # log to wandb if using wandb
        # for idx in random.sample(range(len(preds)), 100):
        #     eval_table.add_data(idx, labels[idx].replace("<pad>", ""), corv_string_list[idx], preds_[idx], masked_sentences_[idx],
        #                         convowel_score['corv_score'], convowel_score['lev_score'], mt5coh_score['t5-prp-fluency'])

        # if wandb.run:
        #     wandb.run.log({"eval_table": eval_table})

        return convowel_score | mt5coh_score

    def calculate_metric(self, predicted_words, corvs):

        processed_predicted_words = list()
        scores = list()
        lev_distances = list()

        for predicted_word, corv in zip(predicted_words, corvs):

            predicted_word = predicted_word.replace("<pad>", "")
            predicted_word = convert_to_beat(convert_to_phones(predicted_word))

            predicted_word = predicted_word.replace(",", "")
            corv = corv.replace(" ", "")

            scores.append(float(predicted_word == corv))

            score = _Levenshtein.normalized_similarity(predicted_word, corv)

            lev_distances.append(score)

            processed_predicted_words.append(predicted_word)

        # get a random number between 0 and len(processed_predicted_words)
        i = random.randint(0, len(processed_predicted_words) - 1)
        logger.info(
            f"Sample: predicted word pattern is {processed_predicted_words[i]} and original pattern is {corvs[i]}")

        output_dict = {
            "corv_score": mean(scores),
            "lev_score": mean(lev_distances)
        }

        return output_dict

    def patch_tokenizer(self):
        super().patch_tokenizer()
        if not self.tokenizer.additional_special_tokens:
            special = {
                "additional_special_tokens": [f"<extra_id_{idx}>" for idx in range(3)],
                'pad_token': '<pad>'
            }
            self.tokenizer.add_special_tokens(special)  # pyright: ignore
            self.model.resize_token_embeddings(len(self.tokenizer))

    def truncate_text(self, text, length):
        if len(text) > length:
            return text[:length] + text[length:].split(' ', 1)[0]
        return text

    def load_dataset(self, test, multiple_words, ablation):

        # logger.info(f"Loading the apcd arabic dataset for processing...")

        # raw_dataset = load_dataset(
        #     "csv", data_files="datasets/apcd_dataset.csv")

        # raw_dataset = raw_dataset['train']

        logger.info(f"Loading the arabic datasets for processing...")

        raw_dataset = load_dataset(
            "csv", data_files="datasets/tash_train.csv")
        raw_apcd_dataset = load_dataset(
            "csv", data_files="datasets/apcd_train.csv")

        # combine the two datasets

        # raw_eval_dataset = load_from_disk("arabic_poetry/arabic_full_diac_Tashkeelah_dataset.hf")

        raw_dataset['apcd_train'] = raw_apcd_dataset['train']
        raw_dataset['train'] = raw_dataset['train']

        # remove untash,phones columns
        # raw_dataset = raw_dataset.remove_columns(['untash', 'phones'])

        if test:
            raw_dataset = raw_dataset[:10000]
            raw_dataset = Dataset.from_dict(raw_dataset)

        # else:
        #     raw_dataset = raw_dataset.shuffle(seed=42).select(range(2000000))

        # truncate the dataset to 256 only
        raw_dataset = raw_dataset.map(lambda examples: {
            'lines': [self.truncate_text(text, 256) for text in examples['lines']]
        }, batched=True)

        raw_dataset = raw_dataset.filter(
            lambda example: len(example['lines'].split(" ")) > 4)

        # tokenizing the dataset.
        logger.info(f"Tokenizing dataset with _tokenizer_arabic")
        tokenized_dataset = raw_dataset.map(
            _tokenizer_arabic,
            batched=True,
            fn_kwargs={  # pyright: ignore
                "tokenizer": self.tokenizer,
                "is_encoder_decoder": self.model.config.is_encoder_decoder,
                "multiple": multiple_words,
                "ablation": ablation,
            },
            load_from_cache_file=False
        )

        tash_train_tokenized_dataset = tokenized_dataset['train']
        apcd_train_tokenized_dataset = tokenized_dataset['apcd_train']
        # test_tokenized_dataset = tokenized_dataset['test']

        tash_train_tokenized_dataset, tash_eval_tokenized_dataset = tash_train_tokenized_dataset.train_test_split(
            test_size=3500, seed=42).values()
        apcd_train_tokenized_dataset, apcd_eval_tokenized_dataset = apcd_train_tokenized_dataset.train_test_split(
            test_size=3500, seed=42).values()

        index = randrange(len(tash_train_tokenized_dataset))
        sample = tash_train_tokenized_dataset[index]
        detokenized = self.decode(sample["input_ids"])
        logger.info(
            f"Input sample {index} of the training set: {sample['input_ids']}")
        logger.info(
            f"Input sample {index} of the training set (detokenized): {detokenized}")
        if "labels" in sample:  # pyright: ignore
            detokenized = self.decode(sample["labels"])
            logger.info(
                f"Label sample {index} of the training set: {sample['labels']}")
            logger.info(
                f"Label sample {index} of the training set (detokenized): {detokenized}")

        return tash_train_tokenized_dataset, tash_eval_tokenized_dataset, apcd_train_tokenized_dataset, apcd_eval_tokenized_dataset
