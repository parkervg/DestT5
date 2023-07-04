# Set up logging
import logging
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import json
from ast import literal_eval
from typing import List, Optional

import numpy as np
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..utils.args import ModelArguments
from ..utils.dataset import DataTrainingArguments, normalize, serialize_schema
from ..utils.trainer import EvalPrediction, Seq2SeqTrainer
from ..utils.spider import spider_get_input


def splash_get_input(
    question: str,
    serialized_schema: str,
    feedback: str,
    predicted_parse_explanation: List[str],
    predicted_parse_with_values: str,
    prefix: str,
    normalize_query: bool,
    include_explanation: bool = True,
    include_question: bool = True,
) -> str:
    """
    For Splash, we need to normalize input query too
    """
    _normalize = normalize if normalize_query else (lambda x: x)
    predicted_parse_with_values = _normalize(predicted_parse_with_values)
    out = ""
    if include_question:
        out += prefix + question.strip() + " || "
    out += predicted_parse_with_values + " || " + serialized_schema.strip()
    if include_explanation:
        out += " || " + " ".join(literal_eval(predicted_parse_explanation))
    out += " || " + feedback
    return out


def splash_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def splash_concept_get_target(
    gold_serialized_schema: str,
) -> str:
    return gold_serialized_schema[2:].strip()  # Index at 2: to cut off initial '|'


def splash_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    model_args: ModelArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    if data_training_args.use_gold_concepts:
        logger.warning("Using gold concepts!!!")

    prefix = (
        data_training_args.source_prefix
        if data_training_args.source_prefix is not None
        else ""
    )

    # Inputs for both spider_schema and spider are the same
    if data_training_args.shuffle_splash_feedback:
        print("GENERATING SHUFFLED FEEDBACK")
        np.random.seed(42)
        shuffled_feedback_idxs = np.random.permutation(
            list(range(len(batch["feedback"])))
        )
        if data_training_args.spider_eval_on_splash:
            raise ValueError(
                "Can't do spider_eval_on_splash with shuffle_splash_feedback. The model won't use the feedback"
            )
        else:
            inputs = [
                splash_get_input(
                    question=question,
                    feedback=batch["feedback"][shuffled_feedback_idx],
                    predicted_parse_explanation=predicted_parse_explanation,
                    predicted_parse_with_values=predicted_parse_with_values,
                    serialized_schema=serialized_schema,
                    prefix=prefix,
                    include_explanation=data_training_args.include_explanation,
                    include_question=data_training_args.include_question,
                    normalize_query=data_training_args.normalize_query,
                )
                for question, serialized_schema, shuffled_feedback_idx, predicted_parse_explanation, predicted_parse_with_values, db_id in zip(
                    batch["question"],
                    batch["serialized_schema"],
                    shuffled_feedback_idxs,
                    batch["predicted_parse_explanation"],
                    batch["predicted_parse_with_values"],
                    batch["db_id"],
                )
            ]
    elif data_training_args.shuffle_splash_question:
        print("GENERATING SHUFFLED QUESTIONS")
        np.random.seed(42)
        shuffled_feedback_idxs = np.random.permutation(
            list(range(len(batch["question"])))
        )
        if data_training_args.spider_eval_on_splash:
            raise ValueError(
                "Can't do spider_eval_on_splash with shuffle_splash_question. The model won't use the feedback"
            )
        else:
            inputs = [
                splash_get_input(
                    question=batch["question"][shuffled_feedback_idx],
                    feedback=feedback,
                    predicted_parse_explanation=predicted_parse_explanation,
                    predicted_parse_with_values=predicted_parse_with_values,
                    serialized_schema=serialized_schema,
                    prefix=prefix,
                    include_explanation=data_training_args.include_explanation,
                    include_question=data_training_args.include_question,
                    normalize_query=data_training_args.normalize_query,
                )
                for feedback, serialized_schema, shuffled_feedback_idx, predicted_parse_explanation, predicted_parse_with_values, db_id in zip(
                    batch["feedback"],
                    batch["serialized_schema"],
                    shuffled_feedback_idxs,
                    batch["predicted_parse_explanation"],
                    batch["predicted_parse_with_values"],
                    batch["db_id"],
                )
            ]
    else:
        if data_training_args.spider_eval_on_splash:
            inputs = [
                spider_get_input(
                    question=question,
                    serialized_schema=serialized_schema,
                    prefix=prefix,
                )
                for question, serialized_schema in zip(
                    batch["question"], batch["serialized_schema"]
                )
            ]
        else:
            inputs = [
                splash_get_input(
                    question=question,
                    feedback=feedback,
                    predicted_parse_explanation=predicted_parse_explanation,
                    predicted_parse_with_values=predicted_parse_with_values,
                    serialized_schema=serialized_schema,
                    prefix=prefix,
                    include_explanation=data_training_args.include_explanation,
                    include_question=data_training_args.include_question,
                    normalize_query=data_training_args.normalize_query,
                )
                for question, serialized_schema, feedback, predicted_parse_explanation, predicted_parse_with_values, db_id in zip(
                    batch["question"],
                    batch["serialized_schema"],
                    batch["feedback"],
                    batch["predicted_parse_explanation"],
                    batch["predicted_parse_with_values"],
                    batch["db_id"],
                )
            ]

    untruncated_inputs: list = tokenizer(inputs, truncation=False)
    print(
        f"{len(list(filter(lambda x: len(x.ids) > max_source_length, untruncated_inputs.encodings)))} out of {len(untruncated_inputs.encodings)} items will be truncated!"
    )

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    if data_training_args.task_type == "schema_prediction":
        if model_args.schema_prediction_model_type == "generative":
            """
            Cast targets as all concepts appearing in query
            """
            targets = [
                splash_concept_get_target(gold_serialized_schema=gold_serialized_schema)
                for db_id, gold_serialized_schema in zip(
                    batch["db_id"], batch["gold_serialized_schema"]
                )
            ]
        elif model_args.schema_prediction_model_type == "classification":
            raise NotImplementedError()

    else:
        targets = [
            splash_get_target(
                query=query,
                db_id=db_id,
                normalize_query=data_training_args.normalize_query,
                target_with_db_id=data_training_args.target_with_db_id,
            )
            for db_id, query in zip(batch["db_id"], batch["gold_parse"])
        ]

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def splash_add_serialized_schema(
    ex: dict, data_training_args: DataTrainingArguments
) -> dict:
    if (
        data_training_args.use_gold_concepts
        and data_training_args.task_type == "schema"
    ):
        raise ValueError(
            "Both use_gold_concepts and task_type == 'schema' can't be true"
        )
    gold_serialized_schema = None
    if data_training_args.use_serialization_file is not None:
        serialized_schema = data_training_args.question_to_serialization[
            (ex["question"], ex["db_id"], ex["feedback"])
        ]
    else:
        serialized_schema = serialize_schema(
            question=ex["question"],
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            db_column_names=ex["db_column_names"],
            db_table_names=ex["db_table_names"],
            schema_serialization_type=data_training_args.schema_serialization_type,
            schema_serialization_randomized=data_training_args.schema_serialization_randomized,
            schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
            normalize_query=data_training_args.normalize_query,
            use_gold_concepts=data_training_args.use_gold_concepts,
            query=ex["gold_parse"],
        )
    if data_training_args.task_type == "schema_prediction":
        gold_serialized_schema = serialize_schema(
            question=ex["question"],
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            db_column_names=ex["db_column_names"],
            db_table_names=ex["db_table_names"],
            schema_serialization_type=data_training_args.schema_serialization_type,
            schema_serialization_randomized=data_training_args.schema_serialization_randomized,
            schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
            normalize_query=data_training_args.normalize_query,
            use_gold_concepts=True,
            query=ex["gold_parse"],
        )
    return {
        "serialized_schema": serialized_schema,
        "gold_serialized_schema": gold_serialized_schema,
    }


class SplashTrainer(Seq2SeqTrainer):
    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode(
            [f["input_ids"] for f in features], skip_special_tokens=True
        )
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(
                label_ids != -100, label_ids, self.tokenizer.pad_token_id
            )
        decoded_label_ids = self.tokenizer.batch_decode(
            _label_ids, skip_special_tokens=True
        )
        metas = [
            {
                "gold_parse": x["gold_parse"],
                "question": x["question"],
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x["db_table_names"],
                "db_column_names": x["db_column_names"],
                "db_foreign_keys": x["db_foreign_keys"],
                "predicted_parse_with_values": x["predicted_parse_with_values"],
                # "predicted_parse": x["predicted_parse"],
                "predicted_parse_explanation": x["predicted_parse_explanation"],
                "feedback": x["feedback"],
                "context": context,
                "label": label,
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        assert len(metas) == len(predictions)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [
                    dict(**{"prediction": prediction}, **meta)
                    for prediction, meta in zip(predictions, metas)
                ],
                f,
                indent=4,
            )
        return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        if self.target_with_db_id:
            # Remove database id from all predictions
            predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        # TODO: using the decoded reference labels causes a crash in the spider evaluator
        # if self.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        # decoded_references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # references = [{**{"query": r}, **m} for r, m in zip(decoded_references, metas)]
        references = metas
        return self.metric.compute(predictions=predictions, references=references)
