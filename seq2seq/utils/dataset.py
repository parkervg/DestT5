import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Generator

from datasets import interleave_datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from sql_metadata import Parser
from transformers.training_args import TrainingArguments

from .bridge_content_encoder import get_database_matches
from .spider_sql import SpiderSQL


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    val_max_time: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum allowed time in seconds for generation of one example. This setting can be used to stop "
            "generation whenever the full generation exceeds the specified amount of time."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation or test examples to this "
            "value if set."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_beam_groups: int = field(
        default=1,
        metadata={
            "help": "Number of beam groups to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    diversity_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Diversity penalty to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of sequences to generate during evaluation. This argument will be passed to "
            "``model.generate``, which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    schema_serialization_type: str = field(
        default="peteshaw",
        metadata={
            "help": "Choose between ``verbose`` and ``peteshaw`` schema serialization."
        },
    )
    schema_serialization_randomized: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomize the order of tables."},
    )
    schema_serialization_with_db_id: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to add the database id to the context. Needed for Picard."
        },
    )
    schema_serialization_with_db_content: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use the database content to resolve field matches."
        },
    )
    normalize_query: bool = field(
        default=True,
        metadata={
            "help": "Whether to normalize the SQL queries with the process in the 'Decoupling' paper"
        },
    )
    target_with_db_id: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to add the database id to the target. Needed for Picard."
        },
    )
    ################################
    ##### Added by Parker ##########
    ################################
    use_gold_concepts: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to serialize input only with columns/tables/values present in the gold query."
        },
    )

    use_serialization_file: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "If specified, points to the output of a T5 concept prediction model. Uses predictions as serialization to current text-to-sql model"
        },
    )

    include_explanation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Boolean defining whether to serialize explanation in SPLASH training"
        },
    )

    include_question: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Boolean defining whether to serialize question in SPLASH training"
        },
    )

    splash_train_with_spider: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Boolean defining whether to interleave Spider train set with Splash train"
        },
    )

    shuffle_splash_feedback: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Test to see if model is actually using feedback, by running evaluation on test set with shuffled feedback"
        },
    )

    shuffle_splash_question: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Test to see if model is actually using question, by running evaluation on test set with shuffled questions"
        },
    )

    task_type: Optional[str] = field(
        default="text2sql",
        metadata={"help": "One of text2sql, schema_prediction"},
    )

    spider_eval_on_splash: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether we're running a Spider model on SPLASH. Only use question, in that case."
        },
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class DataArguments:
    dataset: str = field(
        metadata={
            "help": "The dataset to be used. Choose between ``spider``, ``cosql``, or ``cosql+spider``, or ``spider_realistic``, or ``spider_syn``, or ``spider_dk``."
        },
    )
    dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": "./seq2seq/datasets/spider",
            "splash": "./seq2seq/datasets/splash",
            "cosql": "./seq2seq/datasets/cosql",
            "spider_realistic": "./seq2seq/datasets/spider_realistic",
            "spider_syn": "./seq2seq/datasets/spider_syn",
            "spider_dk": "./seq2seq/datasets/spider_dk",
        },
        metadata={"help": "Paths of the dataset modules."},
    )
    spider_dataset_url: str = field(
        default="",
        metadata={"help": "Path of spider.zip"},
    )
    splash_dataset_url: str = field(
        default="",
        metadata={"help": "Path of splash.zip"},
    )
    metric_config: str = field(
        default="both",
        metadata={
            "help": "Choose between ``exact_match``, ``test_suite``, or ``both``."
        },
    )
    # we are referencing spider_realistic to spider metrics only as both use the main spider dataset as base.
    metric_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": "./seq2seq/metrics/spider",
            "splash": "./seq2seq/metrics/splash",
            "spider_schema": "./seq2seq/metrics/spider_schema",
            "splash_schema": "./seq2seq/metrics/splash_schema",
            "spider_realistic": "./seq2seq/metrics/spider",
            "cosql": "./seq2seq/metrics/cosql",
            "spider_syn": "./seq2seq/metrics/spider",
            "spider_dk": "./seq2seq/metrics/spider",
        },
        metadata={"help": "Paths of the metric modules."},
    )
    test_suite_db_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the test-suite databases."}
    )
    data_config_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to data configuration file (specifying the database splits)"
        },
    )
    test_sections: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Sections from the data config to use for testing"},
    )


@dataclass
class TrainSplit(object):
    dataset: Dataset
    schemas: Dict[str, dict]


@dataclass
class EvalSplit(object):
    dataset: Dataset
    examples: Dataset
    schemas: Dict[str, dict]


@dataclass
class DatasetSplits(object):
    train_split: Optional[TrainSplit]
    eval_split: Optional[EvalSplit]
    test_splits: Optional[Dict[str, EvalSplit]]
    schemas: Dict[str, dict]


def _get_schemas(examples: Dataset) -> Dict[str, dict]:
    schemas: Dict[str, dict] = dict()
    for ex in examples:
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = {
                "db_table_names": ex["db_table_names"],
                "db_column_names": ex["db_column_names"],
                "db_column_types": ex["db_column_types"],
                "db_primary_keys": ex["db_primary_keys"],
                "db_foreign_keys": ex["db_foreign_keys"],
            }
    return schemas


def _prepare_train_split(
    dataset: Dataset,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> TrainSplit:
    schemas = _get_schemas(examples=dataset)
    dataset = dataset.map(
        add_serialized_schema,
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    if data_training_args.max_train_samples is not None:
        dataset = dataset.select(range(data_training_args.max_train_samples))
    column_names = dataset.column_names
    dataset = dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    return TrainSplit(dataset=dataset, schemas=schemas)


def _prepare_eval_split(
    dataset: Dataset,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> EvalSplit:
    if (
        data_training_args.max_val_samples is not None
        and data_training_args.max_val_samples < len(dataset)
    ):
        eval_examples = dataset.select(range(data_training_args.max_val_samples))
    else:
        eval_examples = dataset
    schemas = _get_schemas(examples=eval_examples)
    eval_dataset = eval_examples.map(
        add_serialized_schema,
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    column_names = eval_dataset.column_names
    eval_dataset = eval_dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.val_max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    return EvalSplit(dataset=eval_dataset, examples=eval_examples, schemas=schemas)


def prepare_splits(
    dataset_dict: DatasetDict,
    data_args: DataArguments,
    training_args: TrainingArguments,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
    spider_dataset_dict: DatasetDict = None,
    spider_pre_process_function: Callable[
        [dict, Optional[int], Optional[int]], dict
    ] = None,
    spider_add_serialized_schema: Callable[[dict], dict] = None,
) -> DatasetSplits:
    train_split, eval_split, test_splits = None, None, None

    if training_args.do_train:
        train_split = _prepare_train_split(
            dataset_dict["train"],
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )
        if data_training_args.splash_train_with_spider:
            # should have about 17496 training instances now
            assert spider_dataset_dict is not None
            spider_train_split = _prepare_train_split(
                spider_dataset_dict["train"],
                data_training_args=data_training_args,
                add_serialized_schema=spider_add_serialized_schema,
                pre_process_function=spider_pre_process_function,
            )
            # Spider train split schemas has 140 keys, splash has 111
            # interleave train sets, but use spider schemas
            interleaved_datasets = interleave_datasets(
                [train_split.dataset, spider_train_split.dataset],
                probabilities=[0.65, 0.35],
                stopping_strategy="all_exhausted",
            )
            train_split.dataset = interleaved_datasets
            train_split.schemas = spider_train_split.schemas

    if training_args.do_eval:
        # import datasets
        # import pandas as pd
        # val_subset = datasets.Dataset.from_pandas(pd.DataFrame(data=[i for i in dataset_dict["validation"] if i["db_id"] == "world_1"][:5]))
        # dataset_dict["validation"]
        eval_split = _prepare_eval_split(
            dataset_dict["validation"],
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )

    if training_args.do_predict:
        test_splits = {
            section: _prepare_eval_split(
                dataset_dict[section],
                data_training_args=data_training_args,
                add_serialized_schema=add_serialized_schema,
                pre_process_function=pre_process_function,
            )
            for section in data_args.test_sections
        }
        test_split_schemas = {}
        for split in test_splits.values():
            test_split_schemas.update(split.schemas)

    schemas = {
        **(train_split.schemas if train_split is not None else {}),
        **(eval_split.schemas if eval_split is not None else {}),
        **(test_split_schemas if test_splits is not None else {}),
    }

    return DatasetSplits(
        train_split=train_split,
        eval_split=eval_split,
        test_splits=test_splits,
        schemas=schemas,
    )


def normalize(sql):
    """
    https://github.com/RUCKBReasoning/RESDSQL
    preprocessing.py
    """

    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])

        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                if in_quotation:
                    in_quotation = False
                else:
                    in_quotation = True

        return out_s

    # remove ";"
    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation
    def double2single(s):
        return s.replace('"', "'")

    def add_asc(s):
        pattern = re.compile(
            r"order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*"
        )
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")

        return s

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]

        tables_aliases = new_tables_aliases
        for k, v in tables_aliases.items():
            s = s.replace("as " + k + " ", "")
            s = s.replace(k, v)

        return s

    processing_func = lambda x: remove_table_alias(
        add_asc(lower(white_space_fix(double2single(remove_semicolon(x)))))
    )

    return processing_func(sql)


def serialize_schema(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    schema_serialization_type: str = "peteshaw",
    schema_serialization_randomized: bool = False,
    schema_serialization_with_db_id: bool = True,
    schema_serialization_with_db_content: bool = False,
    normalize_query: bool = True,
    use_gold_concepts: bool = False,
    query: str = None,
) -> str:
    if use_gold_concepts and not query:
        raise ValueError(
            "If use_gold_concepts is True, need to pass gold SQL query as well"
        )
    if schema_serialization_type == "verbose":
        db_id_str = "Database: {db_id}. "
        table_sep = ". "
        table_str = "Table: {table}. Columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    else:
        raise NotImplementedError

    def get_column_str(
        table_name: str, column_name: str, gold_values: List[str] = None
    ) -> str:
        column_name_str = column_name.lower() if normalize_query else column_name
        if schema_serialization_with_db_content:
            if use_gold_concepts:
                # Encode the gold values from query
                if gold_values:
                    return column_str_with_values.format(
                        column=column_name_str, values=value_sep.join(gold_values)
                    )
                else:
                    return column_str_without_values.format(column=column_name_str)
            else:
                matches = get_database_matches(
                    question=question,
                    table_name=table_name,
                    column_name=column_name,
                    db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
                )
                if matches:
                    return column_str_with_values.format(
                        column=column_name_str, values=value_sep.join(matches)
                    )
                else:
                    return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    if use_gold_concepts:
        # Filter down schema, only to those concepts included in gold SQL
        try:
            ssql = SpiderSQL(
                data_dir="seq2seq/datasets/spider/spider",
                db_path_fmt="database/{db_id}/{db_id}.sqlite"
            )
            items = ssql.to_gold_concepts(query, db_id=db_id)
            db_column_names = items.get("db_column_names")
            db_table_names = items.get("db_table_names")
        except Exception as e:
            print(e)
            print(f"ERROR: {question}")
    else:
        # Just use the full 'db_column_names', 'db_table_names' we passed into this function
        pass

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(
                        table_name=table_name, column_name=y[1], gold_values=y[2]
                    ),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                            db_column_names.get(
                                "values", [None] * len(db_column_names["column_name"])
                            ),
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    # print()
    # print("**************************************************************************")
    # print(query)
    # print(serialized_schema)
    # print("**************************************************************************")
    # print()
    return serialized_schema
