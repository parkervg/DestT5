# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""splash"""


import json
import os
from typing import Any, Dict, Generator, List, Tuple

import datasets
from third_party.spider.preprocess.get_tables import dump_db_json_schema

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{Elgohary20Speak,
    Title = {Speak to your Parser: Interactive Text-to-SQL with Natural Language Feedback},
    Author = {Ahmed Elgohary and Saghar Hosseini and Ahmed Hassan Awadallah},
    Year = {2020},
    Booktitle = {Association for Computational Linguistics},
}
"""

_DESCRIPTION = """\

"""

_HOMEPAGE = "https://yale-lily.github.io/spider"

_LICENSE = "CC BY-SA 4.0"

_SPIDER_URL = "../spider/spider.zip"
_SPLASH_URL = "./splash.zip"


class Splash(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="splash",
            version=VERSION,
            description="splash",
        ),
    ]

    def __init__(
        self,
        *args,
        writer_batch_size=None,
        spider_dataset_url=_SPIDER_URL,
        splash_dataset_url=_SPLASH_URL,
        **kwargs,
    ) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)

        self.schema_cache = dict()
        self._spider_url = spider_dataset_url
        self._splash_url = splash_dataset_url

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "gold_parse": datasets.Value("string"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence(
                    {"column_id": datasets.Value("int32")}
                ),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
                "predicted_parse_with_values": datasets.Value("string"),
                # "predicted_parse": datasets.Value("string"),
                "predicted_parse_explanation": datasets.Value("string"),
                "feedback": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        spider_downloaded_filepath = dl_manager.download_and_extract(
            url_or_urls=self._spider_url
        )
        splash_downloaded_filepath = dl_manager.download_and_extract(
            url_or_urls=self._splash_url
        )
        logger.info("-" * 10)
        logger.info("Downloaded paths:")
        logger.info(f"{spider_downloaded_filepath}, {splash_downloaded_filepath}")
        logger.info(os.path.join(splash_downloaded_filepath, "splash/train.json"))
        logger.info("-" * 10)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join(splash_downloaded_filepath, "splash/train.json"),
                    ],
                    "db_path": os.path.join(
                        spider_downloaded_filepath, "spider/database"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join(splash_downloaded_filepath, "splash/test.json")
                    ],
                    "db_path": os.path.join(
                        spider_downloaded_filepath, "spider/database"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="ratsql",
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join(
                            splash_downloaded_filepath, "splash/splash-ratsql.json"
                        ),
                    ],
                    "db_path": os.path.join(
                        spider_downloaded_filepath, "spider/database"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="editsql",
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join(
                            splash_downloaded_filepath, "splash/splash-editsql.json"
                        )
                    ],
                    "db_path": os.path.join(
                        spider_downloaded_filepath, "spider/database"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="tabert",
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join(
                            splash_downloaded_filepath, "splash/splash-tabert.json"
                        )
                    ],
                    "db_path": os.path.join(
                        spider_downloaded_filepath, "spider/database"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="3vnuv1vf",
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join(
                            splash_downloaded_filepath, "splash/splash-t5-3vnuv1vf.json"
                        )
                    ],
                    "db_path": os.path.join(
                        spider_downloaded_filepath, "spider/database"
                    ),
                },
            ),
        ]

    def _generate_examples(
        self, data_filepaths: List[str], db_path: str
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath, encoding="utf-8") as f:
                splash = json.load(f)
                for idx, sample in enumerate(splash):
                    db_id = sample["db_id"]
                    if db_id not in self.schema_cache:
                        self.schema_cache[db_id] = dump_db_json_schema(
                            db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                        )
                    schema = self.schema_cache[db_id]
                    yield idx, {
                        "gold_parse": sample["gold_parse"],
                        "question": sample["question"],
                        "db_id": db_id,
                        "db_path": db_path,
                        "db_table_names": schema["table_names_original"],
                        "db_column_names": [
                            {"table_id": table_id, "column_name": column_name}
                            for table_id, column_name in schema["column_names_original"]
                        ],
                        "db_column_types": schema["column_types"],
                        "db_primary_keys": [
                            {"column_id": column_id}
                            for column_id in schema["primary_keys"]
                        ],
                        "db_foreign_keys": [
                            {"column_id": column_id, "other_column_id": other_column_id}
                            for column_id, other_column_id in schema["foreign_keys"]
                        ],
                        "predicted_parse_with_values": sample[
                            "predicted_parse_with_values"
                        ],
                        # "predicted_parse": sample["predicted_parse"],
                        "predicted_parse_explanation": sample[
                            "predicted_parse_explanation"
                        ],
                        "feedback": sample["feedback"],
                    }
