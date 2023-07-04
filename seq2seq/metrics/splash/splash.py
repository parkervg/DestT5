"""Spider concept metrics."""

from typing import Optional, Union

import datasets
from seq2seq.metrics.spider.spider_exact_match import compute_exact_match_metric

_DESCRIPTION = """
splash metrics.
https://aclanthology.org/2020.acl-main.187
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """\
@inproceedings{Elgohary20Speak,
    Title = {Speak to your Parser: Interactive Text-to-SQL with Natural Language Feedback},
    Author = {Ahmed Elgohary and Saghar Hosseini and Ahmed Hassan Awadallah},
    Year = {2020},
    Booktitle = {Association for Computational Linguistics},
}
@misc{zhong2020semantic,
  title={Semantic Evaluation for Text-to-SQL with Distilled Test Suites}, 
  author={Ruiqi Zhong and Tao Yu and Dan Klein},
  year={2020},
  eprint={2010.02840},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
"""

# _URL = (
#     "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"
# )


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Splash(datasets.Metric):
    def __init__(
        self,
        config_name: Optional[str] = None,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs
        )
        self.test_suite_db_dir: Optional[str] = kwargs.pop("test_suite_db_dir", None)

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": {
                        "gold_parse": datasets.Value("string"),
                        "question": datasets.Value("string"),
                        "context": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                        "db_path": datasets.Value("string"),
                        "db_table_names": datasets.features.Sequence(
                            datasets.Value("string")
                        ),
                        "db_column_names": datasets.features.Sequence(
                            {
                                "table_id": datasets.Value("int32"),
                                "column_name": datasets.Value("string"),
                            }
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
                    },
                }
            ),
            reference_urls=[""],
        )

    def _compute(self, predictions, references):
        for ref in references:
            # Convert keys, so spider metric will work
            ref["query"] = ref["gold_parse"]
        exact_match = compute_exact_match_metric(predictions, references)
        return exact_match
