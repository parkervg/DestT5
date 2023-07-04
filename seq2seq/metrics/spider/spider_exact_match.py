"""Spider exact match metric."""
import json
from typing import Any, Dict

from third_party.spider import evaluation as spider_evaluation


def compute_exact_match_metric(predictions, references) -> Dict[str, Any]:
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[
                reference["db_id"]
            ] = spider_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )
    evaluator = spider_evaluation.Evaluator(
        references[0]["db_path"], foreign_key_maps, "match"
    )
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        _ = evaluator.evaluate_one(reference["db_id"], reference["query"], prediction)
    evaluator.finalize()
    return {
        "exact_match": evaluator.scores["all"]["exact"],
    }


if __name__ == "__main__":
    from pathlib import Path

    with open("splash_train/predictions_eval_None.json", "r") as f:
        d = json.load(f)
    predictions = [item["prediction"] for item in d]
    predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
    for item in d:
        item["query"] = item["gold_parse"]
        item["db_path"] = Path("../data/spider/database")
    print(len(predictions))
    print(len(d))
    print(compute_exact_match_metric(predictions, d))
