import copy
import re
from statistics import mean
from typing import Any, Dict, List

from sklearn.metrics import f1_score, precision_score, recall_score


def get_possible_concepts(ref: dict):
    all_concepts = []
    for table_id, column_name in zip(
        ref["db_column_names"]["table_id"], ref["db_column_names"]["column_name"]
    ):
        if table_id == -1:
            continue
        table_name = ref["db_table_names"][table_id]
        if table_name not in all_concepts:
            all_concepts.append(f"{ref['db_table_names'][table_id]}".lower())
        all_concepts.append(f"{table_name}:{column_name}".lower())
    return all_concepts


def one_hot_from_serialized(s: str, possible_concepts: List[str], db_id: str):
    out = [0] * len(possible_concepts)
    # Check to see if we can split by "|", remove db_id
    if all(x in s for x in ["|", ":"]) and s.index("|") < s.index(":"):
        s = "|".join(s.split("|")[1:]).strip()
    for chunk in s.split("|"):
        split_chunk = chunk.split(":")
        table_name = split_chunk[0].strip()
        column_names = [""]
        if len(split_chunk) > 1:
            column_names = re.split(
                r",\s*(?![^()]*\))", split_chunk[1]
            )  # Split by commas not in parentheses
        if not column_names[0].strip():
            if table_name in possible_concepts:
                out[possible_concepts.index(table_name)] = 1
            else:
                print(f"Invalid pred for {db_id}! {table_name}")
                out.append(1)
                continue
        else:
            for column_name in column_names:
                pred = f"{table_name}:{column_name.strip()}".lower()
                # TODO: should really generate values too, and track in F1 calculation
                # Swap out values
                pred_before_re = copy.deepcopy(pred)
                # Need to add a space before lookahead
                # since there's a column named official_ratings_(millions)
                removed_l_parentheses = re.search(r".* (?=\()", pred)
                if removed_l_parentheses:
                    pred = removed_l_parentheses.group().strip()
                removed_r_parentheses = re.search(r".* (?=\))", pred)
                if removed_r_parentheses:
                    pred = removed_r_parentheses.group().strip()
                # pred = re.sub(r"\([^)]*\)", "", pred).strip()
                if pred in possible_concepts:
                    out[possible_concepts.index(pred)] = 1
                else:
                    # print(f"Invalid pred for {db_id}! {pred}")
                    # print(s)
                    # print(pred_before_re)
                    # print()
                    # print()
                    out.append(1)
                    continue
    return out


def compute_multilabel_f1_metric(predictions, references) -> Dict[str, Any]:
    """
    predictions: List[str]
    references: List[dict]
    """
    all_f1_scores = []
    all_precision_scores = []
    all_recall_scores = []
    for pred, ref in zip(predictions, references):
        label = ref.get("label")
        db_id = ref.get("db_id")
        possible_concepts = get_possible_concepts(ref)
        one_hot_preds = one_hot_from_serialized(
            s=pred, possible_concepts=possible_concepts, db_id=db_id
        )
        one_hot_golds = one_hot_from_serialized(
            s=label, possible_concepts=possible_concepts, db_id=db_id
        )
        # Make sure arrays are equal in case of over-prediction
        while len(one_hot_preds) > len(one_hot_golds):
            one_hot_golds.append(0)
        curr_f1_score = f1_score(one_hot_golds, one_hot_preds)
        all_f1_scores.append(curr_f1_score)
        curr_precision_score = precision_score(one_hot_golds, one_hot_preds)
        all_precision_scores.append(curr_precision_score)
        curr_recall_score = recall_score(one_hot_golds, one_hot_preds)
        all_recall_scores.append(curr_recall_score)
    return {
        "f1_score": mean(all_f1_scores),
        "precision": mean(all_precision_scores),
        "recall": mean(all_recall_scores),
    }
