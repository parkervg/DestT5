import json
from typing import Dict, List

def is_value(val):
    return isinstance(val, (int, float, str))

def format_val(val):
    if isinstance(val, str):
        return val
    if val.is_integer():
        return str(int(val))
    return str(val)

def add_table_id(table_id: int, out: Dict) -> Dict:
    if table_id == -1:  # Ignore '*'
        return out
    if table_id not in out:
        out[table_id] = {}
    return out


def add_cond_unit_concepts(cond_unit, db_data: Dict, out: Dict):
    if isinstance(cond_unit, str):  # "and", "or"
        return out
    val_unit = cond_unit[2]
    out = add_val_unit_concepts(
        val_unit=val_unit,
        db_data=db_data,
        out=out,
        values=[
            format_val(i)
            for i in [cond_unit[3], cond_unit[4]]
            if is_value(i)
        ],
    )
    for val in [cond_unit[3], cond_unit[4]]:
        if val is None:
            continue
        if isinstance(val, dict):
            return get_gold_concepts(spider_sql=val, db_data=db_data, out=out)
        elif is_value(val):
            continue
        if isinstance(val[1], list):
            out = add_val_unit_concepts(
                val_unit=val,
                db_data=db_data,
                out=out,
                values=[i for i in [val[3], val[4]] if is_value(i)],
            )
        else:
            out = add_col_unit_concepts(col_unit=val, db_data=db_data, out=out)
    return out


def add_val_unit_concepts(
    val_unit, db_data: Dict, out: Dict, values: List[str] = None
) -> Dict:
    if values is None:
        values = []
    for col_unit in val_unit[1:]:
        if col_unit is None:
            continue
        col_id = col_unit[1]
        table_id = db_data["column_names_original"][col_id][0]
        if table_id == -1:
            continue
        out = add_table_id(table_id, out)
        out[table_id][col_id] = values
    return out


def add_col_unit_concepts(col_unit, db_data: Dict, out: Dict) -> Dict:
    col_id = col_unit[1]
    table_id = db_data["column_names_original"][col_id][0]
    if table_id == -1:  # Ignore '*'
        return out
    if table_id not in out:
        out[table_id] = {}
    out[table_id][col_id] = []
    return out


def get_gold_concepts(spider_sql: dict, db_data: dict, out: Dict = None):
    """
    Given a query in the Spider json format, returns all tables/columns/values present in the query.
    Returns db_column_names: Dict[str, str],  db_table_names: List[str], to be fed as input to serialize_schema() function.
    Returns:

         # >>> "concert : name , stadium_id | stadium : stadium_id , name"
         {
            "db_table_names": ['car_makers', 'cars_data'],
            "table_id": [2, 5],
            "column_name": [[2, 'Maker'], [5, 'Year']],
            "values: [[], ['"1970"']]
        }
    """

    if out is None:
        out = {}

    # Recover from select
    for idx, (agg_id, val_unit) in enumerate(spider_sql["select"][1]):
        out = add_val_unit_concepts(val_unit=val_unit, db_data=db_data, out=out)

    # Handle table names, conditions
    cond_pointer = -1
    for idx, (table_type, table_id) in enumerate(spider_sql["from"]["table_units"]):
        if isinstance(table_id, dict):
            out = get_gold_concepts(spider_sql=table_id, db_data=db_data, out=out)
            continue
        if table_id not in out:
            out[table_id] = {}
        if cond_pointer < 0:
            cond_pointer += 1
            continue
        if len(spider_sql["from"]["conds"]) == 0:
            continue
        # try to get the cond unit
        cond_unit = spider_sql["from"]["conds"][cond_pointer]
        if cond_unit == "and":
            cond_pointer += 1
            cond_unit = spider_sql["from"]["conds"][cond_pointer]

        elif cond_unit == "or":
            cond_pointer += 1
            cond_unit = spider_sql["from"]["conds"][cond_pointer]

        out = add_cond_unit_concepts(cond_unit=cond_unit, db_data=db_data, out=out)
        cond_pointer += 1

    # Handle 'WHERE' clause
    if spider_sql["where"]:
        for idx, cond_unit in enumerate(spider_sql["where"]):
            out = add_cond_unit_concepts(cond_unit=cond_unit, db_data=db_data, out=out)

    # Handle 'GROUP BY' clause
    if spider_sql["groupBy"]:
        for idx, (col_unit) in enumerate(spider_sql["groupBy"]):
            out = add_col_unit_concepts(col_unit=col_unit, db_data=db_data, out=out)

    # Handle 'EXCEPT' clause
    if spider_sql["except"]:
        return get_gold_concepts(
            spider_sql=spider_sql["except"], db_data=db_data, out=out
        )

    # Handle 'HAVING' clause
    if spider_sql["having"]:
        for idx, cond_unit in enumerate(spider_sql["having"]):
            out = add_cond_unit_concepts(cond_unit=cond_unit, db_data=db_data, out=out)

    # Handle 'UNION' clause
    if spider_sql["union"]:
        return get_gold_concepts(
            spider_sql=spider_sql["union"], db_data=db_data, out=out
        )

    # Handle 'INTERSECT' clause
    if spider_sql["intersect"]:
        return get_gold_concepts(
            spider_sql=spider_sql["intersect"], db_data=db_data, out=out
        )

    # Handle 'ORDER BY' clause
    if spider_sql["orderBy"]:
        for idx, (val_unit) in enumerate(spider_sql["orderBy"][1]):
            for col_unit in val_unit[1:]:
                if col_unit is None:
                    continue
                out = add_col_unit_concepts(col_unit=col_unit, db_data=db_data, out=out)

    return out


def consolidate_gold_concepts(table_id_to_columns: Dict, db_data: Dict):
    """
    Takes output from get_gold_concepts, and cleans up to the structure that serialize_schema() in picard expects.
    """
    # Index of table_name in db_table_names needs to align with id of db_column_names["table_id"]
    db_column_names = {"table_id": [], "column_name": [], "values": []}
    db_table_names = []
    for table_id in table_id_to_columns:
        table_name = db_data["table_names_original"][table_id]
        if table_name not in db_table_names:
            db_table_names.append(db_data["table_names_original"][table_id])
            new_table_id = len(db_table_names) - 1  # Based on index of table_name
        else:
            new_table_id = db_table_names.index(table_name)
        db_column_names["table_id"].append(new_table_id)
        for idx, (column_id, values) in enumerate(
            table_id_to_columns[table_id].items()
        ):
            _table_id = db_data["column_names"][column_id][0]
            assert table_id == _table_id
            column_name = db_data["column_names_original"][column_id][1]
            db_column_names["column_name"].append(column_name)
            db_column_names["values"].append(values)
            if idx > 0:  # To account for the only value being "*"
                db_column_names["table_id"].append(new_table_id)
    return {"db_column_names": db_column_names, "db_table_names": db_table_names}
