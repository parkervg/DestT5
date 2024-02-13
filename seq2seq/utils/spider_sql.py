from attr import attrs, attrib
import json
from pathlib import Path
from sqlite3 import OperationalError

from third_party.spider.process_sql import get_sql, tokenize, get_tables_with_alias, parse_sql
from .get_gold_concepts import get_gold_concepts, consolidate_gold_concepts

def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql, toks

def get_schemas_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    db_names = [db['db_id'] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db['db_id']
        schema = {} #{'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db['column_names_original']
        table_names_original = db['table_names_original']
        tables[db_id] = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables

class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, schema, idMap: dict = None):
        self._schema = schema
        if idMap is None:
            self._idMap = self._map(self._schema)
        else:
            self._idMap = idMap

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {"*": "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = (
                    "__" + key.lower() + "." + val.lower() + "__"
                )
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap

def get_schema_from_table_info(table_info: dict):
    schema = {}
    idMap = {}
    for idx, (table_id, colname) in enumerate(table_info["column_names_original"]):
        colname = colname.lower()
        if colname == "*":
            idMap[colname] = idx
            continue
        table_name = table_info["table_names_original"][table_id].lower()
        if table_name not in schema:
            schema[table_name] = []
            idMap[table_name] = table_id
        schema[table_name].append(colname)
        idMap[table_name + "." + colname] = idx
    return Schema(schema, idMap=idMap)


@attrs
class SpiderSQL:
    data_dir: str = attrib()
    db_path_fmt: str = attrib()
    table_json_path: str = attrib(default=None)

    def __attrs_post_init__(self):
        self.data_dir = Path(self.data_dir)
        if (
                self.table_json_path is None
        ):  # Assume it's just in data_dir as 'tables.json'
            self.table_json_path = self.data_dir / "tables.json"
        else:
            self.table_json_path = Path(self.table_json_path)
        if not self.table_json_path.is_file():
            raise ValueError(
                f"File {str(self.table_json_path)} does not exist! Create a tables.json file first.\nSee sql_toolkit/examples/create_tables_json.py for more details"
            )
        with open(self.table_json_path, "r") as fp:
            self.table_info = json.load(fp)
        if isinstance(self.table_info, dict):
            print(
                f"{self.table_json_path.name} is a dict, but it should be a list.\nConverting to list now and saving as {self.table_json_path.stem}_aslist{self.table_json_path.suffix}..."
            )
            self.table_info = [self.table_info]
            with open(
                    f"{self.table_json_path.stem}_aslist{self.table_json_path.suffix}", "w"
            ) as f:
                json.dump(self.table_info, f)
            self.table_json_path = (
                f"{self.table_json_path.stem}_aslist{self.table_json_path.suffix}"
            )
        self.db_id_to_info = {table["db_id"]: table for table in self.table_info}
        self.schemas, self.db_names, self.tables = get_schemas_from_json(
            self.table_json_path
        )
        self.all_attributes_for_entity_type = {}

    def get_db_path(self, db_id: str) -> str:
        return self.data_dir / self.db_path_fmt.format(db_id=db_id)

    def convert_to_json(self, db_path, query, question, db_id):
        """
        Calls the appropriate Spider scripts and fills in the remaining JSON fields.
        """
        out = {"query": query}

        # Get the schema of the db
        try:
            out["table"] = self.db_id_to_info[db_id]
        except OperationalError as e:
            print(db_path)
            print(e)

        schema = get_schema_from_table_info(self.db_id_to_info[db_id])

        # Convert sql to spider format
        sql_parsed, query_toks = get_sql(schema, query)

        out["sql"] = sql_parsed
        out["db_id"] = db_id
        out["query_toks"] = query_toks
        out["question"] = question
        out["question_toks"] = tokenize(question)

        return out

    def to_spider(self, query: str, db_id: str, question: str = None) -> dict:
        """
        Convert a SQL query to the Spider JSON format.
        JSON format is described in this script: https://github.com/taoyds/spider/blob/master/process_sql.py
        """
        return self.convert_to_json(
            db_path=self.get_db_path(db_id=db_id),
            query=query,
            question=question,
            db_id=db_id,
        )

    def to_gold_concepts(self, query: str, db_id: str):
        """
        Get all tables, columns, and values present in a given SQL query.

        Returns:
             {
                "db_column_names": {
                    "table_id": [],
                    "column_name": [],
                    "values": []
                }
                "db_table_names": db_table_names
            }
        """
        spider_json = self.convert_to_json(
            db_path=self.get_db_path(db_id), query=query, question="", db_id=db_id
        )
        table_id_to_columns = get_gold_concepts(
            spider_sql=spider_json["sql"], db_data=self.db_id_to_info[db_id]
        )
        return consolidate_gold_concepts(
            table_id_to_columns=table_id_to_columns,
            db_data=self.db_id_to_info[db_id],
        )


