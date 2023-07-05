# DestT5 (Correcting Semantic Parses with Natural Language through Dynamic Schema Encoding)
Dataset & code for DestT5 (NLP for ConvAI, ACL 2023)

[Link to Paper 📝](https://arxiv.org/pdf/2305.19974.pdf)

![Model Diagram](./img/model_diagram.png)

If you use this dataset or repository, please cite the following paper:

```
@inproceedings{glenn2023correcting,
  author = {Parker Glenn, Parag Pravin Dakle, Preethi Raghavan},
  title = "Correcting Semantic Parses with Natural Language through Dynamic Schema Encoding",
  booktitle = "Proceedings of the 5th Workshop on NLP for Conversational AI",
  publisher = "Association for Computational Linguistics",
  year = "2023"
}
```

## Performance 
Below we display the exact-match (EM%) and execution-accuracy (EX%) of DestT5 on the [SPLASH dataset](https://github.com/MSR-LIT/Splash), as well as the auxiliary test sets available in the [NLEdit codebase](https://github.com/MSR-LIT/NLEdit).
<table>
<thead>
  <tr>
    <th></th>
    <th></th>
    <th>Seq2Struct (SPLASH)</th>
    <th>EditSQL</th>
    <th>TaBERT</th>
    <th>RAT-SQL</th>
    <th>T5-Large</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">DestT5 (<a href="https://huggingface.co/parkervg/destt5-schema-prediction">parkervg/destt5-schema-prediction</a> with <a href="https://huggingface.co/parkervg/destt5-text2sql"> parkervg/destt5-text2sql</a>)</td>
    <td>EM%</td>
    <td>53.43</td>
    <td>31.82</td>
    <td>31.47</td>
    <td>28.37</td>
    <td>26.1</td>
  </tr>
  <tr>
    <td>EX%</td>
    <td>56.86</td>
    <td>40.3</td>
    <td>28.84</td>
    <td>36.53</td>
    <td>30.43</td>
  </tr>
</tbody>
</table>

## T5-large Dataset
The file [data/splash-t5-3vnuv1vf.json](./data/splash-t5-3vnuv1vf.json) contains 112 annotations for interactive semantic parsing.

Given randomly selected errors on the [Spider](https://github.com/taoyds/spider) dataset by [tscholak/3vnuv1vf](https://huggingface.co/tscholak/3vnuv1vf), natural language feedback is given to correct the erroneous parse.  


## Model Training 

Our codebase is based off the great implementation of [Picard](https://github.com/ServiceNow/picard). Specifically, we make the following updates to the `DataTrainingArguments` at [seq2seq/utils/dataset.py](./seq2seq/utils/dataset.py) to re-create the experiments described in the paper.


```python
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
    metadata={"help": "Whether we're running a Spider model on SPLASH. Only use question, in that case."},
)
```

To run the training for DestT5, run the following command.

```
python run_seq2seq.py ./configs/question/text2sql-t5-base-schema-generator.json
```

