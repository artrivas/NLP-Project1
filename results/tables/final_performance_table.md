| dataset | model_type | model_name | configuration | accuracy | precision_macro | recall_macro | f1_macro | precision_weighted | recall_weighted | f1_weighted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ag_news | distilbert | distilbert-base-uncased | freeze_lower_layers | 0.75 | 0.375 | 0.5 | 0.428571 | 0.5625 | 0.75 | 0.642857 |
| ag_news | bert | bert-base-uncased | bert_base | 0.75 | 0.375 | 0.5 | 0.428571 | 0.5625 | 0.75 | 0.642857 |
| sst2 | distilbert | distilbert-base-uncased | freeze_lower_layers | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| sst2 | bert | bert-base-uncased | bert_base | 0.75 | 0.75 | 0.833333 | 0.733333 | 0.875 | 0.75 | 0.766667 |
| yelp_review_full | distilbert | distilbert-base-uncased | small_classifier | 0.25 | 0.5 | 0.125 | 0.2 | 1 | 0.25 | 0.4 |
| yelp_review_full | bert | bert-base-uncased | bert_base | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
