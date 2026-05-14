| dataset | model_type | model_name | configuration | total_params | trainable_params | latency_ms_per_sample | latency_ms_per_batch | gpu_memory_peak_mb | batch_size | fp16 | device_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ag_news | distilbert | distilbert-base-uncased | freeze_lower_layers | 66560772 | 21461508 | 109.25 |  |  | 4 | False | CUDA not available |
| ag_news | bert | bert-base-uncased | bert_base | 109485316 | 109485316 | 245.263 | 490.5250840238296 |  | 2 | False | CUDA not available |
| sst2 | distilbert | distilbert-base-uncased | freeze_lower_layers | 66560258 | 21460994 | 101.875 |  |  | 4 | False | CUDA not available |
| sst2 | bert | bert-base-uncased | bert_base | 109483778 | 109483778 | 313.314 | 626.6286030295305 |  | 2 | False | CUDA not available |
| yelp_review_full | distilbert | distilbert-base-uncased | small_classifier | 66461957 | 66461957 | 116.325 |  |  | 4 | False | CUDA not available |
| yelp_review_full | bert | bert-base-uncased | bert_base | 109486085 | 109486085 | 320.407 | 640.8143830485642 |  | 2 | False | CUDA not available |
