| dataset | ablation_config | freeze_mode | classifier_hidden_sizes | dropout | accuracy | precision_macro | recall_macro | f1_macro | precision_weighted | recall_weighted | f1_weighted | total_params | trainable_params | gpu_memory_peak_mb | latency_ms_per_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ag_news | baseline | none | [] | 0.2 | 0.5 | 0.222222 | 0.222222 | 0.222222 | 0.5 | 0.5 | 0.5 | 66956548 | 66956548 |  | 104.95 |
| ag_news | freeze_lower_layers | lower_layers | [256] | 0.2 | 0.75 | 0.375 | 0.5 | 0.428571 | 0.5625 | 0.75 | 0.642857 | 66560772 | 21461508 |  | 109.25 |
| ag_news | frozen_transformer | all | [] | 0.2 | 0.5 | 0.222222 | 0.222222 | 0.222222 | 0.5 | 0.5 | 0.5 | 66956548 | 593668 |  | 96.9 |
| ag_news | large_classifier | none | [512, 256] | 0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 66888964 | 66888964 |  | 108 |
| ag_news | small_classifier | none | [128] | 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 66461828 | 66461828 |  | 94.8 |
| sst2 | baseline | none | [] | 0.2 | 0.5 | 0.666667 | 0.666667 | 0.5 | 0.833333 | 0.5 | 0.5 | 66955010 | 66955010 |  | 111.125 |
| sst2 | freeze_lower_layers | lower_layers | [256] | 0.2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 66560258 | 21460994 |  | 101.875 |
| sst2 | frozen_transformer | all | [] | 0.2 | 0.5 | 0.666667 | 0.666667 | 0.5 | 0.833333 | 0.5 | 0.5 | 66955010 | 592130 |  | 156.2 |
| sst2 | large_classifier | none | [512, 256] | 0.3 | 0.75 | 0.375 | 0.5 | 0.428571 | 0.5625 | 0.75 | 0.642857 | 66888450 | 66888450 |  | 99.025 |
| sst2 | small_classifier | none | [128] | 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 66461570 | 66461570 |  | 118.25 |
| yelp_review_full | baseline | none | [] | 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 66957317 | 66957317 |  | 218.675 |
| yelp_review_full | freeze_lower_layers | lower_layers | [256] | 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 66561029 | 21461765 |  | 94.275 |
| yelp_review_full | frozen_transformer | all | [] | 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 66957317 | 594437 |  | 110.2 |
| yelp_review_full | large_classifier | none | [512, 256] | 0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 66889221 | 66889221 |  | 234.325 |
| yelp_review_full | small_classifier | none | [128] | 0.2 | 0.25 | 0.5 | 0.125 | 0.2 | 1 | 0.25 | 0.4 | 66461957 | 66461957 |  | 116.325 |
