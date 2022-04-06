## Figure 5 - CDFS for targeted attacks

The following calls can be used to reproduce the results from Fig.4 of the paper, representing the cumulative distribution functions of several *targeted* black-box attacks, with four surrogate networks.
To obtain the results for GFCS and SimBA-ODS the scripts `blackbox_simbaODS.py` is used, which belong to this repository.
Instead, for SquareAttack **TODO**.


### Perform attacks using _four_ surrogates

**GFCS**
```
python GFCS_main.py --model_name vgg16_bn --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set vgg16_bn_batch0 --GFCS --num_sample 1000 --num_step 30000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/targeted/4surr/targeted_GFCS_vgg16target_r34v19d121mobv2surr_vgg16_bn_batch0_30000_12.269_2.0.pt --targeted
python GFCS_main.py --model_name resnet50 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set resnet50_batch0 --GFCS --num_sample 1000 --num_step 30000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/targeted/4surr/targeted_GFCS_res50target_r34v19d121mobv2surr_resnet50_batch0_30000_12.269_2.0.pt --targeted
python GFCS_main.py --model_name inception_v3 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set inceptionv3_batch0 --ODS  --special_margin_gradient_option --revert_to_ODS_when_stuck --net_specific_resampling --num_sample 1000 --num_step 30000 --norm_bound 16.377 --step_size 2.0 --output experimental_results/2022/targeted/4surr/targeted_GFCS_incep3target_r34v19d121mobv2surr_interp_inceptionv3_batch0_30000_16.377_2.0.pt --targeted 
```

**SimBA-ODS**
```
python GFCS_main.py --model_name vgg16_bn --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set vgg16_bn_batch0 --ODS --num_sample 1000 --num_step 30000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/targeted/targeted_4surr/SimbaODS_vgg16target_r34v19d121mobv2surr_vgg16_bn_batch0_30000_12.269_2.0.pt --targeted
python GFCS_main.py --model_name resnet50 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set resnet50_batch0 --ODS --num_sample 1000 --num_step 30000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/targeted/4surr/targeted_SimbaODS_res50target_r34v19d121mobv2surr_resnet50_batch0_30000_12.269_2.0.pt --targeted 
python GFCS_main.py --model_name inception_v3 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set inceptionv3_batch0 --ODS --net_specific_resampling --num_sample 1000 --num_step 30000 --norm_bound 16.377 --step_size 2.0 --output experimental_results/2022/targeted/4surr/targeted_SimbaODS_incep3target_r34v19d121mobv2surr_interp_inceptionv3_batch0_30000_16.377_2.0.pt --targeted
```

**SimBA-ODS with --step-size 0.2**
```
python GFCS_main.py --model_name vgg16_bn --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set vgg16_bn_batch0 --ODS --num_sample 1000 --num_step 30000 --norm_bound 12.269 --step_size 0.2 --output experimental_results/2022/targeted/4surr/targeted_SimbaODS_vgg16target_r34v19d121mobv2surr_vgg16_bn_batch0_30000_12.269_0.2.pt --targeted
python GFCS_main.py --model_name resnet50 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set resnet50_batch0 --ODS --num_sample 1000 --num_step 30000 --norm_bound 12.269 --step_size 0.2 --output experimental_results/2022/targeted/4surr/targeted_SimbaODS_res50target_r34v19d121mobv2surr_resnet50_batch0_30000_12.269_0.2.pt --targeted 
python GFCS_main.py --model_name inception_v3 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set inceptionv3_batch0 --ODS --net_specific_resampling --num_sample 1000 --num_step 30000 --norm_bound 16.377 --step_size 0.2 --output experimental_results/2022/targeted/4surr/targeted_SimbaODS_incep3target_r34v19d121mobv2surr_interp_inceptionv3_batch0_30000_16.377_0.2.pt --targeted 
```

----

### Generate plots data and figures

**Prepare results**

```
python plot_results.py --input experimental_results/2022/targeted/4surr
```

**Plot results**

```
python generate_results_summary.py --expm_json plots_setup/2022/targeted/4surr/vgg16.json --save_to plots_paper/2022/targeted/4surr/vgg16.pdf
python generate_results_summary.py --expm_json plots_setup/2022/targeted/4surr/resnet50.json --save_to plots_paper/2022/targeted/4surr/resnet50.pdf
python generate_results_summary.py --expm_json plots_setup/2022/targeted/4surr/incep3.json --save_to plots_paper/2022/targeted/4surr/incep3.pdf
```

