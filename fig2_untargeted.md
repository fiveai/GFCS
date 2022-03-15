## Figure 2 - CDFs for untargeted attacks

The following calls can be used to reproduce the results from Fig.2 of the paper, representing the cumulative distribution functions of several *untargeted* black-box attacks, with one or four surrogate networks.
To obtain the results for GFCS, SimBA-ODS, P-RGF and ODS-RGF the scripts `blackbox_simbaODS.py` and `rgf_variants_pytorch.py` are used, which belong to this repository.
Instead, the results for SimBA++ and LeBA should be obtained using the modified version of the original LeBA repository, which can be found here **TODO**.
Finally, for SquareAttack **TODO**.


### Perform attacks using _one_ surrogate

**GFCS**
```
python blackbox_simbaODS.py --model_name vgg16_bn --smodel_name resnet152 --data_index_set vgg16_bn_batch0_2 --ODS --special_margin_gradient_option --revert_to_ODS_when_stuck --num_sample 2000 --num_step 20000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/untargeted/1surr/GFCS_vgg16target_resnet152surr_vgg16_bn_batch0+2_20000_12.269_2.0.pt
python blackbox_simbaODS.py --model_name resnet50 --smodel_name resnet152 --data_index_set resnet50_batch0_2 --ODS --special_margin_gradient_option --revert_to_ODS_when_stuck --num_sample 2000 --num_step 20000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/untargeted/1surr/GFCS_res50target_resnet152surr_resnet50_batch0+2_20000_12.269_2.0.pt
python blackbox_simbaODS.py --model_name inception_v3 --smodel_name resnet152 --data_index_set inceptionv3_batch0_2 --ODS --net_specific_resampling --special_margin_gradient_option --revert_to_ODS_when_stuck --num_sample 2000 --num_step 20000 --norm_bound 16.377 --step_size 2.0 --output experimental_results/2022/untargeted/1surr/GFCS_incep3target_resnet152surr_interp_inceptionv3_batch0+2_20000_16.377_2.0.pt
```

**SimBA-ODS**
```
python blackbox_simbaODS.py --model_name vgg16_bn --smodel_name resnet152 --data_index_set vgg16_bn_batch0_2 --ODS --num_sample 2000 --num_step 20000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/untargeted/1surr/simbaODS_vgg16target_resnet152surr_vgg16_bn_batch0+2_20000_12.269_2.0.pt
python blackbox_simbaODS.py --model_name resnet50 --smodel_name resnet152 --data_index_set resnet50_batch0_2 --ODS --num_sample 2000 --num_step 20000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/untargeted/1surr/simbaODS_res50target_resnet152surr_resnet50_batch0+2_20000_12.269_2.0.pt
python blackbox_simbaODS.py --model_name inception_v3 --smodel_name resnet152 --data_index_set inceptionv3_batch0_2 --ODS --net_specific_resampling --num_sample 2000 --num_step 20000 --norm_bound 16.377 --step_size 2.0 --output experimental_results/2022/untargeted/1surr/simbaODS_incep3target_resnet152surr_interp_inceptionv3_batch0+2_20000_16.377_2.0.pt
```

**P-RGF (default hyperparams)**
```
python rgf_variants_pytorch.py --method biased --dataprior --samples_per_draw 50 --model_name vgg16_bn --smodel_name resnet152 --data_index_set vgg16_bn_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.2 --output_dir experimental_results/2022/untargeted/1surr/ --output rgf_biased_dp_spd50_vgg16target_resnet152surr_vgg16_bn_batch0+2_20000_12.269_0.2.pt
python rgf_variants_pytorch.py --method biased --dataprior --samples_per_draw 50 --model_name resnet50 --smodel_name resnet152 --data_index_set resnet50_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.2 --output_dir experimental_results/2022/untargeted/1surr/ --output rgf_biased_dp_spd50_res50target_resnet152surr_resnet50_batch0+2_20000_12.269_0.2.pt7
python rgf_variants_pytorch.py --method biased --dataprior --net_specific_resampling --samples_per_draw 50 --model_name inception_v3 --smodel_name resnet152 --data_index_set inceptionv3_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 16.377 --step_size 0.2 --output_dir experimental_results/2022/untargeted/1surr/ --output rgf_biased_dp_spd50_incep3target_resnet152surr_interp_inceptionv3_batch0+2_20000_16.377_0.2.pt
```

**P-RGF (our hyperparams)**
```
python rgf_variants_pytorch.py --method biased --dataprior --samples_per_draw 10 --model_name vgg16_bn --smodel_name resnet152 --data_index_set vgg16_bn_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.5 --output_dir experimental_results/2022/untargeted/1surr/ --output rgf_biased_dp_spd10_vgg16target_resnet152surr_vgg16_bn_batch0+2_20000_12.269_0.5.pt
python rgf_variants_pytorch.py --method biased --dataprior --samples_per_draw 10 --model_name resnet50 --smodel_name resnet152 --data_index_set resnet50_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.5 --output_dir experimental_results/2022/untargeted/1surr/ --output rgf_biased_dp_spd10_res50target_resnet152surr_resnet50_batch0+2_20000_12.269_0.5.pt
python rgf_variants_pytorch.py --method biased --dataprior --net_specific_resampling --samples_per_draw 10 --model_name inception_v3 --smodel_name resnet152 --data_index_set inceptionv3_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 16.377 --step_size 0.5 --output_dir experimental_results/2022/untargeted/1surr/ --output rgf_biased_dp_spd10_incep3target_resnet152surr_interp_inceptionv3_batch0+2_20000_16.377_0.5.pt
```

**ODS-RGF**
```
python rgf_variants_pytorch.py --method ods --samples_per_draw 10 --model_name vgg16_bn --smodel_name resnet152 --data_index_set vgg16_bn_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.5 --output_dir experimental_results/2022/untargeted/1surr/ --output rgf_ods_spd10_vgg16target_resnet152surr_vgg16_bn_batch0+2_20000_12.269_0.5.pt
python rgf_variants_pytorch.py --method ods --samples_per_draw 10 --model_name resnet50 --smodel_name resnet152 --data_index_set resnet50_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.5 --output_dir experimental_results/2022/untargeted/1surr/ --output rgf_ods_spd10_res50target_resnet152surr_resnet50_batch0+2_20000_12.269_0.5.pt
python rgf_variants_pytorch.py --method ods --net_specific_resampling --samples_per_draw 10 --model_name inception_v3 --smodel_name resnet152 --data_index_set inceptionv3_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 16.377 --step_size 0.5 --output experimental_results/2022/untargeted/1surr/ --output rgf_ods_spd10_incep3target_resnet152surr_interp_inceptionv3_batch0+2_20000_16.377_0.5.pt
```


**SimBA++**  (from LeBA repo)

```
python LeBA10.py --corrected_normalisation_and_interp --mode=SimBA++ --model1=vgg16_bn --model2=resnet152 --data_index_set vgg16_bn_batch0_2 --out_dir simba++_vgg16_rn152 --max_distance 12.269 --num_sample 2000
python LeBA10.py --corrected_normalisation_and_interp --mode=SimBA++ --model1=resnet50 --model2=resnet152 --data_index_set resnet50_batch0_3 --out_dir simba++_resnet50_rn152_b0+3 --max_distance 12.269 --num_sample 2000
python LeBA10.py --corrected_normalisation_and_interp --mode=SimBA++ --model1=inception_v3 --model2=resnet152 --data_index_set inceptionv3_batch0_2 --out_dir simba++_inceptionv3_rn152 --max_distance 16.377 --num_sample 2000
```

Sanitise the results
```
python get_results_sanitised.py --method SimBA++ --log_path 2022/simba++_vgg16_rn152 --out_name SimBA++_vgg16target.pt
python get_results_sanitised.py --method SimBA++ --log_path 2022/simba++_resnet50_rn152_b0+3 --out_name SimBA++_resnet50target_batch0+3.pt
python get_results_sanitised.py --method SimBA++ --log_path 2022/simba++_inceptionv3_rn152 --out_name SimBA++_incep3target.pt
```



**LeBA**  (from LeBA repo)
"Train"
```
python LeBA10.py --corrected_normalisation_and_interp --mode=train --model1=vgg16_bn --model2=resnet152 --data_index_set vgg16_bn_batch4 --out_dir LeBAtrain_vgg16_rn152_batch4 --max_distance 12.269 --num_sample 1000
[crashes] python LeBA10.py --corrected_normalisation_and_interp --mode=train --model1=resnet50 --model2=resnet152 --data_index_set resnet50_batch4 --out_dir LeBAtrain_rn50_rn152_batch4 --max_distance 12.269 --num_sample 1000
python LeBA10.py --corrected_normalisation_and_interp --mode=train --model1=inception_v3 --model2=resnet152 --data_index_set inceptionv3_batch4 --out_dir LeBAtrain_inceptionv3_rn152_batch4 --max_distance 16.377 --num_sample 1000
```

"Test"
```
python LeBA10.py --corrected_normalisation_and_interp --mode=test --model1=vgg16_bn --model2=resnet152 --data_index_set vgg16_bn_batch0_2 --out_dir LeBAtest_vgg16_rn152_batch4train_batch0+2test --max_distance 12.269 --num_sample 2000 --pretrain_weight /data_ssd/luca.bertinetto/LeBA-mirror/LeBAtrain_vgg16_rn152_batch4/snapshot/resnet152_final.pth
python LeBA10.py --corrected_normalisation_and_interp --mode=test --model1=inception_v3 --model2=resnet152 --data_index_set inceptionv3_batch0_2 --out_dir LeBAtest_inceptionv3_rn152_batch4train_batch0+2test --max_distance 16.377 --num_sample 2000 --pretrain_weight /data_ssd/luca.bertinetto/LeBA-mirror/LeBAtrain_inceptionv3_rn152_batch4/snapshot/resnet152_final.pth
```

Sanitise the results
```
python get_results_sanitised.py --method LeBA --log_path 2022/LeBAtest_vgg16_rn152_batch4train_batch0+2test --out_name LeBA_vgg16target.pt
python get_results_sanitised.py --method LeBA --log_path 2022/LeBAtest_inceptionv3_rn152_batch4train_batch0+2test --out_name LeBA_incep3target.pt

python get_results_sanitised.py --method LeBA --log_path 2022/LeBAcheat_vgg16_rn152_batch4train_batch4test --out_name LeBAcheat_vgg16target.pt
python get_results_sanitised.py --method LeBA --log_path 2022/LeBAcheat_inceptionv3_rn152_batch4train_batch4test --out_name LeBAcheat_incep3target.pt
```



### Perform attacks using _four_ surrogates

**GFCS**
```
python blackbox_simbaODS.py --model_name vgg16_bn --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set vgg16_bn_batch0_2 --ODS --special_margin_gradient_option --revert_to_ODS_when_stuck --num_sample 2000 --num_step 20000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/untargeted/4surr/GFCS_vgg16target_r34v19d121mobv2surr_vgg16_bn_batch0+2_20000_12.269_2.0.pt
python blackbox_simbaODS.py --model_name resnet50 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set resnet50_batch0_2 --ODS --special_margin_gradient_option --revert_to_ODS_when_stuck --num_sample 2000 --num_step 20000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/untargeted/4surr/GFCS_res50target_r34v19d121mobv2surr_resnet50_batch0+2_20000_12.269_2.0.pt
python blackbox_simbaODS.py --model_name inception_v3 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set inceptionv3_batch0_2 --ODS  --special_margin_gradient_option --revert_to_ODS_when_stuck --net_specific_resampling --num_sample 2000 --num_step 20000 --norm_bound 16.377 --step_size 2.0 --output experimental_results/2022/untargeted/4surr/GFCS_incep3target_r34v19d121mobv2surr_interp_inceptionv3_batch0+2_20000_16.377_2.0.pt
```

**SimBA-ODS**
```
python blackbox_simbaODS.py --model_name vgg16_bn --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set vgg16_bn_batch0_2 --ODS --num_sample 2000 --num_step 20000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/untargeted/4surr/simbaODS_vgg16target_r34v19d121mobv2surr_vgg16_bn_batch0+2_20000_12.269_2.0.pt
python blackbox_simbaODS.py --model_name resnet50 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set resnet50_batch0_2 --ODS --num_sample 2000 --num_step 20000 --norm_bound 12.269 --step_size 2.0 --output experimental_results/2022/untargeted/4surr/simbaODS_res50target_r34v19d121mobv2surr_resnet50_batch0+2_20000_12.269_2.0.pt
python blackbox_simbaODS.py --model_name inception_v3 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set inceptionv3_batch0_2 --ODS --net_specific_resampling --num_sample 2000 --num_step 20000 --norm_bound 16.377 --step_size 2.0 --output experimental_results/2022/untargeted/4surr/simbaODS_incep3target_r34v19d121mobv2surr_interp_inceptionv3_batch0+2_20000_16.377_2.0.pt
```

**P-RGF (default hyperparams)**
```
python rgf_variants_pytorch.py --method biased --dataprior --samples_per_draw 50 --model_name vgg16_bn --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set vgg16_bn_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.2 --output_dir experimental_results/2022/untargeted/4surr/ --output rgf_biased_dp_spd50_vgg16target_r34v19d121mobv2surr_vgg16_bn_batch0+2_20000_12.269_0.2.pt
python rgf_variants_pytorch.py --method biased --dataprior --samples_per_draw 50 --model_name resnet50 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set resnet50_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.2 --output_dir experimental_results/2022/untargeted/4surr/ --output rgf_biased_dp_spd50_res50target_r34v19d121mobv2surr_resnet50_batch0+2_20000_12.269_0.2.pt
python rgf_variants_pytorch.py --method biased --dataprior --net_specific_resampling --samples_per_draw 50 --model_name inception_v3 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set inceptionv3_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 16.377 --step_size 0.2 --output_dir experimental_results/2022/untargeted/4surr/ --output rgf_biased_dp_spd50_incep3target_r34v19d121mobv2surr_interp_inceptionv3_batch0+2_20000_16.377_0.2.pt
```

**P-RGF (ours hyperparams)**
```
python rgf_variants_pytorch.py --method biased --dataprior --samples_per_draw 10 --model_name vgg16_bn --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set vgg16_bn_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.5 --output_dir experimental_results/2022/untargeted/4surr/ --output rgf_biased_dp_spd10_vgg16target_r34v19d121mobv2surr_vgg16_bn_batch0+2_20000_12.269_0.5.pt
python rgf_variants_pytorch.py --method biased --dataprior --samples_per_draw 10 --model_name resnet50 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set resnet50_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.5 --output_dir experimental_results/2022/untargeted/4surr/ --output rgf_biased_dp_spd10_res50target_r34v19d121mobv2surr_resnet50_batch0+2_20000_12.269_0.5.pt
python rgf_variants_pytorch.py --method biased --dataprior --net_specific_resampling --samples_per_draw 10 --model_name inception_v3 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set inceptionv3_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 16.377 --step_size 0.5 --output_dir experimental_results/2022/untargeted/4surr/ --output rgf_biased_dp_spd10_incep3target_r34v19d121mobv2surr_interp_inceptionv3_batch0+2_20000_16.377_0.5.pt
```

**ODS-RGF**
```
python rgf_variants_pytorch.py --method ods --samples_per_draw 10 --model_name vgg16_bn --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set vgg16_bn_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.5 --output_dir experimental_results/2022/untargeted/4surr/ --output rgf_ods_spd10_vgg16target_r34v19d121mobv2surr_vgg16_bn_batch0+2_20000_12.269_0.5.pt
python rgf_variants_pytorch.py --method ods --samples_per_draw 10 --model_name resnet50 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set resnet50_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 12.269 --step_size 0.5 --output_dir experimental_results/2022/untargeted/4surr/ --output rgf_ods_spd10_res50target_r34v19d121mobv2surr_resnet50_batch0+2_20000_12.269_0.5.pt
python rgf_variants_pytorch.py --method ods --net_specific_resampling --samples_per_draw 10 --model_name inception_v3 --smodel_name vgg19_bn resnet34 densenet121 mobilenet_v2 --data_index_set inceptionv3_batch0_2 --number_images 2000 --max_queries 20000 --norm_bound 16.377 --step_size 0.5 --output_dir experimental_results/2022/untargeted/4surr/ --output rgf_ods_spd10_incep3target_r34v19d121mobv2surr_interp_inceptionv3_batch0+2_20000_16.377_0.5.pt
```

----

### Generate plots data and figures

**Prepare results**

```
python plot_results.py --input experimental_results/2022/untargeted/1surr
python plot_results.py --input experimental_results/2022/untargeted/4surr
```

**Plot results**
```
python generate_results_summary.py --expm_json plots_setup/2022/untargeted/1surr/vgg16.json --color_palette 1 --save_to plots_paper/2022/untargeted/1surr/vgg16.pdf
python generate_results_summary.py --expm_json plots_setup/2022/untargeted/1surr/resnet50.json --color_palette 1 --save_to plots_paper/2022/untargeted/1surr/resnet50.pdf
python generate_results_summary.py --expm_json plots_setup/2022/untargeted/1surr/incep3.json --color_palette 1 --save_to plots_paper/2022/untargeted/1surr/incep3.pdf

python generate_results_summary.py --expm_json plots_setup/2022/untargeted/4surr/vgg16.json --color_palette 1 --save_to plots_paper/2022/untargeted/4surr/vgg16.pdf
python generate_results_summary.py --expm_json plots_setup/2022/untargeted/4surr/resnet50.json --color_palette 1 --save_to plots_paper/2022/untargeted/4surr/resnet50.pdf
python generate_results_summary.py --expm_json plots_setup/2022/untargeted/4surr/incep3.json --color_palette 1 --save_to plots_paper/2022/untargeted/4surr/incep3.pdf

```

