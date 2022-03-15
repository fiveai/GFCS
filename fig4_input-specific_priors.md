## Figure 4 - On the importance of input-specific priors

The following calls can be used to reproduce the results from Fig.4 of the paper from the section "On the importance of input-specific priors".
Note that most of these scripts are located in the SimBA-PCA repo (**TODO**).

### Perform attacks

**SimBA-pixel**
(From SimBA-PCA repo)
```
python scripts_core/mufasa.py --attack_basis pixel --order random --epsilon 2.0 --model=vgg16_bn --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_common_224 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix simbapixel_vgg16_ --batch_size 50 --correct_samples_only --num_iters 20000 --norm_bound 12.269
python scripts_core/mufasa.py --attack_basis pixel --order random --epsilon 2.0 --model=resnet50 --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_common_224 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix simbapixel_resnet50_ --batch_size 50 --correct_samples_only --num_iters 20000 --norm_bound 12.269
python scripts_core/mufasa.py --attack_basis pixel --order random --epsilon 2.0 --model=inception_v3 --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_inception_299 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix simbapixel_inceptionv3_ --batch_size 50 --correct_samples_only --num_iters 20000 --norm_bound 16.377
```

**SimBA-DCT**
(From SimBA-PCA repo)
```
python scripts_core/mufasa.py --attack_basis dct --order random --freq_domain_width 28 --epsilon 2.0 --model=vgg16_bn --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_common_224 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix simbadct_vgg16_ --batch_size 50 --correct_samples_only --num_iters 20000 --norm_bound 12.269
python scripts_core/mufasa.py --attack_basis dct --order random --freq_domain_width 28 --epsilon 2.0 --model=resnet50 --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_common_224 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix simbadct_resnet50_ --batch_size 50 --correct_samples_only --num_iters 20000 --norm_bound 12.269
python scripts_core/mufasa.py --attack_basis dct --order random --freq_domain_width 38 --epsilon 2.0 --model=inception_v3 --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_inception_299 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix simbadct_inceptionv3_ --batch_size 50 --correct_samples_only --num_iters 20000 --norm_bound 16.377
```

**SimBA-PCA-images**
(From SimBA-PCA repo)

Generate adversaries
`python scripts_core/generate_adversaries.py --adversary_type normalised_data --model=resnet152 --dataset=imagenet_train --data_dir=/data_ssd/datasets/ilsvrc12 --data_transform=imagenet_common_224 --data_normalisation=imagenet_common --sample_count=8192 --result_dir=2022 --batch_size 25`

Compute dominant directions (for both image size 224 and 299)
```
python scripts_core/dominant_directions.py --data_folder 2022/generate_adversaries_raw_pert_resnet152_imagenet_train_normalised_data_8192_inf --output_folder=2022 --resampling_factor 0.3 --output_size 224 --output_suffix _R0.3_out224
python scripts_core/dominant_directions.py --data_folder 2022/generate_adversaries_raw_pert_resnet152_imagenet_train_normalised_data_8192_inf --output_folder=2022 --resampling_factor 0.3 --output_size 299 --output_suffix _R0.3_out299
```

Perform attacks
```
python scripts_core/mufasa.py --attack_basis enlo --order straight --epsilon 2.0 --basis_path 2022/generate_adversaries_raw_pert_resnet152_imagenet_train_normalised_data_8192_inf_R0.3_out224.pt --model vgg16_bn --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_common_224 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix mufasa_normalised_data_resnet152_ --save_suffix _D8192_R0.3_standard_simba_onlycorrect --batch_size 50 --correct_samples_only --norm_bound 12.269
python scripts_core/mufasa.py --attack_basis enlo --order straight --epsilon 2.0 --basis_path 2022/generate_adversaries_raw_pert_resnet152_imagenet_train_normalised_data_8192_inf_R0.3_out224.pt --model resnet50 --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_common_224 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix mufasa_normalised_data_resnet152_ --save_suffix _D8192_R0.3_standard_simba_onlycorrect --batch_size 50 --correct_samples_only --norm_bound 12.269
python scripts_core/mufasa.py --attack_basis enlo --order straight --epsilon 2.0 --basis_path 2022/generate_adversaries_raw_pert_resnet152_imagenet_train_normalised_data_8192_inf_R0.3_out299.pt --model inception_v3 --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_inception_299 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix mufasa_normalised_data_resnet152_ --save_suffix _D8192_R0.3_standard_simba_onlycorrect --batch_size 50 --correct_samples_only --norm_bound 16.377
```

**SimBA-PCA-gradients**
(From SimBA-PCA repo)

Generate adversaries
```python scripts_core/generate_adversaries.py --adversary_type raw_gradients --model=resnet152 --dataset=imagenet_train --data_dir=/data_ssd/datasets/ilsvrc12 --data_transform=imagenet_common_224 --data_normalisation=imagenet_common --sample_count=8192 --result_dir=2022 --batch_size 25```

Compute dominant directions (for both image size 224 and 299)
```
python scripts_core/dominant_directions.py --data_folder 2022/generate_adversaries_raw_pert_resnet152_imagenet_train_raw_gradients_8192_inf --output_folder=2022 --resampling_factor 0.3 --output_size 224 --output_suffix _R0.3_out224
python scripts_core/dominant_directions.py --data_folder 2022/generate_adversaries_raw_pert_resnet152_imagenet_train_raw_gradients_8192_inf --output_folder=2022 --resampling_factor 0.3 --output_size 299 --output_suffix _R0.3_out299
```

Perform attacks
```
python scripts_core/mufasa.py --attack_basis enlo --order straight --epsilon 2.0 --basis_path 2022/generate_adversaries_raw_pert_resnet152_imagenet_train_raw_gradients_8192_inf_R0.3_out224.pt --model vgg16_bn --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_common_224 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix mufasa_raw_gradients_resnet152_ --save_suffix _D8192_R0.3_standard_simba_onlycorrect --batch_size 50 --correct_samples_only --norm_bound 12.269
python scripts_core/mufasa.py --attack_basis enlo --order straight --epsilon 2.0 --basis_path 2022/generate_adversaries_raw_pert_resnet152_imagenet_train_raw_gradients_8192_inf_R0.3_out224.pt --model resnet50 --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_common_224 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix mufasa_raw_gradients_resnet152_ --save_suffix _D8192_R0.3_standard_simba_onlycorrect --batch_size 50 --correct_samples_only --norm_bound 12.269
python scripts_core/mufasa.py --attack_basis enlo --order straight --epsilon 2.0 --basis_path 2022/generate_adversaries_raw_pert_resnet152_imagenet_train_raw_gradients_8192_inf_R0.3_out299.pt --model inception_v3 --dataset imagenet_val --data_dir /data_ssd/datasets/ilsvrc12 --data_transform imagenet_inception_299 --data_normalisation imagenet_common --num_runs 1998 --subtract_competition --query_saver --attack_mode standard_simba --result_dir 2022 --save_prefix mufasa_raw_gradients_resnet152_ --save_suffix _D8192_R0.3_standard_simba_onlycorrect --batch_size 50 --correct_samples_only --norm_bound 16.377
```


### Generate plots data and figures

**Prepare results**

```
python plot_results.py --input experimental_results/2022/simbaPCA
```

**Plot results**

```
python generate_results_summary.py --expm_json plots_setup/2022/simbaPCA/vgg16.json --color_palette 1 --save_to plots_paper/2022/simbaPCA/vgg16.pdf
python generate_results_summary.py --expm_json plots_setup/2022/simbaPCA/resnet50.json --color_palette 1 --save_to plots_paper/2022/simbaPCA/resnet50.pdf
python generate_results_summary.py --expm_json plots_setup/2022/simbaPCA/incep3.json --color_palette 1 --save_to plots_paper/2022/simbaPCA/incep3.pdf
```