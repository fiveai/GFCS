**(WORK IN PROGRESS)**

This is the GitHub repository for the ICLR 2022 paper "[Attacking deep networks with surrogate-based adversarial black-box methods is easy](https://openreview.net/forum?id=Zf4ZdI4OQPV)", by Nicholas A. Lord, Romain Mueller, and Luca Bertinetto.

## Requirements

Please see `environment.yml`.

## Reproducing results
Consult the following sets of instructions to reproduce the results of the paper:
* [Main results: untargeted black-box attacks](fig2_untargeted.md) (Fig.2 in the paper) (this repo).
* [Targeted black-box attacks](fig5_targeted.md) (Fig.5 in the paper) (this repo).
* [On the importance of input-specific priors](fig4_input-specific_priors.md) (Fig.4 in the paper) (SimBA-PCA repo **COMING SOON**).

## Citation
```

@inproceedings{lord2022attacking,
    title={Attacking deep networks with surrogate-based adversarial black-box methods is easy},
    author={Nicholas A. Lord and Romain Mueller and Luca Bertinetto},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=Zf4ZdI4OQPV}
}

```

### Acknowledgement
The main method in this repository is based on the original implementation of SimBA-ODS (https://github.com/ermongroup/ODS).
We thank the authors for making their code available.

