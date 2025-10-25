# Research Notes: Model choice for telematics UBI

## Why tree boosting for telematics/tabular
Telematics features (e.g., p95 speed, hard brake rate/100km, jerk quantiles, time-of-day shares) form a classic **tabular** regime. Large independent benchmarks show that **tree-based boosting** remains state-of-the-art on such data sizes, even after extensive hyperparameter search, while deep learning struggles with irregular target functions typical of tabular problems. See:  
- Grinsztajn et al., *Why do tree-based models still outperform deep learning on tabular data?* (NeurIPS Datasets & Benchmarks).

## Why CatBoost in particular
CatBoost’s contributions—**ordered boosting** and improved handling of categorical features—reduce prediction shift/target leakage and deliver strong accuracy across public datasets. These are core reasons CatBoost competes at the top among boosting libraries on tabular tasks. See:  
- Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*. NeurIPS 2018.  
- Dorogush et al., *CatBoost: gradient boosting with categorical features support*. arXiv:1810.11363.

## Monotonic constraints for regulatory alignment
Pricing/risk models in insurance benefit from **shape constraints** to encode domain priors (e.g., “more speeding exposure → higher risk”). CatBoost provides **monotone constraints** out-of-the-box, letting us enforce increasing or decreasing relationships per feature during training—a practical way to keep the model directionally consistent and easier to justify in filings. See:  
- CatBoost Documentation — *Monotonic constraints*.

(For context, other GBMs like XGBoost and LightGBM also support monotonic constraints; the concept is widely accepted in regulated modeling.)

## Evidence from UBI/telematics literature
Empirical studies using telematics show that **gradient boosting** approaches outperform simpler baselines for claim likelihood or frequency prediction, and integrate cleanly into actuarial pricing. Comparisons of GLMs vs boosting (e.g., XGBoost) on telematics signals often report meaningful gains. Our two-stage design (GBM risk → GLM-style pricing map) follows this practice. See:  
- Pesantez-Narvaez et al., *Predicting Motor Insurance Claims Using Telematics Data*. Risks (MDPI), 2019.  
- (Related) Meng et al., *Actuarial intelligence in auto insurance: UBI claim frequency with ML + actuarial distributions*, 2022.

## References
1. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A. (2018). *CatBoost: Unbiased boosting with categorical features.* NeurIPS.  
2. Dorogush, A.V., Ershov, V., Gulin, A. (2018). *CatBoost: gradient boosting with categorical features support.* arXiv:1810.11363.  
3. Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS Datasets & Benchmarks.  
4. CatBoost Documentation. *Monotonic constraints.*  
5. Pesantez-Narvaez, P., Guillen, M., Alcañiz, M. (2019). *Predicting Motor Insurance Claims Using Telematics Data.* Risks (MDPI).  
6. Meng, X. et al. (2022). *Actuarial intelligence in auto insurance: UBI claim frequency with ML + actuarial distributions.*
