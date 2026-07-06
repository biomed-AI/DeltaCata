# Fine-tuning DeltaCata on augmented mutant-mutant training pairs

This directory contains the DeltaCata-ft model checkpoints used for the additional mutant-mutant evaluation described in Supplementary Note 1.

DeltaCata-DB defines mutation effects relative to the wildtype sequence. This wildtype-mutant setting is biologically relevant for enzyme engineering, where the goal is to prioritize variants with improved catalytic properties. However, this setting may also allow models to partially exploit wildtype-related priors, such as the tendency for more conserved or wildtype-like residues to be more favorable.

To examine this, we constructed an additional mutant-mutant evaluation set from the sequence-level test set. To better align the model with this setting, we also constructed an augmented mutant-mutant training set from the original training set and fine-tuned DeltaCata using a learning rate of 1 $\times$ 10<sup>-4</sup> for up to 15 epochs. This fine-tuned model, DeltaCata-ft, further improved performance on the mutant-mutant test sets while maintaining comparable performance on the original wildtype-mutant test sets.


### Performance comparison on the original wildtype-mutant and the derived mutant-mutant test sets

| Dataset | Method | PCC | SCC | R<sup>2</sup> | RMSE |
|:---|:---|---:|---:|---:|---:|
| $\Delta$*k*<sub>cat</sub> wildtype-mutant test set | UniKP | 0.077 | 0.067 | -0.258 | 1.257 |
|  | DEKP | 0.336 | 0.216 | -0.029 | 1.137 |
|  | CatPred | 0.319 | 0.232 | -0.194 | 1.224 |
|  | EITLEM-Kinetics* | 0.375 | 0.318 | 0.043 | 1.096 |
|  | DeltaCata | **0.568** | **0.461** | **0.314** | **0.928** |
|  | DeltaCata-ft | <u>0.559</u> | <u>0.444</u> | <u>0.307</u> | <u>0.933</u> |
| $\Delta$*k*<sub>cat</sub> mutant-mutant test set | UniKP | 0.075 | 0.079 | 0.005 | 1.006 |
|  | DEKP | 0.242 | 0.162 | -0.005 | 1.011 |
|  | CatPred | 0.146 | 0.114 | 0.019 | 0.999 |
|  | EITLEM-Kinetics* | 0.153 | 0.165 | -0.103 | 1.059 |
|  | DeltaCata | <u>0.374</u> | <u>0.305</u> | <u>0.125</u> | <u>0.943</u> |
|  | DeltaCata-ft | **0.438** | **0.351** | **0.188** | **0.909** |
| $\Delta$*K*<sub>m</sub> wildtype-mutant test set | UniKP | 0.058 | 0.014 | -0.135 | 0.757 |
|  | DEKP | 0.073 | 0.053 | -0.165 | 0.767 |
|  | CatPred | 0.118 | 0.127 | -0.125 | 0.754 |
|  | EITLEM-Kinetics* | 0.176 | 0.153 | -0.076 | 0.737 |
|  | DeltaCata | <u>0.364</u> | **0.288** | <u>0.111</u> | <u>0.670</u> |
|  | DeltaCata-ft | **0.372** | <u>0.282</u> | **0.124** | **0.665** |
| $\Delta$*K*<sub>m</sub> mutant-mutant test set | UniKP | -0.003 | -0.031 | -0.009 | 0.676 |
|  | DEKP | 0.032 | 0.046 | -0.088 | 0.702 |
|  | CatPred | 0.051 | 0.073 | <u>0.002</u> | <u>0.672</u> |
|  | EITLEM-Kinetics* | 0.016 | 0.048 | -0.138 | 0.718 |
|  | DeltaCata | <u>0.189</u> | <u>0.150</u> | -0.006 | 0.675 |
|  | DeltaCata-ft | **0.276** | **0.206** | **0.067** | **0.650** |

Note: The best and second-best PCC, SCC, R<sup>2</sup> and RMSE values are indicated by bold and underlined fonts, respectively. EITLEM-Kinetics* denotes the EITLEM-Kinetics architecture retrained without the original transfer learning strategy. DeltaCata-ft denotes DeltaCata fine-tuned on mutant-mutant training pairs constructed from the original training set.