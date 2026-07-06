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
|  | DeltaCata-ft | 0.559 | 0.444 | 0.307 | 0.933 |
| $\Delta$*k*<sub>cat</sub> mutant-mutant test set | UniKP | 0.075 | 0.079 | 0.005 | 1.006 |
|  | DEKP | 0.242 | 0.162 | -0.005 | 1.011 |
|  | CatPred | 0.146 | 0.114 | 0.019 | 0.999 |
|  | EITLEM-Kinetics* | 0.153 | 0.165 | -0.103 | 1.059 |
|  | DeltaCata | 0.374 | 0.305 | 0.125 | 0.943 |
|  | DeltaCata-ft | **0.438** | **0.351** | **0.188** | **0.909** |
| $\Delta$*K*<sub>m</sub> wildtype-mutant test set | UniKP | 0.058 | 0.014 | -0.135 | 0.757 |
|  | DEKP | 0.073 | 0.053 | -0.165 | 0.767 |
|  | CatPred | 0.118 | 0.127 | -0.125 | 0.754 |
|  | EITLEM-Kinetics* | 0.176 | 0.153 | -0.076 | 0.737 |
|  | DeltaCata | 0.364 | **0.288** | 0.111 | 0.670 |
|  | DeltaCata-ft | **0.372** | 0.282 | **0.124** | **0.665** |
| $\Delta$*K*<sub>m</sub> mutant-mutant test set | UniKP | -0.003 | -0.031 | -0.009 | 0.676 |
|  | DEKP | 0.032 | 0.046 | -0.088 | 0.702 |
|  | CatPred | 0.051 | 0.073 | 0.002 | 0.672 |
|  | EITLEM-Kinetics* | 0.016 | 0.048 | -0.138 | 0.718 |
|  | DeltaCata | 0.189 | 0.150 | -0.006 | 0.675 |
|  | DeltaCata-ft | **0.276** | **0.206** | **0.067** | **0.650** |

Note: The best PCC, SCC, R<sup>2</sup> and RMSE values are indicated by bold fonts. EITLEM-Kinetics* denotes the EITLEM-Kinetics architecture retrained without the original transfer learning strategy. DeltaCata-ft denotes DeltaCata fine-tuned on mutant-mutant training pairs constructed from the original training set.