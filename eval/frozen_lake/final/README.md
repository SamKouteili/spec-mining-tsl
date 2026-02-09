# FrozenLake Evaluation Results

This directory contains the final compiled results comparing SpecMining(TSLf), BehavClon(NN), DecTree(wLR), and Q-Learning on the FrozenLake environment.

## Files

### Plot
- `combined_all_methods_plot.png` - Main comparison plot

### Data Files
- `qlearning_results.json` - Q-Learning results (100-50k episodes)
- `qlearning_table.tex` - Q-Learning LaTeX table
- `bc_dt_high_samples.json` - BC/DT results for high sample counts (100-50k)
- `tslf_bc_dt_var_config.json` - TSLf/BC/DT results testing on var_config
- `tslf_bc_dt_fixed_to_var_size.json` - Fixed training → var_size testing
- `tslf_bc_dt_var_to_var_size.json` - Var training → var_size testing

### Scripts
- `plot_combined_results.py` - Script to regenerate the plot

## Visual Encoding

- **Line style**: dotted = test on var_config, solid = test on var_size
- **Markers**: circle (o) = train on fixed, diamond (◇) = train on var
- **Colors**: Blue = SpecMining, Orange = BehavClon, Green = DecTree, Red = Q-Learning

## Key Results Summary

### SpecMining(TSLf)
| Train → Test | Best Win Rate | Samples Needed |
|--------------|---------------|----------------|
| fixed → var_config | 100% | 8 demos |
| var → var_config | 100% | 8 demos |
| fixed → var_size | 100% | 4 demos |

### BehavClon(NN)
| Train → Test | Best Win Rate | Samples Needed |
|--------------|---------------|----------------|
| fixed → var_config | ~11% (degrades) | - |
| var → var_config | 82% | 50k demos |
| fixed → var_size | ~15% (degrades) | - |
| var → var_size | 69% | 50k demos |

### DecTree(wLR)
| Train → Test | Best Win Rate | Samples Needed |
|--------------|---------------|----------------|
| fixed → var_config | ~22% | - |
| var → var_config | 93% | 10k demos |
| fixed → var_size | ~20% | - |
| var → var_size | 86% | 50k demos |

### Q-Learning
| Train → Test | Best Win Rate | Episodes Needed |
|--------------|---------------|-----------------|
| fixed → var_config | 0% | - |
| var → var_config | 75% | 50k episodes |
| fixed → var_size | 0% | - |
| var → var_size | 41% | 50k episodes |

## Key Findings

1. **SpecMining achieves 100% with 8 demos** - perfect generalization
2. **BC overfits on fixed boards** - performance degrades with more data
3. **DT competitive at 93%** but needs 10k samples
4. **Q-Learning has zero transfer** from fixed training
5. **var_size harder than var_config** for all methods except SpecMining
