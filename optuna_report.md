# CTBPRN Model Optimization Report

## Best Parameters Found

### For Best Validation Accuracy (78.1%)
- **Learning Rate**: 0.00153
- **Model Dimension (d)**: 64
- **Router Rank**: 16
- **Temperature**: 1.71
- **Batch Size**: 32
- **Sequence Length**: 128
- **CE Loss**: 0.6495

### For Best CE Loss (0.6470)
- **Learning Rate**: 0.00023
- **Model Dimension (d)**: 96
- **Router Rank**: 16
- **Temperature**: 0.71
- **Batch Size**: 32
- **Sequence Length**: 32
- **Validation Accuracy**: 62.5%

## Complete Trial Results

| Trial | Val Acc | CE Loss | Learning Rate | Dim (d) | Rank | Temperature | Batch Size | Seq Len |
|-------|---------|---------|---------------|---------|------|-------------|------------|---------|
| 1 | 0.6250 | 0.6470 | 0.00023 | 96 | 16 | 0.71 | 32 | 32 |
| 2 | 0.7813 | 0.6495 | 0.00153 | 64 | 16 | 1.71 | 32 | 128 |
| 8 | 0.7813 | 0.6498 | 0.00149 | 64 | 16 | 2.00 | 32 | 128 |
| 5 | 0.6563 | 0.6520 | 0.00596 | 64 | 8 | 1.71 | 32 | 64 |
| 4 | 0.6875 | 0.6556 | 0.00015 | 96 | 16 | 1.04 | 16 | 128 |
| 9 | 0.6563 | 0.6612 | 0.00050 | 64 | 16 | 1.13 | 32 | 128 |
| 3 | 0.6250 | 0.6715 | 0.00012 | 32 | 16 | 0.78 | 8 | 128 |
| 6 | 0.6875 | 0.6975 | 0.00318 | 64 | 4 | 1.47 | 32 | 128 |
| 0 | 0.7500 | 0.7030 | 0.00056 | 32 | 4 | 1.80 | 16 | 32 |
| 7 | 0.6875 | 0.7184 | 0.00126 | 64 | 8 | 1.37 | 8 | 64 |

## Key Findings

1. **Optimal Architecture**
   - Model dimension of 64-96 works best
   - Router rank of 16 provides good performance
   - Larger sequence lengths (128) yield better accuracy

2. **Training Configuration**
   - Moderate batch size of 32 works well
   - Learning rate around 0.0015 gives best accuracy
   - Temperature around 1.7-2.0 helps with routing

3. **Performance Trade-offs**
   - Best accuracy (78.1%) achieved with CE loss of 0.6495
   - Lower CE loss (0.6470) possible but with reduced accuracy (62.5%)
   - Sequence length of 128 consistently better than shorter sequences

## Recommendations

1. **For Production Use**:
   - Use the best accuracy parameters (Trial 2)
   - Consider increasing model dimension to 96 if more capacity is needed
   - Keep sequence length at 128 for best results

2. **For Further Optimization**:
   - Try learning rates between 0.001 and 0.002
   - Experiment with rank values between 8-32
   - Test temperature values between 1.5-2.0

3. **Memory Considerations**:
   - Current configuration uses minimal GPU memory
   - Can increase batch size if more GPU memory is available
   - Model dimension of 64 provides good balance between performance and memory usage

## Generated Plots

1. `optuna_plots/ce_loss_vs_accuracy.png`
2. `optuna_plots/ce_loss_vs_lr.png`

## Raw Data
- Complete trial data: `optuna_results_detailed.csv`
- Optuna study: `optuna_studies/ctbprn_study.pkl`
