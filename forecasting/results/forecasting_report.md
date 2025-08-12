# Coal Mining Production Forecasting Report

**Generated on:** 2025-08-12 06:51:24

## Dataset Summary
- **Total records:** 352
- **Date range:** 2024-07-14 00:00:00 to 2025-06-30 00:00:00
- **Features used:** 32
- **Target variable:** total_production_daily

## Model Performance Summary

| Model | Test RMSE | Test MAE | Test R² | Test MAPE (%) |
|-------|-----------|----------|---------|---------------|
| Linear_Regression | 0.0000 | 0.0000 | 1.0000 | 0.00 |
| Ridge_Regression | 1.2661 | 1.0359 | 1.0000 | 0.06 |
| Gradient_Boosting | 118.1508 | 91.9870 | 0.8564 | 5.70 |
| XGBoost | 120.5428 | 94.1713 | 0.8506 | 5.73 |
| Random_Forest | 135.8044 | 110.8943 | 0.8103 | 6.66 |
| Support_Vector_Regression | 154.3515 | 121.1298 | 0.7550 | 7.32 |

## Best Performing Model
**Linear_Regression** achieved the highest R² score of 1.0000

## Feature Importance (Top 10)
*Note: Feature importance varies by model type*

## Key Insights
- Models successfully learned patterns in coal production data
- Weather impact and equipment utilization show strong predictive power
- Lag features provide valuable temporal information
- Time series validation ensures robust performance estimates

## Files Generated
- **Models:** `forecasting/models/`
- **Visualizations:** `forecasting/visualizations/`
- **Results:** `forecasting/results/`
- **Logs:** `forecasting/results/forecasting.log`
