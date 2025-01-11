import pandas as pd
from predict2table import *

data = metrics

# Prepare data
rows = []
for model, details in data.items():
    row = {"Model": model}
    for case, metrics in details.items():
        row[f"{case}_MSE Batch"] = metrics["Predict/mse_batch"]
        row[f"{case}_MSE Regressive"] = metrics["Predict/mse_regressive"]
    rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)

# Set hierarchical headers
df = df.set_index("Model")
columns = pd.MultiIndex.from_tuples([
    ("y_w", "MSE Batch"), ("y_w", "MSE Regressive"),
    ("w_y", "MSE Batch"), ("w_y", "MSE Regressive"),
    ("4single", "MSE Batch"), ("4single", "MSE Regressive"),
])
df.columns = columns

# Control decimal places
df = df.round(2)

# Print table
print(df)

"""
                                                    y_w                      w_y                  4single               
                                              MSE Batch MSE Regressive MSE Batch MSE Regressive MSE Batch MSE Regressive
Model                                                                                                                   
vq_mh_y_w_ts112_lr0.003/2025-01-11-00-01-54     1936.46        5904.30   3429.42        7694.76   2127.65       33840.21
lstm_y_w_ts112_lr0.0005/2025-01-10-21-50-56     1114.03        1496.32   2020.89        2870.24   1477.79       34907.24
vq_y_w_ts112_lr0.001/2025-01-10-23-17-38        1939.85        5500.62   3533.72        6586.74   2018.94       29926.12
att_mh_y_w_ts112_lr0.0003/2025-01-10-23-15-54   1088.57        1082.98   1981.93        2110.90   6017.92       58717.93
att_y_w_ts112_lr0.003/2025-01-10-23-06-46       1115.52        1098.24   2023.63        2035.35   4857.97       49269.07
"""

# Export to LaTeX table
latex_table = df.to_latex(
    index=True, 
    caption="Comparison of MSE Metrics Across Datasets and Models",
    label="tab:mse_comparison",
    multicolumn=True,
    multirow=True,
)
with open("table.tex", "w") as f:
    f.write(latex_table)
