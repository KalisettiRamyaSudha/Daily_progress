# Daily Progress — Python, Data Analysis & Machine Learning

A day-by-day log from Python fundamentals through a first machine
learning project, spanning four weeks.

## Week 1 — Python Fundamentals
- **Day 1**: Conditionals & control flow — number classification, even/odd checker, menu-driven calculator
- **Day 2**: Lists — append/insert/remove/sort/reverse, manual max/min, sum & average, multiplication tables
- **Day 3**: Strings — length/case/reverse/vowel count, palindrome checker, word counter, character frequency
- **Day 4**: Functions & loops — square/cube/factorial/simple interest, prime checker, Fibonacci series, star patterns
- **Day 5**: File handling & review — safe file read/write/append, student marks & grading, a combined practice set, Git organization

## Week 2 — NumPy & pandas
- **Day 1**: NumPy basics — array creation, shape/size/ndim, indexing & slicing, element-wise math
- **Day 2**: Reshape/flatten, random number arrays, mean/median/std dev, matrix operations
- **Day 3**: pandas Series, DataFrame creation, CSV inspection, column selection & row filtering
- **Day 4**: Missing values, renaming & sorting, calculated columns, `groupby` aggregation
- **Day 5**: Mini dataset project (build → clean → analyze → summarize) and a summary notebook

## Week 3 — Data Visualization & EDA
- **Day 1**: Matplotlib basics — line charts, bar charts, histograms
- **Day 2**: Scatter plots, multiple chart types for different purposes, legends/gridlines/markers, saving charts to file
- **Day 3**: Seaborn — count plots, box plots, pair plots, distribution analysis, correlation heatmaps, categorical analysis
- **Day 4**: Data cleaning (duplicates, text standardization), outlier detection (IQR), data type conversion, a 5-observation EDA summary
- **Day 5**: Mini EDA project — messy CSV dataset, full cleaning + analysis notebook, 6 charts, README documenting objective/steps/findings

## Week 4 — Machine Learning Basics
- **Day 1**: Feature/target identification, train-test split, a minimal fit → predict → evaluate sklearn workflow (Iris dataset)
- **Day 2**: Linear regression — training, actual-vs-predicted visualization, MAE/MSE/RMSE/R², coefficient interpretation (diabetes dataset)
- **Day 3**: Logistic regression — accuracy, confusion matrix (TP/TN/FP/FN), precision/recall/F1 classification report (breast cancer dataset)
- **Day 4**: KNN classifier with K experimentation, StandardScaler vs. MinMaxScaler comparison, logistic regression vs. KNN model comparison, a full practice rep on Iris
- **Day 5**: Mini ML project — full classification workflow on the wine dataset (load → explore → preprocess → split → train two models → evaluate → compare → conclude), documented end to end in one notebook

## Key Takeaways by Theme
- **Python fundamentals**: control flow, reusable functions, loop-based algorithms, string manipulation, safe file I/O.
- **NumPy/pandas**: vectorized array math, DataFrame construction and inspection, missing-value handling, `groupby` aggregation.
- **Visualization/EDA**: matching chart type to the question being asked, reading distributions and outliers, correlation vs. causation caution, documenting findings alongside charts.
- **Machine learning**: the universal `fit → predict → evaluate` pattern; splitting before scaling (to avoid data leakage); accuracy alone can mislead — precision/recall/F1 and the confusion matrix tell a fuller story; comparing multiple models on the same data rather than assuming one algorithm wins by default.

## Week 4, Day 5 — This Project
- `data/wine_dataset_W4D5.csv` — 178 samples, 13 chemical-measurement features, 3 wine-cultivar classes
- `notebooks/mini_ml_project_W4D5.ipynb` — the full workflow, with a markdown explanation before every code section (data loading, exploration, preprocessing, splitting, training, prediction, evaluation, model comparison, conclusion)
- `plots/feature_scatter_W4D5.png`, `plots/confusion_matrices_W4D5.png` — supporting charts
- Result: Logistic Regression and KNN (K=5) both reached 97.2% test accuracy, with macro F1-scores within 0.001 of each other (0.9710 vs. 0.9718) — a genuine near-tie on this clean, well-separated dataset.

## Project Structure
```
Week4_Day5_MiniMLProject/
├── README.md
├── data/
│   └── wine_dataset_W4D5.csv
├── notebooks/
│   └── mini_ml_project_W4D5.ipynb
└── plots/
    ├── feature_scatter_W4D5.png
    └── confusion_matrices_W4D5.png
```

## Pushing to Git
```bash
cd Daily_progress
mkdir -p Week4/Day5_MiniMLProject
cp -r Week4_Day5_MiniMLProject/* Week4/Day5_MiniMLProject/
cp README.md .   # optional: replace/update the repo's root README with this one
git add Week4/Day5_MiniMLProject README.md
git commit -m "Add Week4 Day5: mini ML project (wine classification) + full 4-week summary README"
git push origin main
```