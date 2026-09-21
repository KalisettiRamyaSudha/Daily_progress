# Week 2, Day 5 — Git Push Cleanup

## Commands to push this batch

```bash
cd Daily_progress          # your cloned repo
cp -r ~/Downloads/Week2_Tasks_D5/* Week2/Day5_MiniDatasetProject/  # adjust destination as needed
git add .
git commit -m "Add Week2 Day5: mini dataset project, summary notebook, git cleanup notes"
git push origin main
```

If you're keeping the flat `taskname_W2D#.py` naming instead of nested day
folders, just `cp` the files directly into wherever your Week 2 files live
and adjust the `git add` path to match.

## Files in this batch

| File | What it covers |
|---|---|
| `mini_dataset_project_W2D5.py` | Small sales dataset — build, clean (missing values, duplicates), add a calculated column, analyze with `groupby`, print a summary report. |
| `summary_notebook_W2D5.ipynb` | Runnable notebook documenting every NumPy/pandas concept from Week 2 (Days 1-4), each with a short explanation and a live example. |
| `README_W2D5.md` | This file. |

## What I learned this week (NumPy & pandas)

- **Array fundamentals** — creating 1D/2D arrays, checking shape/size/ndim, indexing and slicing rows, columns, and sub-matrices.
- **Array math** — element-wise addition, subtraction, multiplication, division, and the difference between `*` (element-wise) and `@`/`matmul` (true matrix multiplication).
- **Reshape vs. flatten** — `reshape()` usually returns a view into the same data (editing it can affect the original); `flatten()` always returns an independent copy.
- **Random arrays** — generating reproducible random integers and floats with `np.random.default_rng(seed=...)`, and inspecting their ranges.
- **Descriptive statistics** — computing mean, median, and standard deviation with NumPy, and confirming they match a from-scratch manual calculation.
- **pandas Series** — labeled 1D data, indexing by position vs. by label, and boolean filtering.
- **DataFrames** — building them from dictionaries and lists, inspecting with `head`/`tail`/`info`/`describe`, and reading real CSV files.
- **Column selection and row filtering** — single/multiple column selection, and filtering with one or multiple boolean conditions.
- **Missing values** — identifying with `isna()`, filling with a statistic (mean/median) or constant, and dropping rows/columns either fully or with a `thresh`.
- **Renaming and sorting** — `rename(columns=...)` and `sort_values()` on one or multiple columns, ascending or descending.
- **Calculated columns** — deriving new columns (totals, percentages, pass/fail flags) from existing ones.
- **Groupby aggregation** — counts and averages by category, and combining several aggregations at once with `.agg([...])`.

## Workflow note
Committing each day's work separately (rather than one big commit at the
end) kept the Git history readable and made it easy to see exactly what was
added and when.
