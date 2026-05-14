"""
week7_outlier_detection.py
---------------------------
Week 7 deliverable for the IDX Exchange MLS Analytics Internship.

Applies IQR-based outlier detection to key numeric fields in both the
sold and listed datasets. Following the handbook's tiered approach:

  Tier 1 — Business rule removals (already done in Week 4-5)
  Tier 2 — IQR flagging (this script)
  Tier 3 — Two output datasets: full flagged + clean filtered

The IQR method identifies statistical outliers using the formula:
  Lower bound = Q1 - (1.5 x IQR)
  Upper bound = Q3 + (1.5 x IQR)
  where IQR = Q3 - Q1

WHY WE FLAG RATHER THAN DELETE:
  A $50M Malibu estate is a real legitimate sale. Deleting it permanently
  would be wrong. But including it in a median price dashboard for a typical
  neighborhood would be misleading. Flagging lets analysts use the clean
  filtered dataset for most dashboards while preserving the full dataset
  for luxury market or competitive intelligence analysis.

Inputs  (from Week 6):
    output/crmls_sold_engineered.csv
    output/crmls_listed_engineered.csv

Outputs:
    output/crmls_sold_flagged.csv       <- full dataset with outlier flags
    output/crmls_listed_flagged.csv     <- full dataset with outlier flags
    output/crmls_sold_filtered.csv      <- outliers removed, clean analysis
    output/crmls_listed_filtered.csv    <- outliers removed, clean analysis
    output/outlier_report.csv           <- before/after comparison report

Usage:
    python week7_outlier_detection.py
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD ENGINEERED DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("LOADING ENGINEERED DATASETS")
print("=" * 65)

sold   = pd.read_csv("output/crmls_sold_engineered.csv",   low_memory=False)
listed = pd.read_csv("output/crmls_listed_engineered.csv", low_memory=False)

print(f"Sold   loaded: {len(sold):,} rows x {sold.shape[1]} cols")
print(f"Listed loaded: {len(listed):,} rows x {listed.shape[1]} cols")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFINE IQR FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def apply_iqr(df, col, label, multiplier=1.5):
    """
    Applies IQR outlier detection to a single numeric column.

    Steps:
      1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
      2. Calculate IQR = Q3 - Q1
      3. Set lower bound = Q1 - (multiplier x IQR)
      4. Set upper bound = Q3 + (multiplier x IQR)
      5. Flag any value outside those bounds as an outlier

    Parameters:
      df         — the dataframe to apply IQR to
      col        — the column name to check
      label      — human readable name for reporting
      multiplier — 1.5 is standard. Higher values are more lenient
                   (flag fewer records), lower values are stricter.

    Returns:
      df with a new outlier flag column added
      a dict with the IQR stats for the report
    """
    if col not in df.columns:
        print(f"  [{col}] not found, skipping")
        return df, None

    s = pd.to_numeric(df[col], errors="coerce").dropna()

    q1  = s.quantile(0.25)
    q3  = s.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - (multiplier * iqr)
    upper = q3 + (multiplier * iqr)

    # Create the outlier flag column
    # True = outlier (outside bounds), False = normal
    flag_col = f"outlier_{col.lower()}"
    df[flag_col] = (
        df[col].notna() &
        ((df[col] < lower) | (df[col] > upper))
    )

    n_outliers = df[flag_col].sum()
    pct        = n_outliers / len(df) * 100

    print(f"  {label}")
    print(f"    Q1={q1:,.2f}  Q3={q3:,.2f}  IQR={iqr:,.2f}")
    print(f"    Lower bound: {lower:,.2f}")
    print(f"    Upper bound: {upper:,.2f}")
    print(f"    Outliers flagged: {n_outliers:,} ({pct:.2f}%)")
    print()

    stats = {
        "field":       col,
        "label":       label,
        "q1":          round(q1, 2),
        "q3":          round(q3, 2),
        "iqr":         round(iqr, 2),
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2),
        "n_outliers":  n_outliers,
        "pct_outliers": round(pct, 3),
    }

    return df, stats

# ─────────────────────────────────────────────────────────────────────────────
# 3. APPLY IQR TO SOLD DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("IQR OUTLIER DETECTION — SOLD")
print("=" * 65)

sold_stats = []

# ClosePrice — the most important field to check for outliers
# A $50M estate or a $10K distressed sale both skew market averages
sold, stats = apply_iqr(sold, "ClosePrice",   "Close Price")
if stats:
    stats["dataset"] = "SOLD"
    sold_stats.append(stats)

# LivingArea — extreme values (1 sqft or 50,000 sqft) distort price-per-sqft
sold, stats = apply_iqr(sold, "LivingArea",   "Living Area (sqft)")
if stats:
    stats["dataset"] = "SOLD"
    sold_stats.append(stats)

# DaysOnMarket — extreme values (500+ days) distort market velocity metrics
sold, stats = apply_iqr(sold, "DaysOnMarket", "Days on Market")
if stats:
    stats["dataset"] = "SOLD"
    sold_stats.append(stats)

# Also apply to price_per_sqft since it's derived from two fields and
# can have extreme values when either ClosePrice or LivingArea is unusual
sold, stats = apply_iqr(sold, "price_per_sqft", "Price Per Sqft")
if stats:
    stats["dataset"] = "SOLD"
    sold_stats.append(stats)

# Master outlier flag — True if ANY of the above fields is flagged
# This is the single flag you'd use in Tableau to filter all outliers at once
sold_outlier_flags = [c for c in sold.columns if c.startswith("outlier_")]
sold["outlier_any"] = sold[sold_outlier_flags].any(axis=1)
print(f"  Total records flagged by ANY outlier: {sold['outlier_any'].sum():,} "
      f"({sold['outlier_any'].sum()/len(sold)*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. APPLY IQR TO LISTED DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("IQR OUTLIER DETECTION — LISTED")
print("=" * 65)

listed_stats = []

# ListPrice instead of ClosePrice for listings (most haven't closed yet)
listed, stats = apply_iqr(listed, "ListPrice",           "List Price")
if stats:
    stats["dataset"] = "LISTED"
    listed_stats.append(stats)

listed, stats = apply_iqr(listed, "LivingArea",          "Living Area (sqft)")
if stats:
    stats["dataset"] = "LISTED"
    listed_stats.append(stats)

listed, stats = apply_iqr(listed, "DaysOnMarket",        "Days on Market")
if stats:
    stats["dataset"] = "LISTED"
    listed_stats.append(stats)

listed, stats = apply_iqr(listed, "list_price_per_sqft", "List Price Per Sqft")
if stats:
    stats["dataset"] = "LISTED"
    listed_stats.append(stats)

# Master outlier flag for listed
listed_outlier_flags = [c for c in listed.columns if c.startswith("outlier_")]
listed["outlier_any"] = listed[listed_outlier_flags].any(axis=1)
print(f"  Total records flagged by ANY outlier: {listed['outlier_any'].sum():,} "
      f"({listed['outlier_any'].sum()/len(listed)*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BEFORE/AFTER COMPARISON REPORT
# ─────────────────────────────────────────────────────────────────────────────
# The handbook specifically asks for a written comparison of dataset size
# and median values before and after filtering. This produces that comparison.
print("\n" + "=" * 65)
print("BEFORE / AFTER COMPARISON")
print("=" * 65)

comparison_rows = []

def compare_before_after(df_full, df_filtered, col, label, dataset):
    """
    Compares key statistics for a field before and after outlier removal.
    Shows how much the median and mean shift when outliers are excluded.
    """
    if col not in df_full.columns:
        return

    before = pd.to_numeric(df_full[col],     errors="coerce").dropna()
    after  = pd.to_numeric(df_filtered[col], errors="coerce").dropna()

    print(f"\n  [{dataset}] {label}")
    print(f"    Rows    — before: {len(before):,}   after: {len(after):,}   "
          f"removed: {len(before)-len(after):,}")
    print(f"    Median  — before: {before.median():,.2f}   "
          f"after: {after.median():,.2f}   "
          f"diff: {after.median()-before.median():,.2f}")
    print(f"    Mean    — before: {before.mean():,.2f}   "
          f"after: {after.mean():,.2f}   "
          f"diff: {after.mean()-before.mean():,.2f}")
    print(f"    Max     — before: {before.max():,.2f}   after: {after.max():,.2f}")

    comparison_rows.append({
        "dataset":       dataset,
        "field":         col,
        "rows_before":   len(before),
        "rows_after":    len(after),
        "rows_removed":  len(before) - len(after),
        "median_before": round(before.median(), 2),
        "median_after":  round(after.median(), 2),
        "median_diff":   round(after.median() - before.median(), 2),
        "mean_before":   round(before.mean(), 2),
        "mean_after":    round(after.mean(), 2),
        "mean_diff":     round(after.mean() - before.mean(), 2),
        "max_before":    round(before.max(), 2),
        "max_after":     round(after.max(), 2),
    })

# Create filtered datasets — outlier_any == False means clean records only
sold_filtered   = sold[~sold["outlier_any"]].copy()
listed_filtered = listed[~listed["outlier_any"]].copy()

# Run comparisons
compare_before_after(sold,   sold_filtered,   "ClosePrice",         "Close Price",       "SOLD")
compare_before_after(sold,   sold_filtered,   "LivingArea",         "Living Area",       "SOLD")
compare_before_after(sold,   sold_filtered,   "DaysOnMarket",       "Days on Market",    "SOLD")
compare_before_after(sold,   sold_filtered,   "price_per_sqft",     "Price Per Sqft",    "SOLD")
compare_before_after(listed, listed_filtered, "ListPrice",          "List Price",        "LISTED")
compare_before_after(listed, listed_filtered, "LivingArea",         "Living Area",       "LISTED")
compare_before_after(listed, listed_filtered, "DaysOnMarket",       "Days on Market",    "LISTED")
compare_before_after(listed, listed_filtered, "list_price_per_sqft","List Price Per Sqft","LISTED")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print(f"\nSOLD")
print(f"  Full flagged dataset : {len(sold):,} rows")
print(f"  Clean filtered       : {len(sold_filtered):,} rows")
print(f"  Outliers removed     : {len(sold)-len(sold_filtered):,} "
      f"({(len(sold)-len(sold_filtered))/len(sold)*100:.2f}%)")

print(f"\nLISTED")
print(f"  Full flagged dataset : {len(listed):,} rows")
print(f"  Clean filtered       : {len(listed_filtered):,} rows")
print(f"  Outliers removed     : {len(listed)-len(listed_filtered):,} "
      f"({(len(listed)-len(listed_filtered))/len(listed)*100:.2f}%)")

print(f"\nOutlier flag columns added to SOLD:")
for c in sold_outlier_flags + ["outlier_any"]:
    print(f"  {c:<35} : {sold[c].sum():,} flagged")

print(f"\nOutlier flag columns added to LISTED:")
for c in listed_outlier_flags + ["outlier_any"]:
    print(f"  {c:<35} : {listed[c].sum():,} flagged")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SAVING OUTPUTS")
print("=" * 65)

# Full datasets with outlier flags attached
sold.to_csv(  f"{OUTPUT_DIR}/crmls_sold_flagged.csv",   index=False)
listed.to_csv(f"{OUTPUT_DIR}/crmls_listed_flagged.csv", index=False)

# Clean filtered datasets with outliers removed — primary input for Tableau
sold_filtered.to_csv(  f"{OUTPUT_DIR}/crmls_sold_filtered.csv",   index=False)
listed_filtered.to_csv(f"{OUTPUT_DIR}/crmls_listed_filtered.csv", index=False)

# Outlier report combining IQR stats and before/after comparison
all_stats  = pd.DataFrame(sold_stats + listed_stats)
comparison = pd.DataFrame(comparison_rows)

all_stats.to_csv( f"{OUTPUT_DIR}/outlier_iqr_stats.csv",       index=False)
comparison.to_csv(f"{OUTPUT_DIR}/outlier_before_after.csv",    index=False)

print(f"  crmls_sold_flagged.csv    — {len(sold):,} rows x {sold.shape[1]} cols")
print(f"  crmls_listed_flagged.csv  — {len(listed):,} rows x {listed.shape[1]} cols")
print(f"  crmls_sold_filtered.csv   — {len(sold_filtered):,} rows x {sold_filtered.shape[1]} cols")
print(f"  crmls_listed_filtered.csv — {len(listed_filtered):,} rows x {listed_filtered.shape[1]} cols")
print(f"  outlier_iqr_stats.csv     — IQR bounds and flag counts per field")
print(f"  outlier_before_after.csv  — median/mean comparison before vs after")

print("\nDone.")