"""
week2_3_eda_and_enrichment.py
------------------------------
Weeks 2-3 deliverable for the IDX Exchange MLS Analytics Internship.

Covers:
  1. Load & filter Residential only (with row-count logging)
  2. Drop analytically irrelevant columns (same list as feature_engineering.py)
  3. EDA  — shape, dtypes, unique property types
  4. Missing value analysis — null counts, pct, flag columns >90% null
  5. Numeric distribution summary — all key fields saved as ONE combined CSV
  6. Suggested EDA questions answered in console output
  7. FRED MORTGAGE30US fetch → resample weekly→monthly → merge onto both datasets
  8. Validate merge (no null rates after join)
  9. Save enriched CSVs

WHY WE DROP COLUMNS HERE (not based on null % alone):
  The 90% null threshold is a signal to investigate, not an automatic drop rule.
  We drop columns based on whether they are analytically useful for residential
  market analysis. Columns are dropped for one of three reasons:
    A) Not relevant to residential market analysis (e.g. BusinessType, Levels)
    B) Redundant — the data exists in a better column (e.g. LotSizeArea is
       redundant when LotSizeAcres and LotSizeSquareFeet are kept)
    C) Too granular for dashboard use and adds no analytical value
       (e.g. ListAgentFirstName — we keep ListAgentFullName instead)

Inputs  (from Week 1 pipeline):
    output/crmls_sold_final.csv
    output/crmls_listed_final.csv

Outputs:
    output/crmls_sold_week3.csv
    output/crmls_listed_week3.csv
    output/missing_value_report_sold.csv
    output/missing_value_report_listed.csv
    output/numeric_distribution_combined.csv   <- sold + listed in one file

Usage:
    python week2_3_eda_and_enrichment.py
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("LOADING DATASETS")
print("=" * 65)

sold   = pd.read_csv("output/crmls_sold_final.csv",   low_memory=False)
listed = pd.read_csv("output/crmls_listed_final.csv", low_memory=False)

print(f"Sold raw rows   : {len(sold):,}   | columns: {sold.shape[1]}")
print(f"Listed raw rows : {len(listed):,}  | columns: {listed.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROPERTY TYPE BREAKDOWN & RESIDENTIAL FILTER
# ─────────────────────────────────────────────────────────────────────────────
# We filter before dropping columns and before EDA so that all downstream
# analysis reflects only residential records — the focus of this program.
print("\n" + "=" * 65)
print("PROPERTY TYPE BREAKDOWN")
print("=" * 65)

print("\nSold — unique PropertyType values:")
print(sold["PropertyType"].value_counts(dropna=False).to_string())

print("\nListed — unique PropertyType values:")
print(listed["PropertyType"].value_counts(dropna=False).to_string())

sold_pre   = len(sold)
listed_pre = len(listed)

sold   = sold[sold["PropertyType"]   == "Residential"].copy()
listed = listed[listed["PropertyType"] == "Residential"].copy()

print(f"\nRows BEFORE Residential filter — sold: {sold_pre:,}  | listed: {listed_pre:,}")
print(f"Rows AFTER  Residential filter — sold: {len(sold):,}  | listed: {len(listed):,}")
print(f"Residential share — sold: {len(sold)/sold_pre*100:.1f}%  | listed: {len(listed)/listed_pre*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DROP ANALYTICALLY IRRELEVANT COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
# Matches DROP_COLS in feature_engineering.py exactly. Reason for each:
#
#   AboveGradeFinishedArea, BelowGradeFinishedArea
#     Grade-based area breakdowns rarely populated for CA residential.
#     LivingArea is the standard field used for price-per-sqft analysis.
#
#   CoveredSpaces
#     Redundant — GarageSpaces + ParkingTotal already capture this.
#
#   ElementarySchoolDistrict, MiddleOrJuniorSchoolDistrict
#     HighSchoolDistrict is the standard district identifier in CA.
#     Sub-district fields are sparsely populated and rarely used in analysis.
#
#   LotSizeDimensions
#     Free-text field (e.g. "100 x 150"), not useful for numeric analysis.
#     LotSizeAcres and LotSizeSquareFeet are kept as the clean numeric fields.
#
#   BusinessType
#     Commercial field — completely irrelevant for residential analysis.
#
#   CoBuyerAgentFirstName, CoListAgentFirstName, CoListAgentLastName
#     Name fragments. Full name fields are kept where analytically needed.
#     Co-agent granularity is not required for market or competitive dashboards.
#
#   BuildingAreaTotal
#     Redundant with LivingArea for residential. BuildingAreaTotal includes
#     unfinished space which distorts price-per-sqft calculations.
#
#   MainLevelBedrooms
#     Too granular — BedroomsTotal is the standard field for bedroom analysis.
#
#   BuyerAgentMlsId
#     Internal MLS identifier, not needed for market or competitive analysis.
#     BuyerAgentFirstName + BuyerAgentLastName are kept for agent tracking.
#
#   BuyerOfficeAOR
#     Board/association metadata. Not meaningful for dashboard consumers.
#
#   Levels
#     Inconsistently populated enum field. Stories is the cleaner equivalent
#     and better understood in a residential context.
#
#   LotSizeArea
#     Redundant — LotSizeAcres and LotSizeSquareFeet are kept and cover
#     the same information in cleaner, consistently typed numeric fields.
#
#   ListAgentFirstName, ListAgentLastName
#     Name fragments. ListAgentFullName is kept as the single agent identifier.
#
#   ListingKeyNumeric
#     Duplicate of ListingKey in numeric form. ListingKey (string) is the
#     primary key — the numeric version adds no analytical value.
#
#   StreetNumberNumeric
#     Duplicate of the street number already embedded in UnparsedAddress.
#
#   PricePerSqFt
#     Pre-calculated in the source data but we recalculate it ourselves in
#     feature_engineering.py using ClosePrice / LivingArea to ensure
#     consistency and guard against source calculation errors.

DROP_COLS = [
    'AboveGradeFinishedArea', 'BelowGradeFinishedArea', 'CoveredSpaces',
    'ElementarySchoolDistrict', 'MiddleOrJuniorSchoolDistrict',
    'LotSizeDimensions', 'BusinessType', 'CoBuyerAgentFirstName',
    'CoListAgentFirstName', 'CoListAgentLastName', 'BuildingAreaTotal',
    'MainLevelBedrooms', 'BuyerAgentMlsId', 'BuyerOfficeAOR',
    'Levels', 'LotSizeArea', 'ListAgentFirstName', 'ListAgentLastName',
    'ListingKeyNumeric', 'StreetNumberNumeric', 'PricePerSqFt',
]

sold_cols_before   = sold.shape[1]
listed_cols_before = listed.shape[1]

sold   = sold.drop(columns=[c for c in DROP_COLS if c in sold.columns])
listed = listed.drop(columns=[c for c in DROP_COLS if c in listed.columns])

print(f"\nColumns dropped — sold: {sold_cols_before} -> {sold.shape[1]}  "
      f"| listed: {listed_cols_before} -> {listed.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. EDA — STRUCTURE & DTYPES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("EDA — STRUCTURE")
print("=" * 65)

for name, df in [("SOLD", sold), ("LISTED", listed)]:
    print(f"\n[{name}]  shape: {df.shape}")
    print(f"  Numeric columns : {df.select_dtypes(include='number').shape[1]}")
    print(f"  Object columns  : {df.select_dtypes(include='object').shape[1]}")
    print(f"  Bool columns    : {df.select_dtypes(include='bool').shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PARSE DATE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
# errors='coerce' turns any unparseable value into NaT (null) rather than
# crashing the script. Raw MLS data can have inconsistent date formats or
# placeholder values like '0000-00-00', so this defensive approach is important.
date_cols = ["ListingContractDate", "PurchaseContractDate",
             "CloseDate", "ContractStatusChangeDate"]

for df in [sold, listed]:
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MISSING VALUE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MISSING VALUE ANALYSIS")
print("=" * 65)

def missing_report(df, label):
    """
    Builds a null count/percentage report per column.
    Flags columns over 90% null as a signal for investigation —
    but note: columns are dropped based on analytical relevance (step 3),
    not automatically based on this threshold alone.
    """
    total = len(df)
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / total * 100).round(2)
    report = pd.DataFrame({
        "dataset":        label,
        "null_count":     null_counts,
        "null_pct":       null_pct,
        "flag_over90pct": null_pct > 90,
    }).sort_values("null_pct", ascending=False)

    over90 = report[report["flag_over90pct"]]
    print(f"\n[{label}] Total rows: {total:,}")
    print(f"  Columns with >90% null ({len(over90)}):")
    if len(over90):
        for col, row in over90.iterrows():
            print(f"    {col:<45} {row['null_pct']:.1f}% null")
    else:
        print("    None")

    top_missing = report[~report["flag_over90pct"]].head(15)
    print(f"\n  Top 15 partially-missing columns (excluding >90%):")
    for col, row in top_missing.iterrows():
        if row["null_pct"] > 0:
            print(f"    {col:<45} {row['null_pct']:.1f}% null")

    return report

sold_missing   = missing_report(sold,   "SOLD")
listed_missing = missing_report(listed, "LISTED")

# Missing value reports saved separately — they are column-level metadata
# and keeping them per dataset makes them easier to review independently.
sold_missing.to_csv(  f"{OUTPUT_DIR}/missing_value_report_sold.csv")
listed_missing.to_csv(f"{OUTPUT_DIR}/missing_value_report_listed.csv")
print(f"\n  Saved: missing_value_report_sold.csv / missing_value_report_listed.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 7. NUMERIC DISTRIBUTION SUMMARY — ONE combined CSV
# ─────────────────────────────────────────────────────────────────────────────
# We combine sold and listed into a single CSV with a 'dataset' column.
# This is cleaner than two separate files — you can filter by dataset in
# Excel or Tableau and compare distributions side by side in one place.
# p5/p95 are especially useful here for spotting extreme outliers that
# will be handled formally in the Week 7 IQR step.
print("\n" + "=" * 65)
print("NUMERIC DISTRIBUTION SUMMARY")
print("=" * 65)

NUMERIC_FIELDS = [
    "ClosePrice", "ListPrice", "OriginalListPrice",
    "LivingArea", "LotSizeAcres",
    "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt",
]

def distribution_summary(df, fields, label):
    rows = []
    for col in fields:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append({
            "dataset": label,
            "field":   col,
            "count":   len(s),
            "nulls":   df[col].isnull().sum(),
            "min":     s.min(),
            "p5":      s.quantile(0.05),
            "p25":     s.quantile(0.25),
            "median":  s.median(),
            "mean":    round(s.mean(), 2),
            "p75":     s.quantile(0.75),
            "p95":     s.quantile(0.95),
            "max":     s.max(),
            "std":     round(s.std(), 2),
        })
    summary = pd.DataFrame(rows)
    print(f"\n[{label}]")
    print(summary.set_index("field").drop(columns="dataset").to_string())
    return summary

sold_dist   = distribution_summary(sold,   NUMERIC_FIELDS, "SOLD")
listed_dist = distribution_summary(listed, NUMERIC_FIELDS, "LISTED")

# Combine and save as one file
combined_dist = pd.concat([sold_dist, listed_dist], ignore_index=True)
combined_dist.to_csv(f"{OUTPUT_DIR}/numeric_distribution_combined.csv", index=False)
print(f"\n  Saved: numeric_distribution_combined.csv (sold + listed in one file)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SUGGESTED EDA QUESTIONS (handbook)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUGGESTED EDA QUESTIONS")
print("=" * 65)

# Q1: Median and average close prices
cp = pd.to_numeric(sold["ClosePrice"], errors="coerce")
print(f"\nClose Price — median: ${cp.median():,.0f}  |  mean: ${cp.mean():,.0f}")

# Q2: Days on Market distribution
dom = pd.to_numeric(sold["DaysOnMarket"], errors="coerce")
print(f"Days on Market — median: {dom.median():.0f}d  |  mean: {dom.mean():.1f}d  |  p90: {dom.quantile(0.90):.0f}d")

# Q3: % sold above vs. below list price
# We use ListPrice (not OriginalListPrice) because ListPrice reflects the
# price at time of offer — it may have been reduced from original.
# A ratio > 1 means the buyer paid more than the final asking price (overbid).
if "ListPrice" in sold.columns and "ClosePrice" in sold.columns:
    lp    = pd.to_numeric(sold["ListPrice"], errors="coerce")
    ratio = cp / lp
    above = (ratio > 1.0).sum()
    at    = (ratio == 1.0).sum()
    below = (ratio < 1.0).sum()
    total_valid = ratio.notna().sum()
    print(f"Sold vs Ask   — Above: {above/total_valid*100:.1f}%  |  "
          f"At: {at/total_valid*100:.1f}%  |  Below: {below/total_valid*100:.1f}%")

# Q4: Date consistency issues
# ListingContractDate should always precede CloseDate.
# If it doesn't, it's a data entry error that needs flagging.
if "ListingContractDate" in sold.columns and "CloseDate" in sold.columns:
    bad_dates = (sold["ListingContractDate"] > sold["CloseDate"]).sum()
    print(f"Date inconsistency (listing after close): {bad_dates:,} records")

# Q5: Counties with highest median close price
if "CountyOrParish" in sold.columns:
    county_median = (
        sold.groupby("CountyOrParish")["ClosePrice"]
        .median()
        .sort_values(ascending=False)
        .head(10)
    )
    print(f"\nTop 10 counties by median close price:")
    for county, price in county_median.items():
        print(f"  {county:<30} ${price:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FRED MORTGAGE RATE FETCH & MERGE
# ─────────────────────────────────────────────────────────────────────────────
# FRED publishes the 30-year fixed mortgage rate weekly (every Thursday).
# We resample to monthly by averaging all Thursday readings in each calendar
# month, then left join onto both datasets using a year_month key so every
# transaction row gets the rate in effect during that month.
#
# Why a left join?
#   A left join keeps all MLS rows even if no matching mortgage rate is found.
#   This is safer than an inner join which would silently drop rows for months
#   where FRED data hasn't been published yet (e.g. the most recent month).
print("\n" + "=" * 65)
print("FRED MORTGAGE RATE ENRICHMENT")
print("=" * 65)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

print("  Fetching MORTGAGE30US from FRED...")
try:
    mortgage = pd.read_csv(FRED_URL, parse_dates=["observation_date"])
    mortgage.columns = ["date", "rate_30yr_fixed"]

    # FRED uses '.' as a placeholder for weeks with no published reading.
    # Drop those before converting to numeric to avoid coercion errors.
    mortgage = mortgage[mortgage["rate_30yr_fixed"] != "."].copy()
    mortgage["rate_30yr_fixed"] = pd.to_numeric(mortgage["rate_30yr_fixed"], errors="coerce")
    mortgage = mortgage.dropna(subset=["rate_30yr_fixed"])

    # to_period('M') groups all dates in the same calendar month together,
    # then we take the mean of all weekly readings within that month.
    mortgage["year_month"] = mortgage["date"].dt.to_period("M")
    mortgage_monthly = (
        mortgage.groupby("year_month")["rate_30yr_fixed"]
        .mean()
        .round(3)
        .reset_index()
    )
    print(f"  Mortgage data: {len(mortgage_monthly)} monthly observations")
    print(f"  Range: {mortgage_monthly['year_month'].min()} -> {mortgage_monthly['year_month'].max()}")
    print(f"  Recent rates:\n{mortgage_monthly.tail(6).to_string(index=False)}")

    # Create join keys on both MLS datasets.
    # Sold uses CloseDate — the rate when the deal actually closed.
    # Listed uses ListingContractDate — the rate when it came to market.
    sold["year_month"]   = sold["CloseDate"].dt.to_period("M")
    listed["year_month"] = listed["ListingContractDate"].dt.to_period("M")

    sold   = sold.merge(mortgage_monthly,   on="year_month", how="left")
    listed = listed.merge(mortgage_monthly, on="year_month", how="left")

    # Validate — null rates after merge means that month had no FRED data,
    # which typically only happens for very recent or future months.
    sold_null_rate   = sold["rate_30yr_fixed"].isnull().sum()
    listed_null_rate = listed["rate_30yr_fixed"].isnull().sum()
    print(f"\n  Merge validation:")
    print(f"    Sold   — null rate values after merge : {sold_null_rate:,}")
    print(f"    Listed — null rate values after merge : {listed_null_rate:,}")

    if sold_null_rate > 0:
        print(f"    [WARNING] Unmatched year_months in sold:")
        print(f"    {sold[sold['rate_30yr_fixed'].isnull()]['year_month'].unique()}")
    if listed_null_rate > 0:
        print(f"    [WARNING] Unmatched year_months in listed:")
        print(f"    {listed[listed['rate_30yr_fixed'].isnull()]['year_month'].unique()}")

    preview_cols = ["CloseDate", "year_month", "ClosePrice", "rate_30yr_fixed"]
    preview_cols = [c for c in preview_cols if c in sold.columns]
    print(f"\n  Sold preview (5 rows):")
    print(sold[preview_cols].dropna().head(5).to_string(index=False))

except Exception as e:
    print(f"  [ERROR] Could not fetch FRED data: {e}")
    print("  Continuing without mortgage rate enrichment.")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE ENRICHED OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SAVING OUTPUTS")
print("=" * 65)

sold.to_csv(  f"{OUTPUT_DIR}/crmls_sold_week3.csv",   index=False)
listed.to_csv(f"{OUTPUT_DIR}/crmls_listed_week3.csv", index=False)

print(f"  crmls_sold_week3.csv   — {sold.shape[0]:,} rows x {sold.shape[1]} cols")
print(f"  crmls_listed_week3.csv — {listed.shape[0]:,} rows x {listed.shape[1]} cols")

print("\nDone.")