"""
week6_feature_engineering.py
-----------------------------
Week 6 deliverable for the IDX Exchange MLS Analytics Internship.

Reads from the Week 4-5 cleaned datasets and engineers all key market
metrics required for Tableau dashboard development in Weeks 8-10.

What this script does:
  1. Load cleaned datasets from Weeks 4-5
  2. Drop analytically irrelevant columns (same list as before)
  3. Parse date fields
  4. Engineer SOLD features:
       - Price ratios (close-to-orig, close-to-list)
       - Price per square foot
       - Time dimensions (year, month, quarter, yrmo, yrqtr)
       - Timeline durations (listing to contract, contract to close, list to close)
       - Market condition label (Above/At/Below Ask) — fixed binning
       - DOM bucket, age tier, price tier, lot tier, bed tier
       - Bool -> Yes/No labels
  5. Engineer LISTED features:
       - List price per square foot
       - Close-to-orig ratio (closed listings only)
       - Time dimensions (listing date based)
       - Days listing to contract
       - Price reduction label
       - Status label and active supply flag
       - DOM bucket, age tier, price tier, lot tier, bed tier
       - Bool -> Yes/No labels
  6. Segment summary tables:
       - By CountyOrParish
       - By PropertySubType
       - By ListOfficeName (top 50)
       - By BuyerOfficeName (top 50)
       - By MLSAreaMajor
  7. Save engineered datasets and segment summary tables

Inputs  (from Weeks 4-5):
    output/crmls_sold_cleaned.csv
    output/crmls_listed_cleaned.csv

Outputs:
    output/crmls_sold_engineered.csv
    output/crmls_listed_engineered.csv
    output/segment_summary_sold.csv
    output/segment_summary_listed.csv

Usage:
    python week6_feature_engineering.py
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD CLEANED DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("LOADING CLEANED DATASETS")
print("=" * 65)

sold   = pd.read_csv("output/crmls_sold_cleaned.csv",   low_memory=False)
listed = pd.read_csv("output/crmls_listed_cleaned.csv", low_memory=False)

print(f"Sold   loaded: {len(sold):,} rows x {sold.shape[1]} cols")
print(f"Listed loaded: {len(listed):,} rows x {listed.shape[1]} cols")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DROP ANALYTICALLY IRRELEVANT COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
# Same list carried through from Week 2-3. Applied again here defensively
# in case any of these columns reappeared after the cleaning step.
DROP_COLS = [
    'AboveGradeFinishedArea', 'BelowGradeFinishedArea', 'CoveredSpaces',
    'ElementarySchoolDistrict', 'MiddleOrJuniorSchoolDistrict',
    'LotSizeDimensions', 'BusinessType', 'CoBuyerAgentFirstName',
    'CoListAgentFirstName', 'CoListAgentLastName', 'BuildingAreaTotal',
    'MainLevelBedrooms', 'BuyerAgentMlsId', 'BuyerOfficeAOR',
    'Levels', 'LotSizeArea', 'ListAgentFirstName', 'ListAgentLastName',
    'ListingKeyNumeric', 'StreetNumberNumeric', 'PricePerSqFt',
]

sold   = sold.drop(columns=[c for c in DROP_COLS if c in sold.columns])
listed = listed.drop(columns=[c for c in DROP_COLS if c in listed.columns])

# ─────────────────────────────────────────────────────────────────────────────
# 3. PARSE DATE FIELDS
# ─────────────────────────────────────────────────────────────────────────────
# Re-parse every time after a CSV reload — datetime columns revert to strings.
date_cols = [
    "ListingContractDate", "PurchaseContractDate",
    "CloseDate", "ContractStatusChangeDate",
]

for df in [sold, listed]:
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

print("\nDates parsed.")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (shared between both datasets)
# ─────────────────────────────────────────────────────────────────────────────

def dom_bucket(d):
    """
    Buckets Days on Market into human-readable speed tiers.
    Prefixed with numbers so Tableau sorts them in the correct order.
    """
    if pd.isna(d): return None
    d = float(d)
    if d <= 7:   return "1 - Very Fast (1-7d)"
    if d <= 30:  return "2 - Fast (8-30d)"
    if d <= 60:  return "3 - Average (31-60d)"
    if d <= 90:  return "4 - Slow (61-90d)"
    return "5 - Very Slow (90+d)"

def age_tier(y):
    """
    Groups YearBuilt into era-based tiers for property age analysis.
    Useful for understanding whether newer or older homes command premiums.
    """
    if pd.isna(y): return None
    y = float(y)
    if y >= 2020: return "New (2020+)"
    if y >= 2000: return "Modern (2000-2019)"
    if y >= 1980: return "Contemporary (1980-1999)"
    if y >= 1960: return "Established (1960-1979)"
    return "Vintage (pre-1960)"

def lot_tier(a):
    """
    Groups LotSizeAcres into size tiers.
    Helps segment the market between urban small-lot and estate properties.
    """
    if pd.isna(a): return None
    a = float(a)
    if a < 0.1:  return "Small (<0.1 ac)"
    if a < 0.25: return "Medium (0.1-0.25 ac)"
    if a < 0.5:  return "Large (0.25-0.5 ac)"
    if a < 1.0:  return "XL (0.5-1 ac)"
    return "Estate (1+ ac)"

def bed_tier(b):
    """
    Groups BedroomsTotal into standard bedroom count tiers.
    5BR+ grouped together since they represent a small, distinct luxury segment.
    """
    if pd.isna(b): return None
    b = int(float(b))
    if b <= 1: return "1BR or less"
    if b == 2: return "2BR"
    if b == 3: return "3BR"
    if b == 4: return "4BR"
    return "5BR+"

# Price tier bins — covers the full CA residential market range
# Under $400K captures affordable/inland markets
# $1.5M+ captures the luxury segment
price_bins   = [0, 400_000, 700_000, 1_000_000, 1_500_000, float("inf")]
price_labels = ["Under $400K", "$400K-$700K", "$700K-$1M", "$1M-$1.5M", "$1.5M+"]

# Boolean to Yes/No mapping for Tableau-friendly labels
# Covers all common forms the boolean might appear in after CSV roundtripping
bool_yn = {
    True: "Yes", False: "No",
    "True": "Yes", "False": "No",
    1: "Yes", 0: "No",
}

# ═════════════════════════════════════════════════════════════════════════════
# 4. SOLD — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ENGINEERING SOLD FEATURES")
print("=" * 65)

# -- Price ratios --------------------------------------------------------------
# close_to_orig_ratio: how much the buyer paid vs the original asking price
# Captures the full negotiation history including any price reductions
# Guards against division by zero with the > 0 check
sold["close_to_orig_ratio"] = np.where(
    sold["OriginalListPrice"].notna() & (sold["OriginalListPrice"] > 0),
    (sold["ClosePrice"] / sold["OriginalListPrice"]).round(4),
    np.nan,
)

# close_to_list_ratio: how much the buyer paid vs the final asking price
# Captures the overbid/underbid relative to what the seller was asking at
# time of offer — this is the more common market competitiveness metric
sold["close_to_list_ratio"] = np.where(
    sold["ListPrice"].notna() & (sold["ListPrice"] > 0),
    (sold["ClosePrice"] / sold["ListPrice"]).round(4),
    np.nan,
)

# -- Price per square foot -----------------------------------------------------
# Normalises price across different property sizes for fair comparison
# A $1.2M 3000sqft house and a $1.2M 1000sqft condo are very different markets
sold["price_per_sqft"] = np.where(
    sold["LivingArea"].notna() & (sold["LivingArea"] > 0),
    (sold["ClosePrice"] / sold["LivingArea"]).round(2),
    np.nan,
)

# -- Time dimensions -----------------------------------------------------------
# These are what Tableau uses to build time-series charts and trend lines.
# Int64 (capital I) is used instead of int64 because it supports null values —
# regular int64 would crash if CloseDate is NaT for any row.
sold["close_year"]    = sold["CloseDate"].dt.year.astype("Int64")
sold["close_month"]   = sold["CloseDate"].dt.month.astype("Int64")
sold["close_quarter"] = sold["CloseDate"].dt.quarter.astype("Int64")
sold["close_yrmo"]    = sold["CloseDate"].dt.to_period("M").astype(str)

# close_yrqtr: fixed version — null-safe string concatenation
# Previous version used astype(str) directly on Int64 which produced
# "<NA>-Q<NA>" strings for rows with null dates instead of a proper null
sold["close_yrqtr"] = np.where(
    sold["close_year"].notna() & sold["close_quarter"].notna(),
    sold["close_year"].astype(str) + "-Q" + sold["close_quarter"].astype(str),
    np.nan,
)

# -- Timeline durations --------------------------------------------------------
# These three metrics together give a complete picture of market velocity:
#   days_listing_to_contract: how long it took to get an offer accepted
#   days_contract_to_close:   how long the escrow period was
#   days_list_to_close:       total time from listing to keys in hand
sold["days_listing_to_contract"] = (
    sold["PurchaseContractDate"] - sold["ListingContractDate"]
).dt.days

sold["days_contract_to_close"] = (
    sold["CloseDate"] - sold["PurchaseContractDate"]
).dt.days

sold["days_list_to_close"] = (
    sold["CloseDate"] - sold["ListingContractDate"]
).dt.days

# -- Market condition ----------------------------------------------------------
# Fixed version using np.select instead of pd.cut.
# pd.cut with bins [0.9999, 1.0] had a gap issue where values like 0.99995
# could fall into the wrong bucket. np.select evaluates conditions explicitly
# so there is no gap and the logic is clear and readable.
#
# Based on close_to_orig_ratio (vs original list price, not final list price)
# so it captures the full pricing history of the property.
conditions = [
    sold["close_to_orig_ratio"] > 1.0,
    sold["close_to_orig_ratio"] == 1.0,
    sold["close_to_orig_ratio"] < 1.0,
]
choices = ["Above Ask", "At Ask", "Below Ask"]
sold["market_condition"] = np.select(
    conditions, choices, default="Unknown"
)

# -- DOM bucket ----------------------------------------------------------------
sold["dom_bucket"] = sold["DaysOnMarket"].apply(dom_bucket)

# -- Age tier ------------------------------------------------------------------
sold["age_tier"] = sold["YearBuilt"].apply(age_tier)

# -- Price tier (ClosePrice) ---------------------------------------------------
sold["price_tier"] = pd.cut(
    sold["ClosePrice"],
    bins=price_bins,
    labels=price_labels,
    right=False,
)

# -- Lot tier ------------------------------------------------------------------
sold["lot_tier"] = sold["LotSizeAcres"].apply(lot_tier)

# -- Bed tier ------------------------------------------------------------------
sold["bed_tier"] = sold["BedroomsTotal"].apply(bed_tier)

# -- Bed/Bath label ------------------------------------------------------------
# Human-readable string like "3bd/2ba" for display in Tableau tooltips
sold["bed_bath_label"] = (
    sold["BedroomsTotal"].fillna(0).astype(int).astype(str)
    + "bd/"
    + sold["BathroomsTotalInteger"].fillna(0).astype(int).astype(str)
    + "ba"
)

# -- Has HOA -------------------------------------------------------------------
if "AssociationFee" in sold.columns:
    sold["HasHOA"] = sold["AssociationFee"].fillna(0) > 0

# -- Bool -> Yes/No labels -----------------------------------------------------
# Tableau works better with "Yes"/"No" strings than True/False booleans
# for filters and calculated fields
bool_cols_sold = [
    "NewConstructionYN", "AttachedGarageYN", "FireplaceYN",
    "PoolPrivateYN", "ViewYN", "BasementYN", "WaterfrontYN", "HasHOA",
]
for c in bool_cols_sold:
    if c in sold.columns:
        sold[f"{c}_label"] = sold[c].map(bool_yn)

print(f"  Sold engineered shape: {sold.shape}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. LISTED — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ENGINEERING LISTED FEATURES")
print("=" * 65)

# -- List price per square foot ------------------------------------------------
# Uses ListPrice (not ClosePrice) since most listings haven't closed yet
listed["list_price_per_sqft"] = np.where(
    listed["LivingArea"].notna() & (listed["LivingArea"] > 0),
    (listed["ListPrice"] / listed["LivingArea"]).round(2),
    np.nan,
)

# -- Close-to-orig ratio (closed listings only) --------------------------------
# Only populated where ClosePrice exists — active listings will be null here
listed["close_to_orig_ratio"] = np.where(
    listed["ClosePrice"].notna() &
    listed["OriginalListPrice"].notna() &
    (listed["OriginalListPrice"] > 0),
    (listed["ClosePrice"] / listed["OriginalListPrice"]).round(4),
    np.nan,
)

# -- Time dimensions (listing date based) --------------------------------------
# For listings we use ListingContractDate not CloseDate since most haven't closed
listed["listing_year"]    = listed["ListingContractDate"].dt.year.astype("Int64")
listed["listing_month"]   = listed["ListingContractDate"].dt.month.astype("Int64")
listed["listing_quarter"] = listed["ListingContractDate"].dt.quarter.astype("Int64")
listed["listing_yrmo"]    = listed["ListingContractDate"].dt.to_period("M").astype(str)

# Null-safe yrqtr string (same fix as sold)
listed["listing_yrqtr"] = np.where(
    listed["listing_year"].notna() & listed["listing_quarter"].notna(),
    listed["listing_year"].astype(str) + "-Q" + listed["listing_quarter"].astype(str),
    np.nan,
)

# -- Days from listing to contract ---------------------------------------------
# Only populated where a purchase contract date exists
listed["days_listing_to_contract"] = np.where(
    listed["PurchaseContractDate"].notna(),
    (listed["PurchaseContractDate"] - listed["ListingContractDate"]).dt.days,
    np.nan,
)

# -- Price reduction label -----------------------------------------------------
# PriceReduction = OriginalListPrice - ListPrice (calculated in Week 1 pipeline)
# A value > 0 means the seller dropped their price at least once
if "PriceReduction" in listed.columns:
    listed["had_price_reduction"] = (
        listed["PriceReduction"].notna() & (listed["PriceReduction"] > 0)
    ).map({True: "Yes", False: "No"})

# -- Status label (human-readable) ---------------------------------------------
# MlsStatus values in raw data are CamelCase without spaces
# This maps them to clean readable labels for Tableau
status_map = {
    "Active":               "Active",
    "Pending":              "Pending",
    "ActiveUnderContract":  "Active Under Contract",
    "Closed":               "Closed",
    "ComingSoon":           "Coming Soon",
    "Expired":              "Expired",
    "Withdrawn":            "Withdrawn",
    "Canceled":             "Canceled",
}
listed["status_label"] = (
    listed["MlsStatus"].map(status_map).fillna(listed["MlsStatus"])
)

# -- Active supply flag --------------------------------------------------------
# Identifies records that represent current active inventory
# Used in Tableau to calculate supply-side metrics like months of inventory
listed["is_active_supply"] = listed["MlsStatus"].isin(
    ["Active", "ActiveUnderContract", "ComingSoon"]
).map({True: "Yes", False: "No"})

# -- DOM / age / lot / bed tiers -----------------------------------------------
listed["dom_bucket"] = listed["DaysOnMarket"].apply(dom_bucket)
listed["age_tier"]   = listed["YearBuilt"].apply(age_tier)
listed["lot_tier"]   = listed["LotSizeAcres"].apply(lot_tier)
listed["bed_tier"]   = listed["BedroomsTotal"].apply(bed_tier)

# -- Price tier (ListPrice) ----------------------------------------------------
listed["price_tier"] = pd.cut(
    listed["ListPrice"],
    bins=price_bins,
    labels=price_labels,
    right=False,
)

# -- Bed/Bath label ------------------------------------------------------------
listed["bed_bath_label"] = (
    listed["BedroomsTotal"].fillna(0).astype(int).astype(str)
    + "bd/"
    + listed["BathroomsTotalInteger"].fillna(0).astype(int).astype(str)
    + "ba"
)

# -- Has HOA -------------------------------------------------------------------
if "AssociationFee" in listed.columns:
    listed["HasHOA"] = listed["AssociationFee"].fillna(0) > 0

# -- Bool -> Yes/No labels -----------------------------------------------------
bool_cols_listed = [
    "NewConstructionYN", "AttachedGarageYN", "FireplaceYN", "HasHOA",
]
for c in bool_cols_listed:
    if c in listed.columns:
        listed[f"{c}_label"] = listed[c].map(bool_yn)

# -- Geographic flags (carry through from cleaning) ----------------------------
# These were already created in Week 4-5 and will persist in the output
# No need to recalculate — just confirming they're still present
geo_flags = ["flag_missing_coords", "flag_zero_coords",
             "flag_positive_long", "flag_any_geo_issue"]
present = [c for c in geo_flags if c in listed.columns]
print(f"  Geographic flags present: {present}")
print(f"  Listed engineered shape: {listed.shape}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. SEGMENT SUMMARY TABLES
# ═════════════════════════════════════════════════════════════════════════════
# The handbook asks for summary statistics grouped by key dimensions.
# These tables are saved as CSVs and can be imported directly into Tableau
# or used as reference during the presentation.
print("\n" + "=" * 65)
print("SEGMENT SUMMARY TABLES")
print("=" * 65)

sold_summaries   = []
listed_summaries = []

def segment_summary(df, group_col, label, dataset):
    """
    Groups by a dimension column and computes key market metrics.
    Returns a summary dataframe with the dimension, dataset label,
    and aggregated statistics.
    """
    if group_col not in df.columns:
        print(f"  [{dataset}] {group_col} not found, skipping")
        return None

    grp = df.groupby(group_col).agg(
        total_records      = (group_col, "count"),
        median_close_price = ("ClosePrice",          "median") if "ClosePrice" in df.columns else (group_col, "count"),
        mean_close_price   = ("ClosePrice",          "mean")   if "ClosePrice" in df.columns else (group_col, "count"),
        median_list_price  = ("ListPrice",           "median") if "ListPrice"  in df.columns else (group_col, "count"),
        median_ppsf        = ("price_per_sqft",      "median") if "price_per_sqft" in df.columns else (group_col, "count"),
        median_dom         = ("DaysOnMarket",        "median"),
        median_cto_ratio   = ("close_to_orig_ratio", "median") if "close_to_orig_ratio" in df.columns else (group_col, "count"),
    ).reset_index()

    grp["segment"]  = label
    grp["dataset"]  = dataset
    grp["mean_close_price"] = grp["mean_close_price"].round(0)
    print(f"  [{dataset}] {label}: {len(grp)} groups")
    return grp

# SOLD segment summaries
for col, label in [
    ("CountyOrParish",   "By County"),
    ("PropertySubType",  "By Property Subtype"),
    ("MLSAreaMajor",     "By MLS Area"),
]:
    result = segment_summary(sold, col, label, "SOLD")
    if result is not None:
        sold_summaries.append(result)

# Top 50 list offices by volume (sold)
if "ListOfficeName" in sold.columns:
    top_offices = (
        sold.groupby("ListOfficeName")
        .agg(
            total_records      = ("ListOfficeName",      "count"),
            total_volume       = ("ClosePrice",          "sum"),
            median_close_price = ("ClosePrice",          "median"),
            median_dom         = ("DaysOnMarket",        "median"),
            median_cto_ratio   = ("close_to_orig_ratio", "median"),
        )
        .reset_index()
        .sort_values("total_volume", ascending=False)
        .head(50)
    )
    top_offices["segment"] = "Top 50 List Offices"
    top_offices["dataset"] = "SOLD"
    sold_summaries.append(top_offices)
    print(f"  [SOLD] Top 50 List Offices by volume: calculated")

# Top 50 buyer offices by volume (sold)
if "BuyerOfficeName" in sold.columns:
    top_buyer_offices = (
        sold.groupby("BuyerOfficeName")
        .agg(
            total_records      = ("BuyerOfficeName",     "count"),
            total_volume       = ("ClosePrice",          "sum"),
            median_close_price = ("ClosePrice",          "median"),
            median_dom         = ("DaysOnMarket",        "median"),
        )
        .reset_index()
        .sort_values("total_volume", ascending=False)
        .head(50)
    )
    top_buyer_offices["segment"] = "Top 50 Buyer Offices"
    top_buyer_offices["dataset"] = "SOLD"
    sold_summaries.append(top_buyer_offices)
    print(f"  [SOLD] Top 50 Buyer Offices by volume: calculated")

# LISTED segment summaries
for col, label in [
    ("CountyOrParish",  "By County"),
    ("PropertySubType", "By Property Subtype"),
    ("MLSAreaMajor",    "By MLS Area"),
    ("status_label",    "By Status"),
]:
    result = segment_summary(listed, col, label, "LISTED")
    if result is not None:
        listed_summaries.append(result)

# Combine and save
if sold_summaries:
    sold_seg = pd.concat(sold_summaries, ignore_index=True)
    sold_seg.to_csv(f"{OUTPUT_DIR}/segment_summary_sold.csv", index=False)
    print(f"\n  Saved: segment_summary_sold.csv")

if listed_summaries:
    listed_seg = pd.concat(listed_summaries, ignore_index=True)
    listed_seg.to_csv(f"{OUTPUT_DIR}/segment_summary_listed.csv", index=False)
    print(f"  Saved: segment_summary_listed.csv")

# ═════════════════════════════════════════════════════════════════════════════
# 7. VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("VALIDATION")
print("=" * 65)

print("\n[SOLD]")
print(f"  Shape: {sold.shape}")

num_check = [
    "close_to_orig_ratio", "close_to_list_ratio", "price_per_sqft",
    "days_listing_to_contract", "days_contract_to_close", "days_list_to_close",
]
for c in num_check:
    if c in sold.columns:
        s = sold[c].dropna()
        print(f"  {c}: median={s.median():.3f}, nulls={sold[c].isnull().sum():,}")

print()
cat_check = ["market_condition", "dom_bucket", "age_tier", "price_tier", "bed_tier"]
for c in cat_check:
    if c in sold.columns:
        print(f"  {c}: {dict(sold[c].value_counts().head(3))}")

print("\n[LISTED]")
print(f"  Shape: {listed.shape}")
for c in ["list_price_per_sqft", "days_listing_to_contract"]:
    if c in listed.columns:
        s = listed[c].dropna()
        print(f"  {c}: median={s.median():.2f}, nulls={listed[c].isnull().sum():,}")

for c in ["status_label", "dom_bucket", "had_price_reduction", "is_active_supply"]:
    if c in listed.columns:
        print(f"  {c}: {dict(listed[c].value_counts().head(3))}")

# ═════════════════════════════════════════════════════════════════════════════
# 8. SAVE OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SAVING OUTPUTS")
print("=" * 65)

sold.to_csv(  f"{OUTPUT_DIR}/crmls_sold_engineered.csv",   index=False)
listed.to_csv(f"{OUTPUT_DIR}/crmls_listed_engineered.csv", index=False)

print(f"  crmls_sold_engineered.csv   — {sold.shape[0]:,} rows x {sold.shape[1]} cols")
print(f"  crmls_listed_engineered.csv — {listed.shape[0]:,} rows x {listed.shape[1]} cols")

print("\nDone.")