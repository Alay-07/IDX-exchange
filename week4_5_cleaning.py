"""
week4_5_cleaning.py
--------------------
Weeks 4-5 deliverable for the IDX Exchange MLS Analytics Internship.

This script takes the Week 3 enriched datasets and applies a full round
of data cleaning and validation. Every transformation is documented with
before/after row counts so the cleaning process is fully auditable.

What this script does:
  1. Load the Week 3 enriched datasets
  2. Confirm and re-parse date fields
  3. Confirm numeric field types and coerce where needed
  4. Flag and remove invalid numeric values
     (ClosePrice <= 0, LivingArea <= 0, DaysOnMarket < 0,
      negative BedroomsTotal or BathroomsTotalInteger)
  5. Date consistency checks and flag columns
     - listing_after_close_flag
     - purchase_after_close_flag
     - negative_timeline_flag
  6. Geographic data quality checks and flag columns
     - flag_missing_coords
     - flag_zero_coords
     - flag_positive_long  (CA longitudes must be negative)
     - flag_out_of_range_coords (implausible for California)
  7. String field cleanup (strip whitespace, normalize nulls)
  8. PostalCode validation (must be 5-digit CA zip)
  9. Summary report — before/after row counts, flag counts,
     data type confirmations
  10. Save cleaned datasets

Inputs  (from Week 3):
    output/crmls_sold_week3.csv
    output/crmls_listed_week3.csv

Outputs:
    output/crmls_sold_cleaned.csv       <- cleaned, flagged, analysis-ready
    output/crmls_listed_cleaned.csv     <- cleaned, flagged, analysis-ready
    output/cleaning_report.csv          <- summary of every transformation

Usage:
    python week4_5_cleaning.py
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# We'll collect a log of every transformation for the cleaning report
cleaning_log = []

def log(dataset, transformation, before, after, notes=""):
    """Record a transformation step with before/after row counts."""
    removed = before - after
    cleaning_log.append({
        "dataset":        dataset,
        "transformation": transformation,
        "rows_before":    before,
        "rows_after":     after,
        "rows_removed":   removed,
        "pct_removed":    round(removed / before * 100, 3) if before > 0 else 0,
        "notes":          notes,
    })
    print(f"  [{dataset}] {transformation}: {before:,} -> {after:,}  "
          f"(removed {removed:,} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("LOADING WEEK 3 DATASETS")
print("=" * 65)

sold   = pd.read_csv("output/crmls_sold_week3.csv",   low_memory=False)
listed = pd.read_csv("output/crmls_listed_week3.csv", low_memory=False)

print(f"Sold   loaded: {len(sold):,} rows x {sold.shape[1]} cols")
print(f"Listed loaded: {len(listed):,} rows x {listed.shape[1]} cols")

sold_original_rows   = len(sold)
listed_original_rows = len(listed)

# ─────────────────────────────────────────────────────────────────────────────
# 2. RE-PARSE DATE FIELDS
# ─────────────────────────────────────────────────────────────────────────────
# When CSVs are saved and reloaded, datetime columns revert to strings.
# We re-parse them here so all date arithmetic works correctly downstream.
# errors='coerce' turns any value that can't be parsed into NaT (null date)
# rather than crashing — important because raw MLS data can have bad values.
print("\n" + "=" * 65)
print("DATE FIELD PARSING")
print("=" * 65)

date_cols = [
    "ListingContractDate", "PurchaseContractDate",
    "CloseDate", "ContractStatusChangeDate",
]

for df, name in [(sold, "SOLD"), (listed, "LISTED")]:
    for c in date_cols:
        if c in df.columns:
            before_nulls = df[c].isnull().sum()
            df[c] = pd.to_datetime(df[c], errors="coerce")
            after_nulls  = df[c].isnull().sum()
            new_nulls    = after_nulls - before_nulls
            print(f"  [{name}] {c}: parsed OK "
                  f"({'no new nulls' if new_nulls == 0 else f'{new_nulls} new NaT values from bad data'})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. NUMERIC FIELD TYPE CONFIRMATION
# ─────────────────────────────────────────────────────────────────────────────
# Ensure all key numeric fields are actually stored as numbers, not strings.
# This can happen when CSVs have mixed types or formatting characters.
print("\n" + "=" * 65)
print("NUMERIC FIELD TYPE COERCION")
print("=" * 65)

numeric_cols = [
    "OriginalListPrice", "ListPrice", "ClosePrice",
    "DaysOnMarket", "BedroomsTotal", "BathroomsTotalInteger",
    "LivingArea", "LotSizeAcres", "LotSizeSquareFeet",
    "YearBuilt", "Stories", "GarageSpaces", "ParkingTotal",
    "AssociationFee", "TaxAnnualAmount", "Latitude", "Longitude",
]

for df, name in [(sold, "SOLD"), (listed, "LISTED")]:
    for c in numeric_cols:
        if c in df.columns:
            before_nulls = df[c].isnull().sum()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            after_nulls  = df[c].isnull().sum()
            new_nulls    = after_nulls - before_nulls
            if new_nulls > 0:
                print(f"  [{name}] {c}: {new_nulls} values coerced to NaN (were non-numeric strings)")
            else:
                print(f"  [{name}] {c}: OK")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FLAG AND REMOVE INVALID NUMERIC VALUES
# ─────────────────────────────────────────────────────────────────────────────
# These are business-rule violations — values that are mathematically possible
# but make no sense in a real estate context and indicate data entry errors.
# We flag them first (so we can count them), then remove from the clean dataset.
print("\n" + "=" * 65)
print("INVALID NUMERIC VALUE REMOVAL")
print("=" * 65)

# --- SOLD dataset ---
print("\n[SOLD]")

# ClosePrice <= 0: a sale with no price is meaningless for market analysis
mask = sold["ClosePrice"].notna() & (sold["ClosePrice"] <= 0)
print(f"  ClosePrice <= 0: {mask.sum():,} records flagged")
before = len(sold)
sold = sold[~mask].copy()
log("SOLD", "Remove ClosePrice <= 0", before, len(sold),
    "Sale price of zero or negative is a data entry error")

# ListPrice <= 0: listing with no price can't be used for ratio calculations
mask = sold["ListPrice"].notna() & (sold["ListPrice"] <= 0)
print(f"  ListPrice <= 0: {mask.sum():,} records flagged")
before = len(sold)
sold = sold[~mask].copy()
log("SOLD", "Remove ListPrice <= 0", before, len(sold),
    "List price of zero or negative is invalid")

# LivingArea <= 0: can't calculate price-per-sqft without valid square footage
mask = sold["LivingArea"].notna() & (sold["LivingArea"] <= 0)
print(f"  LivingArea <= 0: {mask.sum():,} records flagged")
before = len(sold)
sold = sold[~mask].copy()
log("SOLD", "Remove LivingArea <= 0", before, len(sold),
    "Zero or negative living area makes price-per-sqft impossible")

# DaysOnMarket < 0: a property can't sell before it's listed
mask = sold["DaysOnMarket"].notna() & (sold["DaysOnMarket"] < 0)
print(f"  DaysOnMarket < 0: {mask.sum():,} records flagged")
before = len(sold)
sold = sold[~mask].copy()
log("SOLD", "Remove DaysOnMarket < 0", before, len(sold),
    "Negative days on market is a data entry error")

# BedroomsTotal < 0: physically impossible
mask = sold["BedroomsTotal"].notna() & (sold["BedroomsTotal"] < 0)
print(f"  BedroomsTotal < 0: {mask.sum():,} records flagged")
before = len(sold)
sold = sold[~mask].copy()
log("SOLD", "Remove BedroomsTotal < 0", before, len(sold),
    "Negative bedrooms is physically impossible")

# BathroomsTotalInteger < 0: physically impossible
mask = sold["BathroomsTotalInteger"].notna() & (sold["BathroomsTotalInteger"] < 0)
print(f"  BathroomsTotalInteger < 0: {mask.sum():,} records flagged")
before = len(sold)
sold = sold[~mask].copy()
log("SOLD", "Remove BathroomsTotalInteger < 0", before, len(sold),
    "Negative bathrooms is physically impossible")

# --- LISTED dataset ---
print("\n[LISTED]")

mask = listed["ListPrice"].notna() & (listed["ListPrice"] <= 0)
print(f"  ListPrice <= 0: {mask.sum():,} records flagged")
before = len(listed)
listed = listed[~mask].copy()
log("LISTED", "Remove ListPrice <= 0", before, len(listed),
    "List price of zero or negative is invalid")

mask = listed["LivingArea"].notna() & (listed["LivingArea"] <= 0)
print(f"  LivingArea <= 0: {mask.sum():,} records flagged")
before = len(listed)
listed = listed[~mask].copy()
log("LISTED", "Remove LivingArea <= 0", before, len(listed),
    "Zero or negative living area makes price-per-sqft impossible")

mask = listed["DaysOnMarket"].notna() & (listed["DaysOnMarket"] < 0)
print(f"  DaysOnMarket < 0: {mask.sum():,} records flagged")
before = len(listed)
listed = listed[~mask].copy()
log("LISTED", "Remove DaysOnMarket < 0", before, len(listed),
    "Negative days on market is a data entry error")

mask = listed["BedroomsTotal"].notna() & (listed["BedroomsTotal"] < 0)
print(f"  BedroomsTotal < 0: {mask.sum():,} records flagged")
before = len(listed)
listed = listed[~mask].copy()
log("LISTED", "Remove BedroomsTotal < 0", before, len(listed),
    "Negative bedrooms is physically impossible")

mask = listed["BathroomsTotalInteger"].notna() & (listed["BathroomsTotalInteger"] < 0)
print(f"  BathroomsTotalInteger < 0: {mask.sum():,} records flagged")
before = len(listed)
listed = listed[~mask].copy()
log("LISTED", "Remove BathroomsTotalInteger < 0", before, len(listed),
    "Negative bathrooms is physically impossible")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DATE CONSISTENCY FLAGS
# ─────────────────────────────────────────────────────────────────────────────
# These flags mark records with logically impossible date sequences.
# We FLAG rather than remove — these records may still have valid price data
# and we don't want to silently lose them. The flags let analysts filter
# them out in Tableau when date-dependent metrics are being used.
#
# The correct chronological order is:
#   ListingContractDate → PurchaseContractDate → CloseDate
print("\n" + "=" * 65)
print("DATE CONSISTENCY FLAGS")
print("=" * 65)

for df, name in [(sold, "SOLD"), (listed, "LISTED")]:
    # listing_after_close_flag: listing date is after the close date
    # This means the property appears to have sold before it was listed — impossible
    if "ListingContractDate" in df.columns and "CloseDate" in df.columns:
        df["listing_after_close_flag"] = (
            df["ListingContractDate"].notna() &
            df["CloseDate"].notna() &
            (df["ListingContractDate"] > df["CloseDate"])
        )
        count = df["listing_after_close_flag"].sum()
        print(f"  [{name}] listing_after_close_flag   : {count:,} records")

    # purchase_after_close_flag: contract date is after close date
    # This means the purchase contract was signed after the deal already closed
    if "PurchaseContractDate" in df.columns and "CloseDate" in df.columns:
        df["purchase_after_close_flag"] = (
            df["PurchaseContractDate"].notna() &
            df["CloseDate"].notna() &
            (df["PurchaseContractDate"] > df["CloseDate"])
        )
        count = df["purchase_after_close_flag"].sum()
        print(f"  [{name}] purchase_after_close_flag  : {count:,} records")

    # negative_timeline_flag: either of the above is true
    # This is the master flag used to filter out all timeline-invalid records
    df["negative_timeline_flag"] = (
        df.get("listing_after_close_flag",  pd.Series(False, index=df.index)) |
        df.get("purchase_after_close_flag", pd.Series(False, index=df.index))
    )
    count = df["negative_timeline_flag"].sum()
    print(f"  [{name}] negative_timeline_flag      : {count:,} records")

# ─────────────────────────────────────────────────────────────────────────────
# 6. GEOGRAPHIC DATA QUALITY FLAGS
# ─────────────────────────────────────────────────────────────────────────────
# Geographic flags identify records that can't be reliably plotted on a map.
# Again we FLAG rather than remove — a record with a bad coordinate still has
# valid price and transaction data useful for non-geographic analysis.
#
# California-specific rules:
#   Latitude  should be between ~32.5 and ~42.0  (north-south span of CA)
#   Longitude should be between ~-124.5 and ~-114.0  (west-east span of CA)
#   Longitude must be NEGATIVE — positive longitude means somewhere in Asia
print("\n" + "=" * 65)
print("GEOGRAPHIC DATA QUALITY FLAGS")
print("=" * 65)

CA_LAT_MIN, CA_LAT_MAX   =  32.5,  42.0
CA_LON_MIN, CA_LON_MAX   = -124.5, -114.0

for df, name in [(sold, "SOLD"), (listed, "LISTED")]:
    # flag_missing_coords: Latitude or Longitude is null
    df["flag_missing_coords"] = (
        df["Latitude"].isna() | df["Longitude"].isna()
    )
    print(f"  [{name}] flag_missing_coords  : {df['flag_missing_coords'].sum():,}")

    # flag_zero_coords: Latitude or Longitude is exactly 0
    # Zero is a sentinel null value sometimes used instead of leaving blank
    df["flag_zero_coords"] = (
        (df["Latitude"] == 0) | (df["Longitude"] == 0)
    )
    print(f"  [{name}] flag_zero_coords     : {df['flag_zero_coords'].sum():,}")

    # flag_positive_long: Longitude is positive (should always be negative for CA)
    df["flag_positive_long"] = df["Longitude"] > 0
    print(f"  [{name}] flag_positive_long   : {df['flag_positive_long'].sum():,}")

    # flag_out_of_range_coords: coordinates fall outside California's bounding box
    # This catches records where coordinates exist but are clearly wrong
    df["flag_out_of_range_coords"] = (
        df["Latitude"].notna() & df["Longitude"].notna() &
        ~df["flag_missing_coords"] & ~df["flag_zero_coords"] &
        (
            (df["Latitude"]  < CA_LAT_MIN) | (df["Latitude"]  > CA_LAT_MAX) |
            (df["Longitude"] < CA_LON_MIN) | (df["Longitude"] > CA_LON_MAX)
        )
    )
    print(f"  [{name}] flag_out_of_range    : {df['flag_out_of_range_coords'].sum():,}")

    # flag_any_geo_issue: master geographic flag — any of the above is true
    df["flag_any_geo_issue"] = (
        df["flag_missing_coords"]    |
        df["flag_zero_coords"]       |
        df["flag_positive_long"]     |
        df["flag_out_of_range_coords"]
    )
    print(f"  [{name}] flag_any_geo_issue   : {df['flag_any_geo_issue'].sum():,}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 7. STRING FIELD CLEANUP
# ─────────────────────────────────────────────────────────────────────────────
# Strip leading/trailing whitespace and normalize 'nan' strings back to NaN.
# When pandas reads a CSV, missing string values sometimes come back as the
# literal string 'nan' rather than a true null — this fixes that.
print("=" * 65)
print("STRING FIELD CLEANUP")
print("=" * 65)

str_cols = [
    "City", "StateOrProvince", "CountyOrParish",
    "PropertyType", "PropertySubType", "MlsStatus",
    "ListAgentFullName", "ListOfficeName",
    "BuyerAgentFirstName", "BuyerAgentLastName", "BuyerOfficeName",
    "SubdivisionName", "HighSchool", "HighSchoolDistrict",
]

for df, name in [(sold, "SOLD"), (listed, "LISTED")]:
    for c in str_cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.strip()
                .replace("nan", np.nan)
                .replace("", np.nan)
            )
    print(f"  [{name}] string fields cleaned")

# ─────────────────────────────────────────────────────────────────────────────
# 8. POSTALCODE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
# PostalCode should always be a 5-digit string.
# Raw data can have floats (92101.0), 9-digit codes (92101-1234), or garbage.
# We extract the first 5 digits and flag any that don't match CA zip ranges.
# CA zip codes run from 90001 to 96162.
print("\n" + "=" * 65)
print("POSTALCODE VALIDATION")
print("=" * 65)

for df, name in [(sold, "SOLD"), (listed, "LISTED")]:
    if "PostalCode" in df.columns:
        # Standardise to 5-digit string
        df["PostalCode"] = (
            df["PostalCode"]
            .astype(str)
            .str.extract(r"(\d{5})", expand=False)
        )
        # Flag zip codes outside California's range (90001-96162)
        zip_numeric = pd.to_numeric(df["PostalCode"], errors="coerce")
        df["flag_invalid_zip"] = (
            zip_numeric.isna() |
            (zip_numeric < 90001) |
            (zip_numeric > 96162)
        )
        invalid = df["flag_invalid_zip"].sum()
        print(f"  [{name}] PostalCode standardised | invalid/out-of-range zips: {invalid:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("CLEANING SUMMARY")
print("=" * 65)

print(f"\nSOLD")
print(f"  Rows at start  : {sold_original_rows:,}")
print(f"  Rows at end    : {len(sold):,}")
print(f"  Total removed  : {sold_original_rows - len(sold):,} "
      f"({(sold_original_rows - len(sold)) / sold_original_rows * 100:.2f}%)")

print(f"\nLISTED")
print(f"  Rows at start  : {listed_original_rows:,}")
print(f"  Rows at end    : {len(listed):,}")
print(f"  Total removed  : {listed_original_rows - len(listed):,} "
      f"({(listed_original_rows - len(listed)) / listed_original_rows * 100:.2f}%)")

# Flag summary
print(f"\nSOLD — flag column counts:")
flag_cols = [c for c in sold.columns if c.startswith("flag_") or c.endswith("_flag")]
for c in flag_cols:
    print(f"  {c:<35} : {sold[c].sum():,}")

print(f"\nLISTED — flag column counts:")
flag_cols = [c for c in listed.columns if c.startswith("flag_") or c.endswith("_flag")]
for c in flag_cols:
    print(f"  {c:<35} : {listed[c].sum():,}")

# Data type confirmation
print(f"\nSOLD — key column dtypes:")
key_cols = ["CloseDate", "ListingContractDate", "ClosePrice", "LivingArea",
            "DaysOnMarket", "Latitude", "Longitude"]
for c in key_cols:
    if c in sold.columns:
        print(f"  {c:<30} : {sold[c].dtype}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SAVING OUTPUTS")
print("=" * 65)

sold.to_csv(  f"{OUTPUT_DIR}/crmls_sold_cleaned.csv",   index=False)
listed.to_csv(f"{OUTPUT_DIR}/crmls_listed_cleaned.csv", index=False)

# Save cleaning log as a report CSV
cleaning_report = pd.DataFrame(cleaning_log)
cleaning_report.to_csv(f"{OUTPUT_DIR}/cleaning_report.csv", index=False)

print(f"  crmls_sold_cleaned.csv    — {sold.shape[0]:,} rows x {sold.shape[1]} cols")
print(f"  crmls_listed_cleaned.csv  — {listed.shape[0]:,} rows x {listed.shape[1]} cols")
print(f"  cleaning_report.csv       — {len(cleaning_report)} transformation steps logged")

print("\nDone.")