# Feature engineering script
import pandas as pd
import numpy as np
 
print("Loading datasets...")
sold   = pd.read_csv('output/crmls_sold_final.csv',   low_memory=False)
listed = pd.read_csv('output/crmls_listed_final.csv', low_memory=False)
 
# ── Filter Residential only ───────────────────────────────────────────────────
sold   = sold[sold['PropertyType']   == 'Residential'].copy()
listed = listed[listed['PropertyType'] == 'Residential'].copy()
print(f"Residential sold: {len(sold):,}  |  listed: {len(listed):,}")
 
# ── Drop columns per spec ─────────────────────────────────────────────────────
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
 
# ── Parse dates ───────────────────────────────────────────────────────────────
date_cols = ['ListingContractDate','PurchaseContractDate','CloseDate','ContractStatusChangeDate']
for df in [sold, listed]:
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
 
# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (shared between both datasets)
# ─────────────────────────────────────────────────────────────────────────────
def dom_bucket(d):
    if pd.isna(d): return None
    d = float(d)
    if d <= 7:    return '1 - Very Fast (1-7d)'
    if d <= 30:   return '2 - Fast (8-30d)'
    if d <= 60:   return '3 - Average (31-60d)'
    if d <= 90:   return '4 - Slow (61-90d)'
    return '5 - Very Slow (90+d)'
 
def age_tier(y):
    if pd.isna(y): return None
    y = float(y)
    if y >= 2020: return 'New (2020+)'
    if y >= 2000: return 'Modern (2000-2019)'
    if y >= 1980: return 'Contemporary (1980-1999)'
    if y >= 1960: return 'Established (1960-1979)'
    return 'Vintage (pre-1960)'
 
def lot_tier(a):
    if pd.isna(a): return None
    a = float(a)
    if a < 0.1:   return 'Small (<0.1 ac)'
    if a < 0.25:  return 'Medium (0.1-0.25 ac)'
    if a < 0.5:   return 'Large (0.25-0.5 ac)'
    if a < 1.0:   return 'XL (0.5-1 ac)'
    return 'Estate (1+ ac)'
 
def bed_tier(b):
    if pd.isna(b): return None
    b = int(float(b))
    if b <= 1: return '1BR or less'
    if b == 2: return '2BR'
    if b == 3: return '3BR'
    if b == 4: return '4BR'
    return '5BR+'
 
price_bins   = [0, 400_000, 700_000, 1_000_000, 1_500_000, float('inf')]
price_labels = ['Under $400K', '$400K-$700K', '$700K-$1M', '$1M-$1.5M', '$1.5M+']
 
bool_yn = {True:'Yes', False:'No', 'True':'Yes', 'False':'No', 1:'Yes', 0:'No'}
 
# ═════════════════════════════════════════════════════════════════════════════
#  SOLD — feature engineering
# ═════════════════════════════════════════════════════════════════════════════
 
print("\nEngineering SOLD features...")
 
# -- Price ratios ---------------------------------------------------------------
sold['close_to_orig_ratio'] = np.where(
    sold['OriginalListPrice'].notna() & (sold['OriginalListPrice'] > 0),
    (sold['ClosePrice'] / sold['OriginalListPrice']).round(4), np.nan)
 
sold['close_to_list_ratio'] = np.where(
    sold['ListPrice'].notna() & (sold['ListPrice'] > 0),
    (sold['ClosePrice'] / sold['ListPrice']).round(4), np.nan)
 
# -- Price per sq ft -----------------------------------------------------------
sold['price_per_sqft'] = np.where(
    sold['LivingArea'].notna() & (sold['LivingArea'] > 0),
    (sold['ClosePrice'] / sold['LivingArea']).round(2), np.nan)
 
# -- Time dimensions -----------------------------------------------------------
sold['close_year']    = sold['CloseDate'].dt.year.astype('Int64')
sold['close_month']   = sold['CloseDate'].dt.month.astype('Int64')
sold['close_quarter'] = sold['CloseDate'].dt.quarter.astype('Int64')
sold['close_yrmo']    = sold['CloseDate'].dt.to_period('M').astype(str)
sold['close_yrqtr']   = (sold['close_year'].astype(str) + '-Q' + sold['close_quarter'].astype(str))
 
# -- Timeline durations --------------------------------------------------------
sold['days_listing_to_contract'] = (
    sold['PurchaseContractDate'] - sold['ListingContractDate']).dt.days
 
sold['days_contract_to_close'] = (
    sold['CloseDate'] - sold['PurchaseContractDate']).dt.days
 
sold['days_list_to_close'] = (
    sold['CloseDate'] - sold['ListingContractDate']).dt.days
 
# -- Date consistency flags ----------------------------------------------------
sold['flag_listing_after_close']  = (sold['ListingContractDate']  > sold['CloseDate'])
sold['flag_purchase_after_close'] = (sold['PurchaseContractDate'] > sold['CloseDate'])
sold['flag_negative_timeline']    = (
    sold['flag_listing_after_close'] | sold['flag_purchase_after_close'])
 
# -- Geographic flags ----------------------------------------------------------
sold['flag_missing_coords'] = sold['Latitude'].isna() | sold['Longitude'].isna()
sold['flag_zero_coords']    = (sold['Latitude'] == 0)  | (sold['Longitude'] == 0)
sold['flag_positive_long']  = sold['Longitude'] > 0
 
# -- Market condition (vs original list) ----------------------------------------
sold['market_condition'] = pd.cut(
    sold['close_to_orig_ratio'],
    bins=[-np.inf, 0.9999, 1.0, np.inf],
    labels=['Below Ask', 'At Ask', 'Above Ask'])
 
# -- DOM bucket ----------------------------------------------------------------
sold['dom_bucket']  = sold['DaysOnMarket'].apply(dom_bucket)
 
# -- Age tier ------------------------------------------------------------------
sold['age_tier']    = sold['YearBuilt'].apply(age_tier)
 
# -- Price tier (ClosePrice) ---------------------------------------------------
sold['price_tier']  = pd.cut(sold['ClosePrice'],
    bins=price_bins, labels=price_labels, right=False)
 
# -- Lot tier ------------------------------------------------------------------
sold['lot_tier']    = sold['LotSizeAcres'].apply(lot_tier)
 
# -- Bed tier ------------------------------------------------------------------
sold['bed_tier']    = sold['BedroomsTotal'].apply(bed_tier)
 
# -- Bool → Yes/No labels ------------------------------------------------------
bool_cols_sold = ['NewConstructionYN','AttachedGarageYN','FireplaceYN',
                  'PoolPrivateYN','ViewYN','BasementYN','WaterfrontYN','HasHOA']
for c in bool_cols_sold:
    if c in sold.columns:
        sold[f'{c}_label'] = sold[c].map(bool_yn)
 
# ═════════════════════════════════════════════════════════════════════════════
#  LISTED — feature engineering
# ═════════════════════════════════════════════════════════════════════════════
 
print("Engineering LISTED features...")
 
# -- Price per sq ft (list price) ----------------------------------------------
listed['list_price_per_sqft'] = np.where(
    listed['LivingArea'].notna() & (listed['LivingArea'] > 0),
    (listed['ListPrice'] / listed['LivingArea']).round(2), np.nan)
 
# -- Close-to-orig ratio (closed listings only) --------------------------------
listed['close_to_orig_ratio'] = np.where(
    listed['ClosePrice'].notna() & listed['OriginalListPrice'].notna() & (listed['OriginalListPrice'] > 0),
    (listed['ClosePrice'] / listed['OriginalListPrice']).round(4), np.nan)
 
# -- Time dimensions (listing date) --------------------------------------------
listed['listing_year']    = listed['ListingContractDate'].dt.year.astype('Int64')
listed['listing_month']   = listed['ListingContractDate'].dt.month.astype('Int64')
listed['listing_quarter'] = listed['ListingContractDate'].dt.quarter.astype('Int64')
listed['listing_yrmo']    = listed['ListingContractDate'].dt.to_period('M').astype(str)
listed['listing_yrqtr']   = (listed['listing_year'].astype(str) + '-Q' + listed['listing_quarter'].astype(str))
 
# -- Days from listing to contract (where available) ---------------------------
listed['days_listing_to_contract'] = np.where(
    listed['PurchaseContractDate'].notna(),
    (listed['PurchaseContractDate'] - listed['ListingContractDate']).dt.days, np.nan)
 
# -- Price reduction label -----------------------------------------------------
listed['had_price_reduction'] = (
    listed['PriceReduction'].notna() & (listed['PriceReduction'] > 0)).map({True:'Yes', False:'No'})
 
# -- Status label (human-readable) ---------------------------------------------
status_map = {
    'Active':'Active', 'Pending':'Pending',
    'ActiveUnderContract':'Active Under Contract',
    'Closed':'Closed', 'ComingSoon':'Coming Soon',
    'Expired':'Expired', 'Withdrawn':'Withdrawn', 'Canceled':'Canceled',
}
listed['status_label']    = listed['MlsStatus'].map(status_map).fillna(listed['MlsStatus'])
listed['is_active_supply'] = listed['MlsStatus'].isin(
    ['Active','ActiveUnderContract','ComingSoon']).map({True:'Yes',False:'No'})
 
# -- DOM / age / lot / bed tiers -----------------------------------------------
listed['dom_bucket'] = listed['DaysOnMarket'].apply(dom_bucket)
listed['age_tier']   = listed['YearBuilt'].apply(age_tier)
listed['lot_tier']   = listed['LotSizeAcres'].apply(lot_tier)
listed['bed_tier']   = listed['BedroomsTotal'].apply(bed_tier)
 
# -- Price tier (ListPrice) ----------------------------------------------------
listed['price_tier'] = pd.cut(listed['ListPrice'],
    bins=price_bins, labels=price_labels, right=False)
 
# -- Bool → Yes/No -------------------------------------------------------------
bool_cols_listed = ['NewConstructionYN','AttachedGarageYN','FireplaceYN','HasHOA']
for c in bool_cols_listed:
    if c in listed.columns:
        listed[f'{c}_label'] = listed[c].map(bool_yn)
 
# -- Geographic flags ----------------------------------------------------------
listed['flag_missing_coords'] = listed['Latitude'].isna() | listed['Longitude'].isna()
listed['flag_zero_coords']    = (listed['Latitude'] == 0) | (listed['Longitude'] == 0)
listed['flag_positive_long']  = listed['Longitude'] > 0
 
# ═════════════════════════════════════════════════════════════════════════════
#  Validation
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== SOLD validation ===")
print(f"Shape: {sold.shape}")
num_check_sold = ['close_to_orig_ratio','close_to_list_ratio','price_per_sqft',
                  'days_listing_to_contract','days_contract_to_close','days_list_to_close']
for c in num_check_sold:
    s = sold[c].dropna()
    print(f"  {c}: median={s.median():.2f}, nulls={sold[c].isnull().sum()}")
 
cat_check_sold = ['market_condition','dom_bucket','age_tier','price_tier','bed_tier']
for c in cat_check_sold:
    print(f"  {c}: {dict(sold[c].value_counts().head(3))}")
 
print(f"\n  flag_negative_timeline:  {sold['flag_negative_timeline'].sum():,}")
print(f"  flag_missing_coords:     {sold['flag_missing_coords'].sum():,}")
print(f"  flag_positive_long:      {sold['flag_positive_long'].sum():,}")
 
print("\n=== LISTED validation ===")
print(f"Shape: {listed.shape}")
for c in ['list_price_per_sqft','days_listing_to_contract']:
    s = listed[c].dropna()
    print(f"  {c}: median={s.median():.2f}, nulls={listed[c].isnull().sum()}")
for c in ['status_label','dom_bucket','had_price_reduction','is_active_supply']:
    print(f"  {c}: {dict(listed[c].value_counts().head(3))}")
 
# ═════════════════════════════════════════════════════════════════════════════
#  Save outputs
# ═════════════════════════════════════════════════════════════════════════════
print("\nSaving outputs...")
sold.to_csv('output/crmls_sold_engineered.csv', index=False)
listed.to_csv('output/crmls_listed_engineered.csv', index=False)
print(f"  Saved: crmls_sold_engineered.csv   ({sold.shape[0]:,} rows x {sold.shape[1]} cols)")
print(f"  Saved: crmls_listed_engineered.csv ({listed.shape[0]:,} rows x {listed.shape[1]} cols)")
print("\nDone.")
 