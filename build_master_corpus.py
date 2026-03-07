"""
build_master_corpus.py

Cleans and merges:
  1. Comprehensive_State_Law_List (manually curated, 480 bills)
  2. results_20260307_111036.csv (pipeline-collected, 698 bills)

Output: master_corpus.csv + master_corpus.parquet
  - Deduplicated on state + normalized bill number
  - Provenance tracked in 'collection_method' column
  - Normalized state names, bill numbers, status
"""

import pandas as pd
import re
from pathlib import Path

# ── Normalization helpers ─────────────────────────────────────────────────────

STATE_MAP = {
    'NE': 'Nebraska', 'MS': 'Mississippi',
    'UT': 'Utah', 'WY': 'Wyoming',
    'District of Colombia': 'District of Columbia',
}

STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
    'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
}


def normalize_bill_number(s):
    """Normalize bill number to e.g. HB1336, SB299, HJR19."""
    s = str(s).strip().upper()
    s = re.sub(r'^[A-Z]{2}\s+', '', s)           # remove state prefix: 'AL HB209' -> 'HB209'
    s = re.sub(r'([A-Z])\.([A-Z])\.', r'\1\2', s) # H.B. -> HB
    s = re.sub(r'([A-Z]+)\s+(\d+)', r'\1\2', s)   # HB 209 -> HB209
    return s.strip()


def normalize_status(s):
    s = str(s).upper()
    if any(x in s for x in ['ENACT', 'CHAPTER', 'SIGNED', 'PUBLIC ACT']):
        return 'Enacted'
    if any(x in s for x in ['PASS', 'ENROLL']):
        return 'Passed'
    return 'Other'


# ── Load and clean master spreadsheet ────────────────────────────────────────

print("Loading master spreadsheet...")
master = pd.read_csv('/Users/Shea/Desktop/Comprehensive_State_Law_List_UPDATED.xlsx - Sheet1.csv')

master['state_name']    = master['State'].replace(STATE_MAP)
master['state_abbrev']  = master['state_name'].map(STATE_ABBREV)
master['bill_norm']     = master['Bill Number'].apply(normalize_bill_number)
master['year']          = pd.to_datetime(master['Last Action Date'], errors='coerce').dt.year.astype('Int64')
master['status_norm']   = master['Status'].apply(normalize_status)
master['collection_method'] = 'Manual_Curated'

master_clean = master[[
    'state_name', 'state_abbrev', 'bill_norm', 'Title',
    'Legiscan Bill Main Page', 'Legiscan Text Page',
    'year', 'status_norm', 'collection_method'
]].rename(columns={
    'state_name':           'state',
    'state_abbrev':         'state_abbrev',
    'bill_norm':            'bill_number',
    'Title':                'title',
    'Legiscan Bill Main Page': 'legiscan_bill_url',
    'Legiscan Text Page':   'legiscan_text_url',
    'status_norm':          'status',
})

print(f"  Master: {len(master_clean)} bills")

# ── Load and clean pipeline CSV ───────────────────────────────────────────────

print("Loading pipeline CSV...")
pipeline = pd.read_csv('output/results_20260307_111036.csv')

# Skip federal bills
pipeline = pipeline[pipeline['State'].str.upper() != 'US'].copy()

pipeline['state_abbrev'] = pipeline['State'].str.strip().str.upper()
# Reverse lookup full name from abbrev
abbrev_to_name = {v: k for k, v in STATE_ABBREV.items()}
pipeline['state'] = pipeline['state_abbrev'].map(abbrev_to_name)
pipeline['bill_number'] = pipeline['Bill Number'].apply(normalize_bill_number)
pipeline['year'] = pd.to_datetime(pipeline['Last Action Date'], errors='coerce').dt.year.astype('Int64')
pipeline['status_norm'] = pipeline['Status'].apply(normalize_status)
pipeline['collection_method'] = 'API_Pipeline_Mar2026'

pipeline_clean = pipeline[[
    'state', 'state_abbrev', 'bill_number', 'Title',
    'LegiScan Bill Main Page URL', 'LegiScan Text Page URL',
    'year', 'status_norm', 'collection_method'
]].rename(columns={
    'Title':                    'title',
    'LegiScan Bill Main Page URL': 'legiscan_bill_url',
    'LegiScan Text Page URL':   'legiscan_text_url',
    'status_norm':              'status',
})

print(f"  Pipeline: {len(pipeline_clean)} bills (after removing US federal)")

# ── Merge and deduplicate ─────────────────────────────────────────────────────

print("Merging...")
combined = pd.concat([master_clean, pipeline_clean], ignore_index=True)

# Create merge key
combined['_merge_key'] = combined['state_abbrev'].fillna('') + '_' + combined['bill_number']

# Where a bill appears in both sources, keep master version but note both
dupes = combined[combined.duplicated('_merge_key', keep=False)]
both_sources = dupes['_merge_key'].unique()

combined.loc[
    combined['_merge_key'].isin(both_sources) & 
    (combined['collection_method'] == 'Manual_Curated'),
    'collection_method'
] = 'Both_Manual_and_API'

# Drop API-only duplicates where manual version exists
combined = combined.sort_values('collection_method').drop_duplicates('_merge_key', keep='first')
combined = combined.drop(columns=['_merge_key']).reset_index(drop=True)

print(f"  Combined (deduplicated): {len(combined)} bills")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*50)
print(f"  Total unique bills     : {len(combined)}")
print(f"  States covered         : {combined['state_abbrev'].nunique()}")
print(f"\n  Collection method breakdown:")
print(combined['collection_method'].value_counts().to_string())
print(f"\n  Status breakdown:")
print(combined['status'].value_counts().to_string())
print(f"\n  Year range: {combined['year'].min()} - {combined['year'].max()}")
print(f"\n  Missing states:")
all_states = set(STATE_ABBREV.values())
covered = set(combined['state_abbrev'].dropna().unique())
print(f"  {sorted(all_states - covered)}")
print("="*50)

# ── Save ──────────────────────────────────────────────────────────────────────

out_csv     = Path('output/master_corpus.csv')
out_parquet = Path('output/master_corpus.parquet')

combined.to_csv(out_csv, index=False)
combined.to_parquet(out_parquet, index=False)

print(f"\n  Saved: {out_csv}")
print(f"  Saved: {out_parquet}")
