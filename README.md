# Dissertation Data Pipeline

Computational infrastructure for PhD dissertation research on school shooting prevention legislation.

## Research Questions

**RQ1:** What policy strategies have states used to address school shootings?

- **Corpus:** State legislation 2010-2025 via LegiScan API
- **Methods:** Computational text analysis, structural topic modeling

**RQ2:** What demographic variables influence state policy strategy adoption?

- **DV:** Policy strategies by state (from RQ1)
- **IVs:** Implicit bias (IAT), gun ownership, partisanship, Christian nationalism, and others
- **Methods:** Logistic regression, chi-square, Cramer's V

## Repository Structure

```
├── data/
│   ├── raw/              # Pointers to OSF (data too large for GitHub)
│   ├── processed/        # Cleaned, aggregated datasets
│   │   ├── iat/          # State-year IAT aggregations
│   │   ├── legislation/  # Processed legislation data
│   │   └── state_covariates/  # State-level independent variables
│   ├── merged/           # Analysis-ready joined datasets
│   └── codebooks/        # Variable documentation
├── scripts/
│   ├── legislation/      # LegiScan API and text processing
│   ├── iat/              # IAT data aggregation
│   ├── state_covariates/ # Covariate data collection
│   └── merge/            # Dataset joining scripts
├── docs/
│   ├── transformation_log.md  # All data processing decisions
│   ├── data_sources.md        # Master source reference
│   └── fall2025/              # Archived independent study docs
├── archive/              # Previous deliverables
└── outputs/              # Figures and results
```

## Data Access

Raw data files are stored on OSF due to size: https://osf.io/qx7sh/

## Reproducibility

See `docs/transformation_log.md` for all data processing decisions.

## Author

Shea Swauger
PhD Candidate, Education and Human Development
University of Colorado Denver

## License

CC0 1.0 Universal - See LICENSE
