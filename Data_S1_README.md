# Data S1: Comprehensive annotation of 547 multi-omic components

Data S1 of Whalley, Hunsberger, Bennett, Lamar, Peterson and Stutzmann, *Multi-omic Analysis of Sex and Alzheimer's Disease in the Aged Human Cortex*. `Data_S1.csv` has one row for each component of the 547-component catalogue (IDs 1-519 and 521-548). The catalogue comes from a Sparse Decomposition of Arrays (SDA) of snRNA-seq, DNA methylation, H3K9ac ChIP-seq and TMT proteomics data from 276 ROSMAP donors. The 29 columns give:
- the component ID and the number of modalities with selected features;
- the leading cell types;
- donor concentration;
- unadjusted and covariate-adjusted tests against sex, CERAD score and cognitive diagnosis;
- the best-matching SEA-AD component;
- the prioritization;
- the top RNA, CpG and H3K9ac features, and RNA and CpG KEGG terms;
- flags for the proteomics identifier error;
- provenance.

The definitions below expand the Data S1 legend in the Supplementary Information, and the analyses are described in the manuscript Methods. This version replaces earlier versions of the table (546 rows). The differences are noted under the affected columns.

## File format

- Comma-separated values, UTF-8 (ASCII characters only), LF line endings, no byte-order mark. One header row, then 547 rows in ascending `Component` order.
- Fields are double-quoted only when they contain a comma (two KEGG term names).
- Cells with several entries separate them with "; " (`prioritisation_support`, `protein_flag`, `provenance`).
- A blank cell means either that the column does not apply to the component or that no value met the reporting threshold. Each entry below says which. No NA strings are used.
- `top_donor_share` has four decimal places and `participation_ratio` one. The other numeric columns are not rounded to a fixed number of decimal places (up to 17 digits) and use scientific notation when small (for example `7.925535606290116e-44`). `SEA-AD_comp` holds integer IDs written with one decimal (`96.0`).
- Select and join by `Component`, not by row number: from ID 521 on, the row position is one less than the ID.
- "Positive" and "negative" refer to the loadings as deposited in syn75275226. The sign of any single SDA loading is arbitrary. The direction of an effect is given by loading products: sample × cell-type × gene loadings for snRNA-seq, and sample × feature loadings for the other layers (Methods).

## Columns

### Component and cell types

1. **`Component`**: component ID (integer). The run count of the original clustering was corrected. As a result, Component 520 (found in 4 of the 10 SDA runs) is retired, and Components 547 and 548 (5 runs each) are added. Every other component keeps the ID, members, sample loadings and feature loadings of earlier versions; its cell-type loadings were corrected (columns 3 and 4).
2. **`omics_count`**: number of the four fitted modalities (snRNA-seq, DNA methylation, H3K9ac, proteomics) with at least one feature at posterior inclusion probability (PIP) > 0.5 (0 to 4). Proteomics is counted. Protein features contribute to the count in 20 rows, and all 20 carry a `protein_flag`.
3. **`Top_positive_cell_type`**: the cell type with the largest positive cell-type loading (`cell_type.csv` in the deposit), reported where that direction is supported by the individual SDA runs. Values are `Ast` (astrocytes), `Exc` (excitatory neurons), `Inh` (inhibitory neurons), `Mic` (microglia), `OPC` (oligodendrocyte precursor cells) or `Oli` (oligodendrocytes).
   - Blank when no cell type is reported for that direction. A label recomputed from `cell_type.csv` alone can therefore appear where Data S1 is blank.
   - These columns use the corrected cell-type loadings. Each run-level component's cell-type sign now follows its sample and gene signs, so its sample × cell-type × gene products are preserved (Methods). Labels differ from earlier versions in 316 of the 545 retained rows.
4. **`Top_negative_cell_type`**: as `Top_positive_cell_type`, for the most negative loading.

### Donor concentration

For a component's sample scores sᵢ over the 276 donors, centered as dᵢ = sᵢ − s̄:

5. **`top_donor_share`**: the largest single-donor share of the sum of squares, max dᵢ² / Σ dᵢ² (four decimal places). It runs from 1/276 (evenly spread) to 275/276 (one donor carries the whole score).
6. **`participation_ratio`**: (Σ dᵢ²)² / Σ dᵢ⁴, the concentration of the score expressed as a number of donors (one decimal place). It is about 1 (1.007 for 276 donors) when one donor carries the score, 276 when all donors are equally far from the mean, and about 92 for a Gaussian score. It describes how the score is spread; it is not an effective sample size for inference.
7. **`concentration_class`**: `distributed` (top-donor share < 0.10; 193 components), `few-donor` (0.10 to < 0.50; 76) or `concentrated` (≥ 0.50; 278; "donor-concentrated" in the manuscript).
   - In 20,000 simulations of a Gaussian score over 276 donors, the top-donor share has a median of 0.033 and a 99.9th percentile of about 0.07. It reaches 0.10 in 0.005% of simulations.
   - The upper threshold lies in the trough between the continuous tail of the catalogue and the components carried by one donor. Nine components fall between 0.50 and 0.80, and 269 exceed 0.80, each dominated by a different donor.
   - Few-donor components mix heavy-tailed components that span the cohort with components carried by a small number of donors. They are not prioritized.

### Phenotype associations

Each value is a Benjamini-Hochberg false discovery rate (FDR), corrected within the phenotype over the 547 components. A blank cell means FDR ≥ 0.05. The groups are:
- Sex: 129 male and 147 female donors.
- CERAD neuritic plaque score, dichotomized as Alzheimer's disease (scores 1-2, definite or probable; 170 donors) against non-AD (3-4, possible or no AD; 106).
- Final clinical cognitive diagnosis, in three groups: no cognitive impairment (cogdx 1; 97), mild cognitive impairment (2; 65) and Alzheimer's dementia (4; 92). The other 22 donors are excluded.

8. **`Sex`**: unadjusted two-sided Mann-Whitney U test of the component score between male and female donors (3 values).
9. **`ceradsc`**: unadjusted two-sided Mann-Whitney U test between the two CERAD groups (42 values).
10. **`cogdx`**: unadjusted Kruskal-Wallis test across the three diagnosis groups (28 values).
11. **`Sex_adj_q`**: covariate-adjusted test from the main model of the Methods (3 values). The model is a proportional-odds (cumulative-logit) regression of the component score, treated as ordinal, on the phenotype and covariates. The phenotype is tested by likelihood ratio (1 degree of freedom for sex and CERAD, 2 for cognitive diagnosis). The covariates are age at death (coded 90 for donors whose age is released as 90 or older), post-mortem interval, cohort (ROS or MAP) and, where sex is not the phenotype, sex.
12. **`ceradsc_adj_q`**: as `Sex_adj_q`, for the CERAD groups (27 values).
13. **`cogdx_adj_q`**: as `Sex_adj_q`, for the three diagnosis groups (33 values).

The unadjusted columns hold 73 calls and are descriptive. The adjusted columns hold 63: 57 of the 73 unadjusted calls and 6 new cognitive-diagnosis calls. With the corrected run count, the unadjusted FDRs were recomputed over 547 components, and Component 548 adds one CERAD call.

### SEA-AD

14. **`SEA-AD_comp`**: the best-matching component in an SDA decomposition of the Seattle Alzheimer's Disease Brain Cell Atlas (SEA-AD) snRNA-seq data (dorsolateral prefrontal cortex, 82 donors). That decomposition has 141 components that recur in at least five runs.
      - Matching: the component's PIP > 0.5 genes that are measured in SEA-AD are compared with the same genes' loadings in every SEA-AD component. The partner is the SEA-AD component with the highest absolute Pearson correlation.
      - IDs keep the numbering of the original 143 SEA-AD components. Components 9 and 128 were retired with the corrected run count and do not appear. The 14 rows that named them in earlier versions now name their best partner among the 141.
      - Blank when fewer than two genes are shared (38 rows). 509 components have a partner.
15. **`SEA-AD_corr`**: Pearson correlation between the two loading vectors over the shared genes. Its sign reflects the arbitrary orientation of the two components, so the strength of the match is |r|. Blank when `SEA-AD_comp` is blank.
16. **`SEA-AD_spearman`**: Spearman correlation over the same genes and partner. Its sign can differ from that of `SEA-AD_corr`. With few shared genes either correlation can be at or near ±1; the Spearman correlation is ±1 in three rows, which share three or four genes. Blank when `SEA-AD_comp` is blank.

### Prioritization

17. **`prioritised_2026`**: `yes` for the 113 prioritized components, otherwise blank. A component is prioritized if it is in the `distributed` class and has at least one support in `prioritisation_support`.
18. **`prioritisation_support`**: the supports met, for prioritized components only:
      - `phenotype(...)`: a covariate-adjusted FDR < 0.05 (columns 11-13) for each phenotype named in the parentheses (`sex`, `cerad` or `cogdx`). Several phenotypes are separated by ";" without a space, for example `phenotype(cerad;cogdx)`. 34 components.
      - `GWAS`: a genome-wide association of the component score on autosomal SNPs at the study-wide threshold p < 5 × 10⁻⁸ / 547 = 9.1 × 10⁻¹¹, with a genomic inflation factor between 0.95 and 1.05. 1 component.
      - `SEA-AD`: an RNA signature replicated in SEA-AD. This requires |r| ≥ 0.4 and |ρ| ≥ 0.4 with the same sign (columns 15-16); a gene-permutation p < 0.01 (2,000 permutations, repeating the partner search over all 141 SEA-AD components); and a partner whose own top-sample share over the 82 SEA-AD donors is below 0.5. 96 components. The permutation p and the partner's share are not in this file, so the correlation condition alone selects more components: 104 of the 193 distributed components meet it.

### Top features and pathways

Loadings are those of the deposited matrices (`RNA.txt`, `meth.txt`, `chip.txt`), which hold values only for features with PIP > 0.5; the PIPs themselves are not deposited. A component's loading is the mean over its run-level members, and its PIP is their median.

19. **`Top_Positive_RNA`**: among the genes with PIP > 0.5, the one with the highest positive loading (gene symbol as in `RNA.txt`). Blank when no PIP > 0.5 gene has a positive loading.
20. **`Top_Negative_RNA`**: as `Top_Positive_RNA`, for the most negative loading.
21. **`GeneExpression_Pathway`**: the KEGG pathway (KEGG 2021 Human, Enrichr through gseapy) with the smallest adjusted p (ties broken by the smaller unadjusted p) for the component's PIP > 0.5 genes, tested against the 25,098 genes of the RNA input. Blank when the component has no PIP > 0.5 gene or no pathway reaches FDR < 0.05.
22. **`Top_Positive_meth`**: as `Top_Positive_RNA`, for CpG sites (Illumina HumanMethylation450 probe IDs as in `meth.txt`; GRCh37 coordinates in `meth_features.csv`).
23. **`Top_Negative_meth`**: as `Top_Negative_RNA`, for CpG sites.
24. **`CpG_Site_Pathway`**: as `GeneExpression_Pathway`, for the genes annotated to the PIP > 0.5 CpG sites (manifest RefGene annotation), tested against the 19,872 RefGene-annotated genes of the methylation array.
      - Component 74 used Enrichr's default background. Against the 19,872 genes its top term is unchanged.
      - These terms are not corrected for the number of array probes per gene and should be read as uncorrected annotations.
25. **`Top_Positive_chip`**: as `Top_Positive_RNA`, for H3K9ac peaks (peak IDs as in `chip.txt`; GRCh37 coordinates and peak class, promoter, enhancer or other, in `chip_features.csv`).
26. **`Top_Negative_chip`**: as `Top_Negative_RNA`, for H3K9ac peaks. There is no pathway column for H3K9ac peaks.

### Proteomics

An identifier error in the proteomics input matched 29 of the 105 proteomes to the wrong donors (Methods, Proteomics), so no protein-derived results are reported. The proteomics layer remains in the decomposition as a modeled input. The protein feature and pathway columns of earlier versions (`Top_Positive_proteins`, `Top_Negative_proteins`, `Protein_Pathway`) are withdrawn. The two columns below flag the components that meet diagnostic criteria for possible influence of the error and give the sensitivity of each component's scores to the corrected protein rows.

27. **`protein_flag`**: one or more codes, or blank. 38 components are flagged:
      - `P1`: at least one run-level member has protein features (PIP > 0.5), and that member's dominant donor (top-donor share above 0.5) had a proteomics row belonging to another person (23 components).
      - `P2`: protein features (PIP > 0.5) and scores that change when the protein rows are corrected (correlation below 0.95 in both fixed-loading implementations; 7).
      - `P3`: a donor-concentrated component whose dominant donor had another person's proteomics row, and whose dominant-donor score falls below 0.8 of its value when the protein rows are corrected (both implementations). 4 components, all also `P1`, written `P1; P3`.
      - `P4`: any other component with protein features (PIP > 0.5; 8).
28. **`score_r_corrected_protein`**: Pearson correlation over the 276 donors between two sets of re-estimated sample scores, with the component loadings held fixed. One set is re-estimated from the input as used, the other from the input with the protein rows rebuilt under the corrected mapping. Both use the implementation calibrated to the published scores (Methods).
      - It is not a correlation with the published scores.
      - Ten components fall below 0.99 (minimum 0.799, Component 356): the seven `P2` components, Component 56, and Components 544 and 506. Components 544 and 506 have no protein features, and their lower values reflect the calibration of this implementation.
      - These values do not assess changes in component discovery or feature loadings under a full refit.

### Provenance

29. **`provenance`**: begins with `D+A` in every row.
      - D is original discovery: ten SDA runs taken at iteration 1,500, keeping components that recur in at least five runs.
      - A is corrected aggregation: sign alignment and run count, cell-type columns, FDR over 547 components, and SEA-AD partners among 141 components.
      - Components 547 and 548 read `D+A; new ID (qualifies only under the corrected run count)`.
      - `omics_count` and the feature and pathway columns are derived from the original fitted run outputs, using the corrected catalogue and the feature-selection and enrichment procedures above. The cell-type, unadjusted phenotype and SEA-AD partner columns come from A.
      - `score_r_corrected_protein` and flags `P2` and `P3` come from the fixed-loading re-estimation, a sensitivity measure that replaces no value.
      - The concentration, `SEA-AD_spearman`, adjusted phenotype and prioritization columns come from analyses of the corrected catalogue.

## How to cite

Please cite the manuscript and the dataset:

- Whalley, J. P., Hunsberger, H. C., Bennett, D. A., Lamar, M., Peterson, D. A. & Stutzmann, G. E. Multi-omic Analysis of Sex and Alzheimer's Disease in the Aged Human Cortex. *npj Aging* (under review; citation and DOI to be added on publication).
- AD Knowledge Portal dataset syn75275226, version 2. https://doi.org/10.7303/syn75275226.2

Cite the ROSMAP and SEA-AD source data as in the manuscript's Data Availability statement.

## Code

`Data_S1.csv` is built by `scripts/round2/data_s1/build_Data_S1.py` in https://github.com/jpwhalley/rosmap-multiomic-sda (release v2.0.1). The repository also holds the code for the analyses behind each column.
