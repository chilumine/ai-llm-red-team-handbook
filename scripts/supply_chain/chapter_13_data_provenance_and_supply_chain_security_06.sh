#!/bin/bash
# AI LLM Red Team - 2. Data Dependencies
# Source: Chapter_13_Data_Provenance_and_Supply_Chain_Security
# Category: supply_chain

# Find data loading code
grep -r "pd.read_csv\|torch.load\|datasets.load" .

# Check for external data sources
grep -r "http.*download\|s3://\|gs://" .
