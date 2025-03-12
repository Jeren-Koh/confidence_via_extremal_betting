# confidence_via_extremal_betting

Python implementation of a method for computing confidence sequences via betting strategies that are mixtures of extremal strategies, as described in "The Cost of Achieving the Best Portfolio in Hindsight" (Ordentlich and Cover, 1998).

## Files

- **betting_util.py**: Utility functions
- **extremal_conf.py**: Core functions for computing confidence sequences
- **priors.py**: Functions for calculating priors, represented as a numpy array of T+1 wealth allocations
- **csv_conf.py**: Calculates confidence sequences for input data in CSV format, using the functions in extremal_conf.py

## Usage

Use `csv_conf.py` to compute confidence sequences. Each row of the input CSV should contain one sequence with T columns.

Output is written to a new CSV file, with the lower and upper bounds of the confidence intervals on alternating rows.

### Command Line Arguments
python csv_conf.py delta prior_type csv_path out_path

- `delta`: Value between 0 and 1. The confidence level will be 1 - delta.
- `prior_type`: Prior type, either 'minmax' or 'truncated'
- `csv_path`: Path to input CSV file
- `out_path`: Path to output CSV file

### Example
python csv_conf.py 0.1 minmax data.csv outfile.csv
