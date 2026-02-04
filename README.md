# Semantic Enrichment for s-VETT

This project generates semantic descriptions for CSV datasets by analyzing both structured data (CSV columns) and unstructured contextual information (text files). The descriptions are created for use in the semantic-VETT project.

## Overview

The tool reads CSV files from a `datasets` folder and creates comprehensive semantic descriptions for each column. It combines:
- **Structured data analysis**: Statistical analysis of CSV columns (data types, ranges, distributions, etc.)
- **Unstructured context**: Additional information from corresponding text files that provide domain knowledge and context

## Project Structure

```
Semantic_enrichment_for_s-VETT/
├── datasets/                    # Place your CSV files here
├── unstructured_data/          # Place corresponding .txt files here
├── generated_descriptions/     # Output folder for generated descriptions
├── semantic_enrichment.py      # Main processing script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

- Python 3.7+
- pandas 2.0.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jonardan/Semantic_enrichment_for_s-VETT.git
cd Semantic_enrichment_for_s-VETT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your data**:
   - Place CSV files in the `datasets/` folder
   - For each CSV file (e.g., `employees.csv`), create a corresponding text file in `unstructured_data/` (e.g., `employees.txt`)
   - The text file should contain contextual information about the dataset (purpose, field descriptions, business context, etc.)

2. **Run the script**:
```bash
python semantic_enrichment.py
```

3. **View results**:
   - Generated descriptions will be saved as JSON files in `generated_descriptions/`
   - Each column gets its own file: `{dataset_name}_{column_name}_description.json`

## Output Format

Each generated description includes:
- **column_name**: Name of the column
- **structured_analysis**: Statistical analysis including:
  - Data type
  - Null/non-null counts
  - Unique value count
  - Type-specific metrics (min/max/mean for numeric, samples for categorical)
- **semantic_description**: Human-readable description text
- **context_from_unstructured_data**: Relevant context from the text file
- **metadata**: Completeness and uniqueness ratios

## Example

Given `datasets/employees.csv` with columns `id`, `name`, `age`, `department`, `salary`, `hire_date`, and a corresponding `unstructured_data/employees.txt` file, the script will generate 6 JSON files:
- `employees_id_description.json`
- `employees_name_description.json`
- `employees_age_description.json`
- `employees_department_description.json`
- `employees_salary_description.json`
- `employees_hire_date_description.json`

Each file contains comprehensive semantic information about that specific column.

## Sample Data

The repository includes sample datasets for testing:
- `datasets/employees.csv` - Employee information dataset
- `datasets/products.csv` - Product inventory dataset
- Corresponding text files in `unstructured_data/`

You can run the script immediately to see how it works with these examples.

## Features

- ✅ Automatic processing of multiple CSV files
- ✅ Statistical analysis for numeric columns (min, max, mean, median)
- ✅ Categorical analysis for text columns (sample values, uniqueness)
- ✅ Data quality metrics (completeness, uniqueness ratios)
- ✅ Integration of unstructured contextual information
- ✅ Human-readable semantic descriptions
- ✅ JSON output format for easy integration with other tools

## License

MIT License - see LICENSE file for details
