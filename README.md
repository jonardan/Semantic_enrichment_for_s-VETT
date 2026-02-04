# Semantic Enrichment Module

This module generates semantic descriptions for time series data using Google's Gemini LLM API.
The descriptions will be used in the s-VETT project

## Overview

For each column (time series) in your datasets, the module:
1. **Analyzes** the time series data (statistics, patterns, trends, seasonality)
2. **Reads** the unstructured description of the dataset
3. **Generates** a semantic description using Gemini LLM

The generated descriptions are designed for semantic search and retrieval of similar time series.

## Setup

### 1. Install dependencies

```bash
pip install google-generativeai
```

Or update all project dependencies:

```bash
pip install -e .
```

### 2. Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Set it as an environment variable:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"

# Windows CMD
set GEMINI_API_KEY=your-api-key-here

# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"
```

### 3. Prepare description files

For each dataset in `raw_data/`, create a corresponding `.txt` file in `unstructured_description_data/` with the same name:

```
raw_data/
├── energy_dataset.csv
├── weather.csv
└── traffic.csv

unstructured_description_data/
├── energy_dataset.txt
├── weather.txt
└── traffic.txt
```

The `.txt` files should contain any relevant information about the dataset:
- Source and context
- What the variables represent
- Expected patterns (seasonal, daily, etc.)
- Domain-specific knowledge

## Usage

### Command Line

Process all datasets:
```bash
python semantic_enrichment/description_generator.py
```

Process a specific dataset:
```bash
python semantic_enrichment/description_generator.py --dataset energy_dataset
```

Process with basic statistics and sample data included in the prompt:
```bash
python semantic_enrichment/description_generator.py --dataset energy_dataset --include-stats
```

With custom directories:
```bash
python semantic_enrichment/description_generator.py \
    --raw-data-dir raw_data \
    --description-dir unstructured_description_data \
    --output-dir generated_descriptions \
    --api-key YOUR_API_KEY
```

### Python API

```python
from semantic_enrichment import DatasetProcessor

processor = DatasetProcessor(
    raw_data_dir="raw_data",
    description_dir="unstructured_description_data", 
    output_dir="generated_descriptions",
    api_key="your-api-key"
)

# Process all datasets
results = processor.process_all_datasets()

# Or process a single dataset
from pathlib import Path
results = processor.process_dataset(Path("raw_data/energy_dataset.csv"))
```

## Output

The module generates JSON files with descriptions for each column:

```json
{
  "dataset_name": "energy_dataset",
  "columns": {
    "generation solar": {
      "description": "This time series represents hourly solar power generation..."
    }
  }
}
```

## Example Generated Description

For a solar generation column:

> "This time series represents hourly solar power generation in megawatts from the Spanish electricity grid, spanning from January 2015 to December 2018. The data exhibits strong daily patterns with generation occurring only during daylight hours, typically peaking between 12:00 and 15:00. Clear seasonal variation is present, with significantly higher generation during summer months (June-August) averaging around 4,500 MW compared to winter months (December-February) averaging approximately 1,200 MW. The series shows moderate volatility (CV=0.45) primarily driven by weather conditions and cloud cover. Values range from 0 MW (nighttime) to a maximum of 6,847 MW during peak summer days. This renewable energy source contributes to Spain's energy transition goals and shows a slight increasing trend over the observation period, reflecting capacity additions in the solar sector."

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--raw-data-dir` | `raw_data` | Directory with CSV datasets |
| `--description-dir` | `unstructured_description_data` | Directory with .txt descriptions |
| `--output-dir` | `generated_descriptions` | Output directory for JSON files |
| `--api-key` | `$GEMINI_API_KEY` | Gemini API key |
| `--model` | `gemini-1.5-flash` | Gemini model to use |
| `--dataset` | None | Process only this dataset |
| `--include-stats` | False | Include basic statistics and sample data in the LLM prompt |
