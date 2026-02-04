"""
Time Series Description Generator using Gemini LLM API

This module generates semantic descriptions for each time series column in a dataset
based on:
1. The column name/label
2. Statistical properties and patterns in the data
3. Unstructured description of the dataset from .txt files

The generated descriptions are designed for semantic search of time series.
"""
from dotenv import load_dotenv
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
import google.generativeai as genai
from structlog import get_logger

log = get_logger()


class TimeSeriesAnalyzer:
    """Analyzes time series data to extract meaningful statistics and patterns."""
    
    def __init__(self, series: pd.Series, timestamps: pd.Series):
        self.series = series
        self.timestamps = pd.to_datetime(timestamps)
        self.name = series.name
        
    def get_basic_stats(self) -> dict:
        """Calculate basic statistical properties."""
        return {
            'mean': float(self.series.mean()),
            'std': float(self.series.std()),
            'min': float(self.series.min()),
            'max': float(self.series.max()),
            'median': float(self.series.median()),
            'q25': float(self.series.quantile(0.25)),
            'q75': float(self.series.quantile(0.75)),
            'missing_pct': float(self.series.isna().sum() / len(self.series) * 100),
            'n_points': len(self.series),
        }
    
    def get_temporal_info(self) -> dict:
        """Extract temporal information."""
        return {
            'start_date': str(self.timestamps.min()),
            'end_date': str(self.timestamps.max()),
            'duration_days': (self.timestamps.max() - self.timestamps.min()).days,
            'frequency': self._detect_frequency(),
        }
    
    def _detect_frequency(self) -> str:
        """Detect the sampling frequency of the time series."""
        if len(self.timestamps) < 2:
            return "unknown"
        
        diff = self.timestamps.diff().dropna()
        median_diff = diff.median()
        
        if median_diff <= pd.Timedelta(minutes=1):
            return "sub-minute"
        elif median_diff <= pd.Timedelta(minutes=15):
            return "sub-hourly"
        elif median_diff <= pd.Timedelta(hours=1):
            return "hourly"
        elif median_diff <= pd.Timedelta(days=1):
            return "daily"
        elif median_diff <= pd.Timedelta(days=7):
            return "weekly"
        elif median_diff <= pd.Timedelta(days=31):
            return "monthly"
        else:
            return "irregular"
    
    def get_pattern_analysis(self) -> dict:
        """Analyze patterns in the time series."""
        patterns = {}
        
        # Trend analysis (simple linear regression slope)
        try:
            x = np.arange(len(self.series))
            valid_mask = ~self.series.isna()
            if valid_mask.sum() > 1:
                slope = np.polyfit(x[valid_mask], self.series[valid_mask], 1)[0]
                print('-------slope:', slope)
                if slope > 0.001 * self.series.std():
                    patterns['trend'] = 'increasing'
                elif slope < -0.001 * self.series.std():
                    patterns['trend'] = 'decreasing'
                else:
                    patterns['trend'] = 'stable'
                print('-------patterns["trend"]:', patterns['trend'])
            else:
                patterns['trend'] = 'unknown'
                
        except Exception:
            patterns['trend'] = 'unknown'
      
        # Volatility analysis
        try:
            cv = self.series.std() / abs(self.series.mean()) if self.series.mean() != 0 else 0
            if cv < 0.1:
                patterns['volatility'] = 'low'
            elif cv < 0.5:
                patterns['volatility'] = 'moderate'
            else:
                patterns['volatility'] = 'high'
        except Exception:
            patterns['volatility'] = 'unknown'
        
        return patterns
    
    def get_sample_data(self, n_samples: int = 10) -> list:
        """Get sample data points for context."""
        step = max(1, len(self.series) // n_samples)
        samples = []
        for i in range(0, len(self.series), step):
            if len(samples) >= n_samples:
                break
            samples.append({
                'timestamp': str(self.timestamps.iloc[i]),
                'value': float(self.series.iloc[i]) if not pd.isna(self.series.iloc[i]) else None
            })
        return samples
    
    def get_full_analysis(self, include_stats: bool = False) -> dict:
        """Get complete analysis of the time series.
        
        Args:
            include_stats: If True, include basic_stats and sample_data. If False, only include temporal_info and patterns.
        """
        analysis = {
            'column_name': str(self.name),
            'temporal_info': self.get_temporal_info(),
            'patterns': self.get_pattern_analysis(),
        }
        if include_stats:
            analysis['basic_stats'] = self.get_basic_stats()
            analysis['sample_data'] = self.get_sample_data()
        return analysis


class DescriptionGenerator:
    """Generates semantic descriptions for time series using Gemini 3.0 LLM."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro-latest"):
        """
        Initialize the description generator.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use (default: gemini-pro-latest)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        log.info("Initialized Gemini model", model=model_name)
    
    def _build_prompt(self, analysis: dict, dataset_description: str, dataset_name: str, include_stats: bool = False) -> str:
        """Build the prompt for the LLM."""
        
        prompt = f"""You are an expert in time series analysis and data description. Your task is to generate a comprehensive semantic description for a specific time series column that will be used for semantic search and retrieval (to retrieve models to forecast timeseries in similar domains).

## Dataset Information
**Dataset Name:** {dataset_name}

**Dataset Description (from documentation):**
{dataset_description if dataset_description else "No additional description available."}

## Time Series Column Analysis
**Column Name:** {analysis['column_name']}


**Temporal Information:**
- Start Date: {analysis['temporal_info']['start_date']}
- End Date: {analysis['temporal_info']['end_date']}
- Duration: {analysis['temporal_info']['duration_days']} days
- Sampling Frequency: {analysis['temporal_info']['frequency']}

**Detected Patterns:**
{json.dumps(analysis['patterns'], indent=2)}{('\\n\\n**Basic Statistics:**\\n' + json.dumps(analysis.get('basic_stats', {}), indent=2) + '\\n\\n**Sample Data Points:**\\n' + json.dumps(analysis.get('sample_data', [])[:5], indent=2)) if include_stats else ''}

## Instructions
Generate a detailed semantic description for this time series that:
1. Explains what this time series measures based on the column name and dataset context
2. Describes the temporal characteristics (frequency, duration, time period)
3. Describes any detected patterns (trends, seasonality, daily patterns)
4. Provides domain-specific insights based on the dataset description
5. Mentions any notable behaviors (e.g., "values tend to be higher during winter months" or "shows clear daily cycles with peaks in the afternoon")

The description should be:
- Written in clear, professional English
- Between 20-60 words
- Focused on characteristics useful for semantic search
- Specific to THIS column, not the general dataset

Generate ONLY the description, no additional commentary or formatting."""

        return prompt
    
    def generate_description(
        self, 
        analysis: dict, 
        dataset_description: str,
        dataset_name: str,
        include_stats: bool = False
    ) -> str:
        """
        Generate a semantic description for a time series.
        
        Args:
            analysis: Analysis dictionary from TimeSeriesAnalyzer
            dataset_description: Unstructured description of the dataset
            dataset_name: Name of the dataset
            include_stats: If True, include stats and sample data in the prompt
            
        Returns:
            Generated description string
        """
        prompt = self._build_prompt(analysis, dataset_description, dataset_name, include_stats)
        
        try:
            response = self.model.generate_content(prompt)
            description = response.text.strip()
            log.info("Generated description", column=analysis['column_name'], length=len(description))
            return description
        except Exception as e:
            log.error("Failed to generate description", column=analysis['column_name'], error=str(e))
            return f"Description generation failed: {str(e)}"


class DatasetProcessor:
    """Processes datasets and generates descriptions for all columns."""
    
    def __init__(
        self,
        raw_data_dir: str,
        description_dir: str,
        output_dir: str,
        api_key: str,
        model_name: str = "gemini-pro-latest",
        include_stats: bool = False
    ):
        """
        Initialize the dataset processor.
        
        Args:
            raw_data_dir: Directory containing raw CSV datasets
            description_dir: Directory containing .txt description files
            output_dir: Directory to save generated descriptions
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.description_dir = Path(description_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_stats = include_stats
        
        self.generator = DescriptionGenerator(api_key, model_name)
        log.info("Initialized DatasetProcessor", 
                 raw_data_dir=raw_data_dir,
                 description_dir=description_dir,
                 output_dir=output_dir)
    
    def _load_dataset(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load a dataset from file."""
        try:
            # Try to detect the timestamp column
            df = pd.read_csv(filepath)
            
            # Common timestamp column names
            timestamp_cols = ['time', 'timestamp', 'datetime', 'date', 'Datetime', 'Time', 'Date','Date Time']
            
            ts_col = None
            for col in timestamp_cols:
                if col in df.columns:
                    ts_col = col
                    break
            
            # If not found, assume first column is timestamp
            if ts_col is None:
                ts_col = df.columns[0]
            
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.set_index(ts_col)
            
            log.info("Loaded dataset", filepath=str(filepath), shape=df.shape)
            return df
        except Exception as e:
            log.error("Failed to load dataset", filepath=str(filepath), error=str(e))
            return None
    
    def _load_description(self, dataset_name: str) -> str:
        """Load the unstructured description for a dataset."""
        # Try different extensions
        for ext in ['.txt', '.md', '.TXT']:
            desc_path = self.description_dir / f"{dataset_name}{ext}"
            if desc_path.exists():
                try:
                    with open(desc_path, 'r', encoding='utf-8') as f:
                        description = f.read()
                    log.info("Loaded description", dataset=dataset_name)
                    return description
                except Exception as e:
                    log.error("Failed to load description", path=str(desc_path), error=str(e))
        
        log.warning("No description file found", dataset=dataset_name)
        return ""
    
    def process_dataset(self, filepath: Path) -> dict:
        """
        Process a single dataset and generate descriptions for all columns.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Dictionary with column descriptions
        """
        dataset_name = filepath.stem
        log.info("Processing dataset", name=dataset_name)
        
        # Load dataset
        df = self._load_dataset(filepath)
        if df is None:
            return {}
        
        # Load description
        dataset_description = self._load_description(dataset_name)
        
        # Process each column
        results = {
            'dataset_name': dataset_name,
            'dataset_path': str(filepath),
            'processed_at': datetime.now().isoformat(),
            'n_columns': len(df.columns),
            'columns': {}
        }
        
        timestamps = df.index.to_series().reset_index(drop=True)
        
        for col_name in df.columns:
            log.info("Processing column", dataset=dataset_name, column=col_name)
            
            series = df[col_name].reset_index(drop=True)
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(series):
                log.warning("Skipping non-numeric column", column=col_name)
                continue
            
            # Analyze the time series
            analyzer = TimeSeriesAnalyzer(series, timestamps)
            analysis = analyzer.get_full_analysis(include_stats=self.include_stats)
            
            # Generate description
            description = self.generator.generate_description(
                analysis=analysis,
                dataset_description=dataset_description,
                dataset_name=dataset_name,
                include_stats=self.include_stats
            )
            
            results['columns'][col_name] = {
                'analysis': analysis,
                'description': description
            }
           
                # Save individual results
            output_path = self.output_dir / f"{filepath.stem}_descriptions.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log.info("Saved descriptions", path=str(output_path))
    
        return results
    
    def process_all_datasets(self, file_pattern: str = "*.csv") -> dict:
        """
        Process all datasets in the raw_data directory.
        
        Args:
            file_pattern: Glob pattern for dataset files
            
        Returns:
            Dictionary with all results
        """
        all_results = {}
        
        for filepath in self.raw_data_dir.glob(file_pattern):
            try:
                results = self.process_dataset(filepath)
                if results:
                    all_results[filepath.stem] = results
                    
                    # Save individual results
                    output_path = self.output_dir / f"{filepath.stem}_descriptions.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    log.info("Saved descriptions", path=str(output_path))
                    
            except Exception as e:
                log.error("Failed to process dataset", filepath=str(filepath), error=str(e))
        
        # Save combined results
        combined_path = self.output_dir / "all_descriptions.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        log.info("Saved combined descriptions", path=str(combined_path))
        
        return all_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate semantic descriptions for time series datasets")
    parser.add_argument("--raw-data-dir", type=str, default="raw_data",
                        help="Directory containing raw CSV datasets")
    parser.add_argument("--description-dir", type=str, default="unstructured_description_data",
                        help="Directory containing .txt description files")
    parser.add_argument("--output-dir", type=str, default="generated_descriptions",
                        help="Directory to save generated descriptions")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gemini-pro-latest",
                        help="Gemini model to use")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Process only this specific dataset (filename without extension)")
    parser.add_argument("--include-stats", action="store_true", default=False,
                        help="Include basic statistics and sample data in the prompt (default: False)")
    
    args = parser.parse_args()
    
    # Get API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
    if not api_key:
        raise ValueError("API key required. Set --api-key or GEMINI_API_KEY environment variable")
    
    
    # Initialize processor
    processor = DatasetProcessor(
        raw_data_dir=args.raw_data_dir,
        description_dir=args.description_dir,
        output_dir=args.output_dir,
        api_key=api_key,
        model_name=args.model,
        include_stats=args.include_stats
    )
    
    # Process datasets
    if args.dataset:
        filepath = Path(args.raw_data_dir) / f"{args.dataset}.csv"
        if filepath.exists():
            processor.process_dataset(filepath)
        else:
            log.error("Dataset not found", path=str(filepath))
    else:
        processor.process_all_datasets()


if __name__ == "__main__":
    main()
