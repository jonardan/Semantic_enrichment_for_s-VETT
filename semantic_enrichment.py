#!/usr/bin/env python3
"""
Semantic Enrichment Script for s-VETT

This script reads CSV datasets and creates semantic descriptions for each column
by analyzing the structured data in the CSV and incorporating context from
corresponding unstructured text files.
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any


class SemanticEnrichment:
    """
    A class to generate semantic descriptions for CSV dataset columns.
    """
    
    # Maximum length of context to include from unstructured data
    MAX_CONTEXT_LENGTH = 500
    
    def __init__(self, datasets_folder: str = "datasets", 
                 unstructured_folder: str = "unstructured_data",
                 output_folder: str = "generated_descriptions"):
        """
        Initialize the SemanticEnrichment processor.
        
        Args:
            datasets_folder: Path to folder containing CSV files
            unstructured_folder: Path to folder containing text files
            output_folder: Path to folder for saving generated descriptions
        """
        self.datasets_folder = Path(datasets_folder)
        self.unstructured_folder = Path(unstructured_folder)
        self.output_folder = Path(output_folder)
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def analyze_column(self, column_data: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Analyze a single column and extract its characteristics.
        
        Args:
            column_data: Pandas Series containing the column data
            column_name: Name of the column
            
        Returns:
            Dictionary containing column analysis
        """
        analysis = {
            "column_name": column_name,
            "data_type": str(column_data.dtype),
            "non_null_count": int(column_data.count()),
            "null_count": int(column_data.isna().sum()),
            "total_count": len(column_data),
            "unique_count": int(column_data.nunique())
        }
        
        # Add type-specific analysis
        if pd.api.types.is_numeric_dtype(column_data):
            analysis.update({
                "min_value": float(column_data.min()) if not column_data.empty else None,
                "max_value": float(column_data.max()) if not column_data.empty else None,
                "mean_value": float(column_data.mean()) if not column_data.empty else None,
                "median_value": float(column_data.median()) if not column_data.empty else None
            })
        elif pd.api.types.is_string_dtype(column_data) or column_data.dtype == 'object':
            # Get sample values (up to 5 unique values)
            sample_values = column_data.dropna().unique()[:5].tolist()
            analysis["sample_values"] = [str(v) for v in sample_values]
            
            # Average string length if applicable
            if column_data.dropna().apply(lambda x: isinstance(x, str)).any():
                avg_length = column_data.dropna().apply(lambda x: len(str(x))).mean()
                analysis["avg_string_length"] = float(avg_length) if not pd.isna(avg_length) else None
        
        return analysis
    
    def create_semantic_description(self, column_analysis: Dict[str, Any], 
                                   unstructured_context: str) -> Dict[str, Any]:
        """
        Create a semantic description combining structured analysis and unstructured context.
        
        Args:
            column_analysis: Dictionary containing column analysis
            unstructured_context: Text from corresponding unstructured data file
            
        Returns:
            Dictionary containing the semantic description
        """
        description = {
            "column_name": column_analysis["column_name"],
            "structured_analysis": column_analysis,
            "semantic_description": self._generate_description_text(column_analysis),
            "context_from_unstructured_data": unstructured_context[:self.MAX_CONTEXT_LENGTH] if unstructured_context else "No context available",
            "metadata": {
                "completeness_ratio": column_analysis["non_null_count"] / column_analysis["total_count"] if column_analysis["total_count"] > 0 else 0,
                "uniqueness_ratio": column_analysis["unique_count"] / column_analysis["total_count"] if column_analysis["total_count"] > 0 else 0
            }
        }
        
        return description
    
    def _generate_description_text(self, analysis: Dict[str, Any]) -> str:
        """
        Generate a human-readable description text from analysis.
        
        Args:
            analysis: Dictionary containing column analysis
            
        Returns:
            Human-readable description string
        """
        desc_parts = []
        
        # Basic description
        desc_parts.append(f"Column '{analysis['column_name']}' is of type {analysis['data_type']}.")
        
        # Completeness info
        if analysis['null_count'] > 0:
            null_pct = (analysis['null_count'] / analysis['total_count']) * 100
            desc_parts.append(f"It contains {analysis['non_null_count']} non-null values out of {analysis['total_count']} total entries ({null_pct:.1f}% missing).")
        else:
            desc_parts.append(f"It contains {analysis['non_null_count']} complete values with no missing data.")
        
        # Uniqueness info
        if analysis['unique_count'] == analysis['non_null_count']:
            desc_parts.append("All values are unique (potentially an identifier).")
        elif analysis['unique_count'] < 10:
            desc_parts.append(f"It has {analysis['unique_count']} unique values (potentially categorical).")
        else:
            desc_parts.append(f"It has {analysis['unique_count']} unique values.")
        
        # Type-specific description
        if 'min_value' in analysis:
            mean_val = analysis['mean_value']
            if pd.notna(mean_val):
                desc_parts.append(f"Numeric range: {analysis['min_value']} to {analysis['max_value']} (mean: {mean_val:.2f}).")
            else:
                desc_parts.append(f"Numeric range: {analysis['min_value']} to {analysis['max_value']}.")
        elif 'sample_values' in analysis:
            sample_str = ', '.join([f"'{v}'" for v in analysis['sample_values']])
            desc_parts.append(f"Sample values: {sample_str}.")
        
        return " ".join(desc_parts)
    
    def read_unstructured_data(self, csv_filename: str) -> str:
        """
        Read the corresponding unstructured text file for a CSV file.
        
        Args:
            csv_filename: Name of the CSV file (e.g., 'data.csv')
            
        Returns:
            Content of the text file or empty string if not found
        """
        # Replace .csv extension with .txt
        txt_filename = Path(csv_filename).stem + ".txt"
        txt_path = self.unstructured_folder / txt_filename
        
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Unstructured data file not found: {txt_path}")
            return ""
    
    def process_csv(self, csv_path: Path):
        """
        Process a single CSV file and generate descriptions for all columns.
        
        Args:
            csv_path: Path to the CSV file
        """
        print(f"\nProcessing: {csv_path.name}")
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            return
        
        # Read corresponding unstructured data
        unstructured_context = self.read_unstructured_data(csv_path.name)
        
        # Process each column
        dataset_name = csv_path.stem
        
        for column_name in df.columns:
            # Analyze column
            column_analysis = self.analyze_column(df[column_name], column_name)
            
            # Create semantic description
            semantic_desc = self.create_semantic_description(column_analysis, unstructured_context)
            
            # Save description to file
            output_filename = f"{dataset_name}_{column_name}_description.json"
            output_path = self.output_folder / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(semantic_desc, f, indent=2, ensure_ascii=False)
            
            print(f"  - Created description for column: {column_name}")
    
    def process_all_datasets(self):
        """
        Process all CSV files in the datasets folder.
        """
        if not self.datasets_folder.exists():
            print(f"Error: Datasets folder not found: {self.datasets_folder}")
            return
        
        csv_files = list(self.datasets_folder.glob("*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.datasets_folder}")
            return
        
        print(f"Found {len(csv_files)} CSV file(s) to process")
        
        for csv_file in csv_files:
            self.process_csv(csv_file)
        
        print(f"\n✓ Processing complete! Descriptions saved to {self.output_folder}")


def main():
    """
    Main entry point for the script.
    """
    print("=" * 60)
    print("Semantic Enrichment for s-VETT")
    print("=" * 60)
    
    # Initialize and run the processor
    processor = SemanticEnrichment()
    processor.process_all_datasets()


if __name__ == "__main__":
    main()
