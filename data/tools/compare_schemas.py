#!/usr/bin/env python3
"""
Schema Comparison Tool: MongoDB vs Parquet

Analyzes and compares data structures between MongoDB collections and Parquet files.
Identifies missing fields, type mismatches, and coverage statistics.

Usage:
    python compare_schemas.py --mongo mongodb://localhost:27017/db/collection \\
                              --parquet path/to/data/*.parquet

    # Compare with sampling
    python compare_schemas.py --mongo mongodb://localhost:27017/db/collection \\
                              --parquet path/to/data/*.parquet \\
                              --sample 100
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
from urllib.parse import urlparse
import json

import pandas as pd
from pymongo import MongoClient
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import box

console = Console()


class SchemaAnalyzer:
    """Analyze schema structure from MongoDB or Parquet"""
    
    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size
    
    def analyze_mongodb(self, uri: str) -> Dict[str, Any]:
        """
        Analyze MongoDB collection schema.
        URI format: mongodb://host:port/database/collection
        """
        console.print(f"[cyan]→ Analyzing MongoDB: {uri}[/cyan]")
        
        # Parse URI
        parsed = urlparse(uri)
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) < 2:
            raise ValueError("MongoDB URI must include database and collection: mongodb://host:port/db/collection")
        
        db_name = path_parts[0]
        collection_name = path_parts[1]
        mongo_uri = f"{parsed.scheme}://{parsed.netloc}"
        
        # Connect
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        total_docs = collection.count_documents({})
        sample_size = min(self.sample_size, total_docs)
        
        console.print(f"  Total documents: {total_docs:,}")
        console.print(f"  Sampling: {sample_size:,} documents")
        
        # Sample documents
        docs = list(collection.aggregate([{'$sample': {'size': sample_size}}]))
        
        # Analyze schema
        schema = self._extract_schema_from_docs(docs)
        
        client.close()
        
        return {
            'source': 'mongodb',
            'uri': uri,
            'database': db_name,
            'collection': collection_name,
            'total_documents': total_docs,
            'sampled_documents': sample_size,
            'schema': schema
        }
    
    def analyze_parquet(self, pattern: str) -> Dict[str, Any]:
        """
        Analyze Parquet file(s) schema.
        Pattern can be a single file or glob pattern.
        """
        console.print(f"[cyan]→ Analyzing Parquet: {pattern}[/cyan]")
        
        # Find matching files
        path = Path(pattern)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = list(path.glob('*.parquet'))
        else:
            # Glob pattern
            parent = path.parent
            files = list(parent.glob(path.name))
        
        if not files:
            raise FileNotFoundError(f"No parquet files found matching: {pattern}")
        
        console.print(f"  Found {len(files)} parquet file(s)")
        
        # Analyze each file
        all_schemas = {}
        total_rows = 0
        
        for file in files:
            df = pd.read_parquet(file)
            file_name = file.stem  # e.g., "experiments", "powder_doses"
            
            console.print(f"    {file_name}: {len(df):,} rows, {len(df.columns)} columns")
            
            all_schemas[file_name] = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'row_count': len(df),
                'null_counts': df.isnull().sum().to_dict(),
                'sample_values': {col: df[col].dropna().head(3).tolist() for col in df.columns}
            }
            total_rows += len(df)
        
        return {
            'source': 'parquet',
            'pattern': pattern,
            'files': [str(f) for f in files],
            'total_rows': total_rows,
            'file_count': len(files),
            'schemas': all_schemas
        }
    
    def _extract_schema_from_docs(self, docs: List[Dict]) -> Dict[str, Any]:
        """Extract schema information from MongoDB documents"""
        
        # Track field occurrences and types
        field_info = defaultdict(lambda: {
            'count': 0,
            'types': Counter(),
            'null_count': 0,
            'sample_values': []
        })
        
        def process_dict(d: Dict, prefix: str = ''):
            """Recursively process nested dictionaries"""
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if value is None:
                    field_info[full_key]['null_count'] += 1
                    field_info[full_key]['types']['null'] += 1
                else:
                    field_info[full_key]['count'] += 1
                    
                    value_type = type(value).__name__
                    
                    # Handle nested structures
                    if isinstance(value, dict):
                        field_info[full_key]['types']['dict'] += 1
                        process_dict(value, full_key)
                    elif isinstance(value, list):
                        field_info[full_key]['types'][f'list[{len(value)}]'] += 1
                        if value and isinstance(value[0], dict):
                            # Sample first item for nested structure
                            process_dict(value[0], f"{full_key}[]")
                    else:
                        field_info[full_key]['types'][value_type] += 1
                        
                        # Store sample values
                        if len(field_info[full_key]['sample_values']) < 3:
                            if value_type in ['str', 'int', 'float', 'bool']:
                                field_info[full_key]['sample_values'].append(value)
        
        # Process all documents
        for doc in docs:
            process_dict(doc)
        
        # Calculate coverage percentages
        total_docs = len(docs)
        schema = {}
        
        for field, info in field_info.items():
            coverage = (info['count'] / total_docs) * 100
            most_common_type = info['types'].most_common(1)[0][0]
            
            schema[field] = {
                'coverage_percent': round(coverage, 1),
                'occurrences': info['count'],
                'null_count': info['null_count'],
                'primary_type': most_common_type,
                'all_types': dict(info['types']),
                'sample_values': info['sample_values'][:3]
            }
        
        return schema


class SchemaComparator:
    """Compare schemas between MongoDB and Parquet"""
    
    def __init__(self, mongo_analysis: Dict, parquet_analysis: Dict):
        self.mongo = mongo_analysis
        self.parquet = parquet_analysis
    
    def compare(self) -> Dict[str, Any]:
        """Perform comprehensive schema comparison"""
        
        # Flatten MongoDB schema paths
        mongo_fields = set(self.mongo['schema'].keys())
        
        # Create mapping from MongoDB fields to Parquet tables/columns
        comparison = {
            'mongo_only': [],
            'parquet_only': [],
            'matched': [],
            'type_mismatches': [],
            'coverage_issues': []
        }
        
        # Get all parquet columns (flattened with table prefix)
        parquet_fields = {}
        for table_name, table_schema in self.parquet['schemas'].items():
            for column in table_schema['columns']:
                full_name = f"{table_name}.{column}"
                parquet_fields[full_name] = {
                    'table': table_name,
                    'column': column,
                    'dtype': table_schema['dtypes'][column],
                    'null_count': table_schema['null_counts'][column],
                    'row_count': table_schema['row_count']
                }
        
        parquet_field_set = set(parquet_fields.keys())
        
        # Compare fields
        for mongo_field in sorted(mongo_fields):
            mongo_info = self.mongo['schema'][mongo_field]
            
            # Try to find matching parquet field
            matched = False
            
            for parquet_field, parquet_info in parquet_fields.items():
                # Check if MongoDB field maps to this parquet field
                if self._fields_match(mongo_field, parquet_field):
                    matched = True
                    
                    comparison['matched'].append({
                        'mongo_field': mongo_field,
                        'parquet_field': parquet_field,
                        'mongo_type': mongo_info['primary_type'],
                        'parquet_type': parquet_info['dtype'],
                        'mongo_coverage': mongo_info['coverage_percent'],
                        'parquet_nulls': parquet_info['null_count']
                    })
                    
                    # Check for type mismatches
                    if not self._types_compatible(mongo_info['primary_type'], parquet_info['dtype']):
                        comparison['type_mismatches'].append({
                            'field': mongo_field,
                            'mongo_type': mongo_info['primary_type'],
                            'parquet_type': parquet_info['dtype']
                        })
                    
                    # Check for coverage issues
                    if mongo_info['coverage_percent'] < 90:
                        comparison['coverage_issues'].append({
                            'field': mongo_field,
                            'coverage_percent': mongo_info['coverage_percent'],
                            'null_count': mongo_info['null_count']
                        })
                    
                    break
            
            if not matched:
                comparison['mongo_only'].append({
                    'field': mongo_field,
                    'type': mongo_info['primary_type'],
                    'coverage': mongo_info['coverage_percent'],
                    'sample_values': mongo_info['sample_values']
                })
        
        # Find parquet-only fields
        for parquet_field in sorted(parquet_field_set):
            matched = False
            for mongo_field in mongo_fields:
                if self._fields_match(mongo_field, parquet_field):
                    matched = True
                    break
            
            if not matched:
                parquet_info = parquet_fields[parquet_field]
                comparison['parquet_only'].append({
                    'field': parquet_field,
                    'type': parquet_info['dtype'],
                    'table': parquet_info['table']
                })
        
        return comparison
    
    def _fields_match(self, mongo_field: str, parquet_field: str) -> bool:
        """Check if MongoDB field matches Parquet field"""
        # Simple heuristic: check if MongoDB path ends with parquet column name
        # or if they share significant naming similarity
        
        parquet_parts = parquet_field.split('.')
        if len(parquet_parts) == 2:
            table, column = parquet_parts
            
            # Check if mongo field ends with column name
            if mongo_field.endswith(column):
                return True
            
            # Check if last part of mongo field matches column
            mongo_parts = mongo_field.split('.')
            if mongo_parts[-1] == column:
                return True
            
            # Check for common transformations
            # e.g., metadata.powderdosing_results.ActualTransferMass -> dosing_sessions.actual_transfer_mass
            mongo_snake = mongo_parts[-1].lower().replace('_', '')
            parquet_snake = column.lower().replace('_', '')
            if mongo_snake == parquet_snake:
                return True
        
        return False
    
    def _types_compatible(self, mongo_type: str, parquet_type: str) -> bool:
        """Check if MongoDB and Parquet types are compatible"""
        
        type_mapping = {
            'str': ['object', 'string', 'category'],
            'int': ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8'],
            'float': ['float64', 'float32'],
            'bool': ['bool'],
            'dict': ['object'],
            'list': ['object'],
            'ObjectId': ['object', 'string']
        }
        
        return parquet_type in type_mapping.get(mongo_type, [])


def print_comparison_report(comparison: Dict[str, Any], mongo_info: Dict, parquet_info: Dict):
    """Print a beautiful comparison report"""
    
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Schema Comparison Report[/bold cyan]\n"
        f"MongoDB: {mongo_info['collection']} ({mongo_info['sampled_documents']:,} docs)\n"
        f"Parquet: {parquet_info['file_count']} file(s) ({parquet_info['total_rows']:,} rows)",
        box=box.DOUBLE
    ))
    console.print()
    
    # Summary statistics
    summary_table = Table(title="Summary", box=box.SIMPLE)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", justify="right", style="yellow")
    
    summary_table.add_row("✓ Matched Fields", str(len(comparison['matched'])))
    summary_table.add_row("⚠ Type Mismatches", str(len(comparison['type_mismatches'])))
    summary_table.add_row("⚠ Coverage Issues (<90%)", str(len(comparison['coverage_issues'])))
    summary_table.add_row("➕ MongoDB Only", str(len(comparison['mongo_only'])))
    summary_table.add_row("➕ Parquet Only", str(len(comparison['parquet_only'])))
    
    console.print(summary_table)
    console.print()
    
    # MongoDB-only fields (missing in Parquet)
    if comparison['mongo_only']:
        console.print("[bold red]➕ Fields in MongoDB but NOT in Parquet:[/bold red]")
        mongo_only_table = Table(box=box.SIMPLE_HEAD)
        mongo_only_table.add_column("MongoDB Field", style="yellow")
        mongo_only_table.add_column("Type", style="cyan")
        mongo_only_table.add_column("Coverage %", justify="right")
        mongo_only_table.add_column("Sample Values", style="dim")
        
        for item in comparison['mongo_only'][:20]:  # Limit to 20
            coverage_style = "green" if item['coverage'] > 80 else "yellow" if item['coverage'] > 50 else "red"
            sample_str = str(item['sample_values'][:2])[:50]
            mongo_only_table.add_row(
                item['field'],
                item['type'],
                f"[{coverage_style}]{item['coverage']:.1f}%[/{coverage_style}]",
                sample_str
            )
        
        if len(comparison['mongo_only']) > 20:
            mongo_only_table.add_row("...", "...", "...", f"({len(comparison['mongo_only']) - 20} more)")
        
        console.print(mongo_only_table)
        console.print()
    
    # Parquet-only fields (extra in Parquet)
    if comparison['parquet_only']:
        console.print("[bold blue]➕ Fields in Parquet but NOT in MongoDB:[/bold blue]")
        parquet_only_table = Table(box=box.SIMPLE_HEAD)
        parquet_only_table.add_column("Parquet Field", style="yellow")
        parquet_only_table.add_column("Table", style="cyan")
        parquet_only_table.add_column("Type")
        
        for item in comparison['parquet_only'][:20]:
            parquet_only_table.add_row(
                item['field'],
                item['table'],
                item['type']
            )
        
        if len(comparison['parquet_only']) > 20:
            parquet_only_table.add_row("...", "...", f"({len(comparison['parquet_only']) - 20} more)")
        
        console.print(parquet_only_table)
        console.print()
    
    # Type mismatches
    if comparison['type_mismatches']:
        console.print("[bold yellow]⚠ Type Mismatches:[/bold yellow]")
        type_table = Table(box=box.SIMPLE_HEAD)
        type_table.add_column("Field", style="yellow")
        type_table.add_column("MongoDB Type", style="cyan")
        type_table.add_column("Parquet Type", style="magenta")
        
        for item in comparison['type_mismatches'][:20]:
            type_table.add_row(
                item['field'],
                item['mongo_type'],
                item['parquet_type']
            )
        
        console.print(type_table)
        console.print()
    
    # Coverage issues
    if comparison['coverage_issues']:
        console.print("[bold yellow]⚠ Fields with Low Coverage (<90%):[/bold yellow]")
        coverage_table = Table(box=box.SIMPLE_HEAD)
        coverage_table.add_column("Field", style="yellow")
        coverage_table.add_column("Coverage %", justify="right")
        coverage_table.add_column("Null Count", justify="right")
        
        for item in sorted(comparison['coverage_issues'], key=lambda x: x['coverage_percent'])[:20]:
            coverage_style = "yellow" if item['coverage_percent'] > 50 else "red"
            coverage_table.add_row(
                item['field'],
                f"[{coverage_style}]{item['coverage_percent']:.1f}%[/{coverage_style}]",
                str(item['null_count'])
            )
        
        console.print(coverage_table)
        console.print()


def main():
    parser = argparse.ArgumentParser(
        description='Compare schemas between MongoDB and Parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare MongoDB collection to parquet files
  python compare_schemas.py \\
      --mongo mongodb://localhost:27017/temporary/release \\
      --parquet data/parquet/
  
  # Sample only 100 documents
  python compare_schemas.py \\
      --mongo mongodb://localhost:27017/temporary/release \\
      --parquet data/parquet/ \\
      --sample 100
  
  # Export results to JSON
  python compare_schemas.py \\
      --mongo mongodb://localhost:27017/temporary/release \\
      --parquet data/parquet/ \\
      --output comparison_report.json
        """
    )
    
    parser.add_argument(
        '--mongo', '-m',
        required=True,
        help='MongoDB URI: mongodb://host:port/database/collection'
    )
    parser.add_argument(
        '--parquet', '-p',
        required=True,
        help='Parquet file pattern (file, directory, or glob)'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=1000,
        help='Number of MongoDB documents to sample (default: 1000)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Export comparison results to JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        # Analyze both sources
        analyzer = SchemaAnalyzer(sample_size=args.sample)
        
        mongo_analysis = analyzer.analyze_mongodb(args.mongo)
        parquet_analysis = analyzer.analyze_parquet(args.parquet)
        
        # Compare schemas
        comparator = SchemaComparator(mongo_analysis, parquet_analysis)
        comparison = comparator.compare()
        
        # Print report
        print_comparison_report(comparison, mongo_analysis, parquet_analysis)
        
        # Export if requested
        if args.output:
            output_data = {
                'mongo_analysis': mongo_analysis,
                'parquet_analysis': parquet_analysis,
                'comparison': comparison
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            console.print(f"[green]✓ Exported results to: {args.output}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)


if __name__ == '__main__':
    main()

