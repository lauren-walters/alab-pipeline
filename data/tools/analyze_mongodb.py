#!/usr/bin/env python3
"""
MongoDB Database Analyzer

Provides comprehensive analysis of any MongoDB database/collection:
- Structure discovery and field statistics
- Data distribution and patterns
- Grouping detection (categories, series, etc.)
- Temporal patterns
- Data quality metrics

Usage:
    # Analyze entire database
    python analyze_mongodb.py mongodb://localhost:27017/database_name
    
    # Analyze specific collection
    python analyze_mongodb.py mongodb://localhost:27017/database_name/collection_name
    
    # Quick analysis (sample fewer docs)
    python analyze_mongodb.py mongodb://localhost:27017/db/collection --sample 100
    
    # Export results
    python analyze_mongodb.py mongodb://localhost:27017/db/collection --output analysis.json
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from urllib.parse import urlparse
from datetime import datetime
import json
import re

from pymongo import MongoClient
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.columns import Columns
from rich import box
from rich.progress import track

console = Console()


class MongoDBAnalyzer:
    """Comprehensive MongoDB database analyzer"""
    
    def __init__(self, uri: str, sample_size: int = 1000):
        self.uri = uri
        self.sample_size = sample_size
        
        # Parse URI
        parsed = urlparse(uri)
        path_parts = [p for p in parsed.path.strip('/').split('/') if p]
        
        self.mongo_uri = f"{parsed.scheme}://{parsed.netloc}"
        self.database_name = path_parts[0] if path_parts else None
        self.collection_name = path_parts[1] if len(path_parts) > 1 else None
        
        # Connect
        self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
        
    def analyze_all(self) -> Dict[str, Any]:
        """Run comprehensive analysis"""
        
        if self.collection_name:
            # Analyze specific collection
            return self._analyze_collection(self.database_name, self.collection_name)
        elif self.database_name:
            # Analyze all collections in database
            return self._analyze_database(self.database_name)
        else:
            # List all databases
            return self._list_databases()
    
    def _list_databases(self) -> Dict[str, Any]:
        """List all available databases"""
        console.print("[cyan]→ Listing databases...[/cyan]")
        
        dbs = self.client.list_database_names()
        
        db_info = []
        for db_name in dbs:
            if db_name in ['admin', 'config', 'local']:
                continue  # Skip system databases
            
            db = self.client[db_name]
            collections = db.list_collection_names()
            
            # Get rough size estimate
            stats = db.command('dbStats')
            
            db_info.append({
                'name': db_name,
                'collections': len(collections),
                'collection_names': collections,
                'size_mb': stats.get('dataSize', 0) / (1024 * 1024)
            })
        
        return {
            'type': 'database_list',
            'databases': db_info
        }
    
    def _analyze_database(self, db_name: str) -> Dict[str, Any]:
        """Analyze all collections in a database"""
        console.print(f"[cyan]→ Analyzing database: {db_name}[/cyan]")
        
        db = self.client[db_name]
        collections = db.list_collection_names()
        
        collection_analyses = {}
        
        for coll_name in collections:
            console.print(f"  [dim]Analyzing collection: {coll_name}[/dim]")
            try:
                analysis = self._analyze_collection(db_name, coll_name)
                collection_analyses[coll_name] = analysis
            except Exception as e:
                console.print(f"    [red]Error: {e}[/red]")
        
        return {
            'type': 'database_analysis',
            'database': db_name,
            'collections': collection_analyses
        }
    
    def _analyze_collection(self, db_name: str, coll_name: str) -> Dict[str, Any]:
        """Deep analysis of a single collection"""
        console.print(f"[cyan]→ Analyzing collection: {db_name}.{coll_name}[/cyan]")
        
        db = self.client[db_name]
        collection = db[coll_name]
        
        # Basic stats
        total_docs = collection.count_documents({})
        sample_size = min(self.sample_size, total_docs)
        
        console.print(f"  Total documents: {total_docs:,}")
        console.print(f"  Sampling: {sample_size:,} documents")
        
        # Sample documents
        docs = list(collection.aggregate([{'$sample': {'size': sample_size}}]))
        
        if not docs:
            return {'error': 'No documents found'}
        
        # Run all analyses
        analysis = {
            'database': db_name,
            'collection': coll_name,
            'total_documents': total_docs,
            'sampled_documents': sample_size,
            'schema': self._analyze_schema(docs),
            'patterns': self._detect_patterns(docs),
            'groupings': self._detect_groupings(docs),
            'temporal': self._analyze_temporal(docs),
            'data_quality': self._analyze_quality(docs),
            'sample_document': self._get_representative_sample(docs[0])
        }
        
        return analysis
    
    def _analyze_schema(self, docs: List[Dict]) -> Dict[str, Any]:
        """Analyze document schema and field statistics"""
        
        field_stats = defaultdict(lambda: {
            'count': 0,
            'null_count': 0,
            'types': Counter(),
            'sample_values': [],
            'unique_values': set(),
            'numeric_stats': []
        })
        
        def process_field(key: str, value: Any, prefix: str = ''):
            full_key = f"{prefix}.{key}" if prefix else key
            
            if value is None:
                field_stats[full_key]['null_count'] += 1
                field_stats[full_key]['types']['null'] += 1
            else:
                field_stats[full_key]['count'] += 1
                value_type = type(value).__name__
                field_stats[full_key]['types'][value_type] += 1
                
                # Store samples
                if len(field_stats[full_key]['sample_values']) < 5:
                    if isinstance(value, (str, int, float, bool)):
                        field_stats[full_key]['sample_values'].append(value)
                
                # Track unique values for low-cardinality fields
                if isinstance(value, (str, int, bool)) and len(field_stats[full_key]['unique_values']) < 50:
                    field_stats[full_key]['unique_values'].add(value)
                
                # Numeric stats
                if isinstance(value, (int, float)):
                    field_stats[full_key]['numeric_stats'].append(value)
                
                # Recurse into nested structures
                if isinstance(value, dict):
                    for k, v in value.items():
                        process_field(k, v, full_key)
                elif isinstance(value, list) and value:
                    # Sample first item if it's a dict
                    if isinstance(value[0], dict):
                        for k, v in value[0].items():
                            process_field(k, v, f"{full_key}[]")
                    field_stats[full_key]['types'][f'array[{len(value)}]'] += 1
        
        # Process all documents
        for doc in track(docs, description="Analyzing schema...", console=console):
            for key, value in doc.items():
                process_field(key, value)
        
        # Calculate statistics
        total_docs = len(docs)
        schema = {}
        
        for field, stats in field_stats.items():
            coverage = (stats['count'] / total_docs) * 100
            most_common_type = stats['types'].most_common(1)[0][0]
            
            field_info = {
                'coverage_percent': round(coverage, 1),
                'occurrences': stats['count'],
                'null_count': stats['null_count'],
                'primary_type': most_common_type,
                'all_types': dict(stats['types']),
                'cardinality': len(stats['unique_values']) if len(stats['unique_values']) < 50 else '50+',
                'sample_values': stats['sample_values'][:5]
            }
            
            # Add numeric statistics
            if stats['numeric_stats']:
                nums = stats['numeric_stats']
                field_info['numeric_stats'] = {
                    'min': min(nums),
                    'max': max(nums),
                    'avg': sum(nums) / len(nums)
                }
            
            # Add unique values if low cardinality
            if len(stats['unique_values']) < 20:
                field_info['unique_values'] = sorted([str(v) for v in stats['unique_values']])
            
            schema[field] = field_info
        
        return schema
    
    def _detect_patterns(self, docs: List[Dict]) -> Dict[str, Any]:
        """Detect common patterns in the data"""
        
        patterns = {
            'naming_patterns': self._detect_naming_patterns(docs),
            'id_patterns': self._detect_id_patterns(docs),
            'status_fields': self._find_status_fields(docs),
            'array_patterns': self._detect_array_patterns(docs)
        }
        
        return patterns
    
    def _detect_naming_patterns(self, docs: List[Dict]) -> Dict[str, Any]:
        """Detect naming conventions and patterns"""
        
        # Look for name-like fields
        name_fields = []
        for key in docs[0].keys():
            if any(term in key.lower() for term in ['name', 'title', 'label', 'id']):
                name_fields.append(key)
        
        # Analyze naming patterns
        patterns = {}
        for field in name_fields:
            values = [doc.get(field) for doc in docs if doc.get(field)]
            if not values:
                continue
            
            # Check for common prefixes/patterns
            if all(isinstance(v, str) for v in values[:10]):
                prefixes = Counter()
                for v in values:
                    # Extract prefix pattern (e.g., NSC_, PG_, etc.)
                    match = re.match(r'^([A-Z]+[_-]?\d*)', str(v))
                    if match:
                        prefixes[match.group(1)] += 1
                
                if prefixes:
                    patterns[field] = {
                        'sample_values': values[:5],
                        'common_prefixes': prefixes.most_common(5),
                        'total_unique': len(set(values))
                    }
        
        return patterns
    
    def _detect_id_patterns(self, docs: List[Dict]) -> List[str]:
        """Find fields that appear to be IDs or unique identifiers"""
        
        id_fields = []
        for key in docs[0].keys():
            if 'id' in key.lower() or key == '_id':
                values = [str(doc.get(key)) for doc in docs if doc.get(key)]
                uniqueness = len(set(values)) / len(values) if values else 0
                
                if uniqueness > 0.95:  # Mostly unique
                    id_fields.append({
                        'field': key,
                        'uniqueness': round(uniqueness * 100, 1),
                        'sample': values[0] if values else None
                    })
        
        return id_fields
    
    def _find_status_fields(self, docs: List[Dict]) -> List[Dict]:
        """Find fields that represent status, state, or categories"""
        
        status_fields = []
        
        def check_field(key: str, prefix: str = ''):
            full_key = f"{prefix}.{key}" if prefix else key
            
            # Check if field name suggests status
            if any(term in key.lower() for term in ['status', 'state', 'type', 'category', 'classification']):
                values = []
                for doc in docs:
                    val = doc
                    for part in full_key.split('.'):
                        val = val.get(part) if isinstance(val, dict) else None
                        if val is None:
                            break
                    if val is not None:
                        values.append(val)
                
                if values:
                    unique_vals = Counter(values)
                    if len(unique_vals) < 20:  # Low cardinality
                        status_fields.append({
                            'field': full_key,
                            'unique_count': len(unique_vals),
                            'distribution': dict(unique_vals.most_common(10))
                        })
        
        # Check top-level fields
        for key in docs[0].keys():
            check_field(key)
            # Check one level deep
            if isinstance(docs[0].get(key), dict):
                for nested_key in docs[0][key].keys():
                    check_field(nested_key, key)
        
        return status_fields
    
    def _detect_array_patterns(self, docs: List[Dict]) -> List[Dict]:
        """Analyze array fields for patterns"""
        
        array_fields = []
        
        def check_arrays(key: str, value: Any, prefix: str = ''):
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, list) and value:
                lengths = []
                element_types = Counter()
                
                for doc in docs:
                    val = doc
                    for part in full_key.split('.'):
                        val = val.get(part) if isinstance(val, dict) else None
                        if val is None:
                            break
                    
                    if isinstance(val, list):
                        lengths.append(len(val))
                        if val:
                            element_types[type(val[0]).__name__] += 1
                
                if lengths:
                    array_fields.append({
                        'field': full_key,
                        'min_length': min(lengths),
                        'max_length': max(lengths),
                        'avg_length': round(sum(lengths) / len(lengths), 1),
                        'element_type': element_types.most_common(1)[0][0]
                    })
            
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_arrays(k, v, full_key)
        
        for key, value in docs[0].items():
            check_arrays(key, value)
        
        return array_fields
    
    def _detect_groupings(self, docs: List[Dict]) -> Dict[str, Any]:
        """Detect natural groupings in the data (like experiment series)"""
        
        groupings = {}
        
        # Look for fields that could define groups
        for key in docs[0].keys():
            if key in ['_id']:
                continue
            
            values = [doc.get(key) for doc in docs if doc.get(key)]
            if not values:
                continue
            
            # String-based grouping (look for patterns)
            if all(isinstance(v, str) for v in values[:10]):
                # Extract prefixes
                prefix_groups = defaultdict(list)
                for v in values:
                    # Try to extract prefix (letters before numbers/separators)
                    match = re.match(r'^([A-Za-z_]+)', str(v))
                    if match:
                        prefix = match.group(1)
                        prefix_groups[prefix].append(v)
                
                # Only keep meaningful groupings
                if len(prefix_groups) > 1 and len(prefix_groups) < len(values) / 2:
                    groupings[key] = {
                        'type': 'prefix_grouping',
                        'groups': {k: len(v) for k, v in prefix_groups.items() if len(v) > 1},
                        'sample_groups': {k: v[:3] for k, v in list(prefix_groups.items())[:5]}
                    }
            
            # Categorical grouping
            elif isinstance(values[0], (str, int, bool)):
                unique_vals = Counter(values)
                if 2 <= len(unique_vals) <= 20:  # Between 2 and 20 groups
                    groupings[key] = {
                        'type': 'categorical',
                        'groups': dict(unique_vals),
                        'group_count': len(unique_vals)
                    }
        
        return groupings
    
    def _analyze_temporal(self, docs: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in the data"""
        
        temporal = {
            'date_fields': [],
            'time_range': None,
            'temporal_distribution': None
        }
        
        # Find date fields
        for key in docs[0].keys():
            value = docs[0].get(key)
            if isinstance(value, datetime) or 'date' in key.lower() or 'time' in key.lower() or 'updated' in key.lower():
                dates = []
                for doc in docs:
                    val = doc.get(key)
                    if isinstance(val, datetime):
                        dates.append(val)
                    elif isinstance(val, str):
                        # Try to parse
                        try:
                            from dateutil import parser
                            dates.append(parser.parse(val))
                        except:
                            pass
                
                if dates:
                    dates.sort()
                    temporal['date_fields'].append({
                        'field': key,
                        'earliest': dates[0],
                        'latest': dates[-1],
                        'span_days': (dates[-1] - dates[0]).days if len(dates) > 1 else 0,
                        'count': len(dates)
                    })
        
        return temporal
    
    def _analyze_quality(self, docs: List[Dict]) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        
        total_docs = len(docs)
        
        # Find fields with high null rates
        high_null_fields = []
        all_fields = set()
        
        def collect_fields(d: Dict, prefix: str = ''):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                all_fields.add(full_key)
                if isinstance(value, dict):
                    collect_fields(value, full_key)
        
        collect_fields(docs[0])
        
        for field in all_fields:
            null_count = 0
            for doc in docs:
                val = doc
                for part in field.split('.'):
                    val = val.get(part) if isinstance(val, dict) else None
                    if val is None:
                        break
                if val is None:
                    null_count += 1
            
            null_rate = (null_count / total_docs) * 100
            if null_rate > 20:  # More than 20% null
                high_null_fields.append({
                    'field': field,
                    'null_percent': round(null_rate, 1)
                })
        
        return {
            'total_fields': len(all_fields),
            'high_null_fields': sorted(high_null_fields, key=lambda x: x['null_percent'], reverse=True)[:10]
        }
    
    def _get_representative_sample(self, doc: Dict, max_depth: int = 3) -> Dict:
        """Get a cleaned sample document for display"""
        
        def clean_doc(d: Any, depth: int = 0) -> Any:
            if depth >= max_depth:
                return "..."
            
            if isinstance(d, dict):
                return {k: clean_doc(v, depth + 1) for k, v in list(d.items())[:10]}
            elif isinstance(d, list):
                if len(d) > 3:
                    return [clean_doc(d[0], depth + 1), "...", f"({len(d)} items)"]
                return [clean_doc(item, depth + 1) for item in d]
            elif isinstance(d, datetime):
                return d.isoformat()
            else:
                return d
        
        return clean_doc(doc)
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


def print_analysis_report(analysis: Dict[str, Any]):
    """Print beautiful analysis report"""
    
    console.print()
    
    if analysis.get('type') == 'database_list':
        # Database listing
        console.print(Panel.fit(
            "[bold cyan]Available Databases[/bold cyan]",
            box=box.DOUBLE
        ))
        console.print()
        
        table = Table(title="Databases", box=box.SIMPLE_HEAD)
        table.add_column("Database", style="cyan")
        table.add_column("Collections", justify="right", style="yellow")
        table.add_column("Size (MB)", justify="right")
        
        for db in analysis['databases']:
            table.add_row(
                db['name'],
                str(db['collections']),
                f"{db['size_mb']:.2f}"
            )
        
        console.print(table)
        console.print()
        
        # Show collection details
        for db in analysis['databases']:
            if db['collection_names']:
                console.print(f"[cyan]{db['name']}[/cyan] collections: {', '.join(db['collection_names'][:10])}")
                if len(db['collection_names']) > 10:
                    console.print(f"  ... and {len(db['collection_names']) - 10} more")
        
        return
    
    # Collection analysis
    console.print(Panel.fit(
        f"[bold cyan]MongoDB Collection Analysis[/bold cyan]\n"
        f"Database: {analysis['database']}\n"
        f"Collection: {analysis['collection']}\n"
        f"Documents: {analysis['total_documents']:,} (sampled {analysis['sampled_documents']:,})",
        box=box.DOUBLE
    ))
    console.print()
    
    # Groupings (important for discovery)
    if analysis['groupings']:
        console.print("[bold green]🔍 Detected Groupings:[/bold green]")
        for field, grouping in list(analysis['groupings'].items())[:5]:
            if grouping['type'] == 'prefix_grouping':
                console.print(f"\n  [yellow]{field}[/yellow] - Prefix-based groups:")
                for group_name, count in list(grouping['groups'].items())[:10]:
                    console.print(f"    • {group_name}: {count} documents")
                    samples = grouping['sample_groups'].get(group_name, [])[:3]
                    if samples:
                        console.print(f"      Examples: {', '.join(samples)}")
            elif grouping['type'] == 'categorical':
                console.print(f"\n  [yellow]{field}[/yellow] - {grouping['group_count']} categories:")
                for cat, count in list(grouping['groups'].items())[:10]:
                    console.print(f"    • {cat}: {count} documents")
        console.print()
    
    # Status fields
    if analysis['patterns']['status_fields']:
        console.print("[bold blue]📊 Status/Category Fields:[/bold blue]")
        for status_field in analysis['patterns']['status_fields'][:5]:
            console.print(f"\n  [yellow]{status_field['field']}[/yellow]:")
            for value, count in list(status_field['distribution'].items())[:8]:
                pct = (count / analysis['sampled_documents']) * 100
                console.print(f"    • {value}: {count} ({pct:.1f}%)")
        console.print()
    
    # Temporal patterns
    if analysis['temporal']['date_fields']:
        console.print("[bold magenta]📅 Temporal Information:[/bold magenta]")
        for date_field in analysis['temporal']['date_fields']:
            console.print(f"\n  [yellow]{date_field['field']}[/yellow]:")
            console.print(f"    Range: {date_field['earliest']} → {date_field['latest']}")
            console.print(f"    Span: {date_field['span_days']} days")
        console.print()
    
    # Schema summary (top fields)
    console.print("[bold cyan]📋 Schema Summary (Top Fields):[/bold cyan]")
    schema_table = Table(box=box.SIMPLE_HEAD)
    schema_table.add_column("Field", style="yellow")
    schema_table.add_column("Type", style="cyan")
    schema_table.add_column("Coverage", justify="right")
    schema_table.add_column("Cardinality")
    schema_table.add_column("Sample Values", style="dim")
    
    # Show top-level and important fields
    important_fields = sorted(
        [(k, v) for k, v in analysis['schema'].items() if '.' not in k or 'metadata' in k],
        key=lambda x: x[1]['coverage_percent'],
        reverse=True
    )[:15]
    
    for field, info in important_fields:
        coverage_style = "green" if info['coverage_percent'] > 90 else "yellow" if info['coverage_percent'] > 70 else "red"
        sample = str(info['sample_values'][:2])[:40] if info['sample_values'] else ""
        
        schema_table.add_row(
            field,
            info['primary_type'],
            f"[{coverage_style}]{info['coverage_percent']:.0f}%[/{coverage_style}]",
            str(info.get('cardinality', '?')),
            sample
        )
    
    console.print(schema_table)
    console.print()
    
    # Data quality
    if analysis['data_quality']['high_null_fields']:
        console.print("[bold yellow]⚠️  Fields with High Null Rates:[/bold yellow]")
        for field in analysis['data_quality']['high_null_fields'][:8]:
            console.print(f"  • {field['field']}: {field['null_percent']:.1f}% null")
        console.print()
    
    # Array patterns
    if analysis['patterns']['array_patterns']:
        console.print("[bold blue]📚 Array Fields:[/bold blue]")
        for arr in analysis['patterns']['array_patterns'][:5]:
            console.print(f"  • {arr['field']}: {arr['element_type']}[] (avg: {arr['avg_length']}, max: {arr['max_length']})")
        console.print()
    
    # Sample document
    console.print("[bold cyan]📄 Sample Document Structure:[/bold cyan]")
    sample_json = json.dumps(analysis['sample_document'], indent=2, default=str)
    # Show up to 5000 characters instead of 1000
    console.print(sample_json[:5000])
    if len(sample_json) > 5000:
        console.print("\n  [dim]... (truncated)[/dim]")
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze MongoDB database structure and patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all databases
  python analyze_mongodb.py mongodb://localhost:27017
  
  # Analyze entire database
  python analyze_mongodb.py mongodb://localhost:27017/temporary
  
  # Analyze specific collection
  python analyze_mongodb.py mongodb://localhost:27017/temporary/release
  
  # Quick analysis with smaller sample
  python analyze_mongodb.py mongodb://localhost:27017/temporary/release --sample 100
  
  # Export results to JSON
  python analyze_mongodb.py mongodb://localhost:27017/temporary/release --output analysis.json
        """
    )
    
    parser.add_argument(
        'uri',
        help='MongoDB URI: mongodb://host:port[/database[/collection]]'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=1000,
        help='Number of documents to sample (default: 1000)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Export analysis results to JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = MongoDBAnalyzer(args.uri, sample_size=args.sample)
        analysis = analyzer.analyze_all()
        
        print_analysis_report(analysis)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            console.print(f"[green]✓ Exported results to: {args.output}[/green]")
        
        analyzer.close()
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

