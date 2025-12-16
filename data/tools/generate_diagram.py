#!/usr/bin/env python3
"""
Parquet Schema to Database Diagram Generator

Generates visual database diagrams from Parquet files to make the schema
interpretable and document relationships between tables.

Output formats:
- Terminal (Rich visualization)
- Mermaid ERD (for Markdown/documentation)
- Text summary

Usage:
    python generate_diagram.py path/to/parquet/data/
    python generate_diagram.py path/to/parquet/data/ --output schema.md
    python generate_diagram.py path/to/parquet/data/ --format mermaid
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich import box

console = Console()


class ParquetSchemaAnalyzer:
    """Analyze parquet file structure and relationships"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.schemas = {}
        self.relationships = []
        
    def analyze(self) -> Dict[str, Any]:
        """Analyze all parquet files in directory"""
        
        parquet_files = list(self.data_dir.glob('*.parquet'))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        
        console.print(f"[cyan]→ Analyzing {len(parquet_files)} parquet files...[/cyan]")
        
        # Analyze each file
        for file in sorted(parquet_files):
            table_name = file.stem
            df = pd.read_parquet(file)
            
            self.schemas[table_name] = {
                'file': file.name,
                'row_count': len(df),
                'columns': self._analyze_columns(df),
                'primary_key': self._detect_primary_key(table_name, df),
                'foreign_keys': self._detect_foreign_keys(table_name, df)
            }
        
        # Detect relationships between tables
        self._detect_relationships()
        
        return {
            'tables': self.schemas,
            'relationships': self.relationships,
            'table_count': len(self.schemas),
            'total_rows': sum(s['row_count'] for s in self.schemas.values())
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze columns in a dataframe"""
        
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
            
            # Detect if column is nullable
            nullable = null_count > 0
            
            # Detect if column might be unique (potential PK/FK)
            unique_count = df[col].nunique()
            is_unique = unique_count == len(df) and len(df) > 0
            
            columns.append({
                'name': col,
                'type': self._map_dtype(dtype),
                'nullable': nullable,
                'null_percent': round(null_pct, 1),
                'is_unique': is_unique,
                'cardinality': unique_count
            })
        
        return columns
    
    def _map_dtype(self, dtype: str) -> str:
        """Map pandas dtype to SQL-like type"""
        
        mapping = {
            'object': 'VARCHAR',
            'string': 'VARCHAR',
            'int64': 'BIGINT',
            'int32': 'INTEGER',
            'int16': 'SMALLINT',
            'int8': 'TINYINT',
            'float64': 'DOUBLE',
            'float32': 'FLOAT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'category': 'VARCHAR'
        }
        
        for key, value in mapping.items():
            if key in dtype:
                return value
        
        return dtype.upper()
    
    def _detect_primary_key(self, table_name: str, df: pd.DataFrame) -> Optional[str]:
        """Detect primary key column"""
        
        # Common PK patterns
        pk_candidates = [
            f"{table_name}_id",
            f"{table_name.rstrip('s')}_id",  # Remove plural
            'id',
            f"{table_name}_key"
        ]
        
        for candidate in pk_candidates:
            if candidate in df.columns:
                # Check if it's unique
                if df[candidate].nunique() == len(df):
                    return candidate
        
        # Check for any column with 'id' in name that's unique
        for col in df.columns:
            if 'id' in col.lower() and df[col].nunique() == len(df):
                return col
        
        return None
    
    def _detect_foreign_keys(self, table_name: str, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Detect foreign key columns"""
        
        foreign_keys = []
        
        for col in df.columns:
            # Look for columns ending in _id (common FK pattern)
            if col.endswith('_id') and col != f"{table_name}_id":
                # Extract referenced table name
                ref_table = col.replace('_id', '')
                
                # Could be plural
                foreign_keys.append({
                    'column': col,
                    'references_table': ref_table,
                    'references_column': col
                })
        
        return foreign_keys
    
    def _detect_relationships(self):
        """Detect relationships between tables based on foreign keys"""
        
        for table_name, schema in self.schemas.items():
            for fk in schema['foreign_keys']:
                ref_table = fk['references_table']
                
                # Skip self-references
                if ref_table == table_name:
                    continue
                
                # Check if referenced table exists (handle plurals)
                if ref_table in self.schemas:
                    relationship_type = self._determine_relationship_type(
                        table_name, 
                        fk['column'],
                        ref_table
                    )
                    
                    self.relationships.append({
                        'from_table': table_name,
                        'from_column': fk['column'],
                        'to_table': ref_table,
                        'to_column': fk['references_column'],
                        'type': relationship_type
                    })
                elif ref_table + 's' in self.schemas:
                    # Try plural form
                    ref_table_plural = ref_table + 's'
                    relationship_type = self._determine_relationship_type(
                        table_name, 
                        fk['column'],
                        ref_table_plural
                    )
                    
                    self.relationships.append({
                        'from_table': table_name,
                        'from_column': fk['column'],
                        'to_table': ref_table_plural,
                        'to_column': fk['references_column'],
                        'type': relationship_type
                    })
    
    def _determine_relationship_type(self, from_table: str, fk_column: str, to_table: str) -> str:
        """Determine relationship type (1:1, 1:N, N:M)"""
        
        # Simple heuristic: if FK column is unique, it's 1:1, otherwise 1:N
        from_schema = self.schemas[from_table]
        
        for col in from_schema['columns']:
            if col['name'] == fk_column:
                if col['is_unique']:
                    return '1:1'
                else:
                    return '1:N'
        
        return '1:N'  # Default to one-to-many


class DiagramGenerator:
    """Generate various diagram formats"""
    
    def __init__(self, analysis: Dict[str, Any]):
        self.analysis = analysis
    
    def generate_terminal(self):
        """Generate rich terminal visualization"""
        
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Parquet Schema Diagram[/bold cyan]\n"
            f"Tables: {self.analysis['table_count']}\n"
            f"Total Rows: {self.analysis['total_rows']:,}",
            box=box.DOUBLE
        ))
        console.print()
        
        # Display each table
        for table_name, schema in sorted(self.analysis['tables'].items()):
            self._print_table(table_name, schema)
        
        # Display relationships
        if self.analysis['relationships']:
            console.print("\n[bold cyan]═══ Relationships ═══[/bold cyan]\n")
            
            rel_table = Table(box=box.SIMPLE)
            rel_table.add_column("From", style="yellow")
            rel_table.add_column("Type", style="cyan", justify="center")
            rel_table.add_column("To", style="green")
            
            for rel in self.analysis['relationships']:
                rel_table.add_row(
                    f"{rel['from_table']}.{rel['from_column']}",
                    f"[bold]{rel['type']}[/bold]",
                    f"{rel['to_table']}.{rel['to_column']}"
                )
            
            console.print(rel_table)
            console.print()
    
    def _print_table(self, table_name: str, schema: Dict[str, Any]):
        """Print a single table schema"""
        
        # Create table
        table = Table(
            title=f"[bold yellow]{table_name}[/bold yellow] ({schema['row_count']:,} rows)",
            box=box.SIMPLE_HEAD,
            show_header=True
        )
        
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Nullable", justify="center")
        table.add_column("Notes", style="dim")
        
        # Add columns
        for col in schema['columns']:
            nullable = "✓" if col['nullable'] else "✗"
            
            notes = []
            if schema['primary_key'] == col['name']:
                notes.append("PK")
            
            for fk in schema['foreign_keys']:
                if fk['column'] == col['name']:
                    notes.append(f"→ {fk['references_table']}")
            
            if col['is_unique'] and schema['primary_key'] != col['name']:
                notes.append("UNIQUE")
            
            if col['null_percent'] > 50:
                notes.append(f"{col['null_percent']:.0f}% null")
            
            notes_str = ", ".join(notes)
            
            table.add_row(
                col['name'],
                col['type'],
                nullable,
                notes_str
            )
        
        console.print(table)
        console.print()
    
    def generate_mermaid(self) -> str:
        """Generate Mermaid ERD diagram"""
        
        lines = ["```mermaid", "erDiagram"]
        
        # Define tables
        for table_name, schema in sorted(self.analysis['tables'].items()):
            lines.append(f"    {table_name.upper()} {{")
            
            for col in schema['columns']:
                col_type = col['type']
                col_name = col['name']
                
                # Build constraint string (Mermaid format)
                # Use single words or quoted strings to avoid parse errors
                constraint_parts = []
                
                if schema['primary_key'] == col['name']:
                    constraint_parts.append("PK")
                
                is_fk = False
                for fk in schema['foreign_keys']:
                    if fk['column'] == col['name']:
                        is_fk = True
                        break
                
                if is_fk:
                    constraint_parts.append("FK")
                
                if col['is_unique'] and schema['primary_key'] != col['name']:
                    constraint_parts.append("UNIQUE")
                
                # Format: type name "constraints"
                if constraint_parts:
                    # Use quoted string for multiple constraints
                    constraint_str = f" \"{' '.join(constraint_parts)}\""
                else:
                    constraint_str = ""
                
                # Add comment for nullable fields instead of inline
                comment = ""
                if col['nullable'] and not constraint_parts:
                    comment = " \"nullable\""
                
                lines.append(f"        {col_type} {col_name}{constraint_str}{comment}")
            
            lines.append("    }")
        
        # Define relationships
        for rel in self.analysis['relationships']:
            # Mermaid relationship notation
            if rel['type'] == '1:1':
                connector = "||--||"
            elif rel['type'] == '1:N':
                connector = "||--o{"
            else:
                connector = "}o--o{"
            
            lines.append(
                f"    {rel['to_table'].upper()} {connector} "
                f"{rel['from_table'].upper()} : \"{rel['from_column']}\""
            )
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def generate_summary(self) -> str:
        """Generate text summary"""
        
        lines = [
            "# Parquet Schema Documentation",
            "",
            "**Auto-generated from MongoDB → Parquet transformation**",
            "",
            f"- **Tables:** {self.analysis['table_count']}",
            f"- **Total Rows:** {self.analysis['total_rows']:,}",
            f"- **Relationships:** {len(self.analysis['relationships'])}",
            "",
            "---",
            "",
            "## Table Overview",
            ""
        ]
        
        # Add table overview with descriptions
        lines.append("| Table | Rows | Description |")
        lines.append("|-------|-----:|-------------|")
        
        table_descriptions = {
            'experiments': 'Main experiment records with target formulas and status',
            'experiment_elements': 'Elements present in each experiment',
            'dosing_sessions': 'Powder dosing session parameters',
            'powder_doses': 'Individual powder doses with accuracy tracking',
            'heating_sessions': 'Heating/calcination parameters and profiles',
            'temperature_logs': 'Time-series temperature measurements during heating',
            'powder_recovery': 'Post-heating powder collection and yield data',
            'xrd_measurements': 'XRD measurement metadata and parameters',
            'xrd_data_points': 'Raw XRD diffraction pattern data (2θ vs counts)',
            'sample_finalization': 'Sample labeling and storage information',
            'workflow_tasks': 'Lab automation workflow task tracking'
        }
        
        for table_name, schema in sorted(self.analysis['tables'].items()):
            description = table_descriptions.get(table_name, 'Data table')
            lines.append(f"| {table_name} | {schema['row_count']:,} | {description} |")
        
        lines.extend(["", "---", ""])
        
        # List tables with row counts
        lines.append("## Table Details")
        lines.append("")
        
        for table_name, schema in sorted(self.analysis['tables'].items()):
            pk_str = f"`{schema['primary_key']}`" if schema['primary_key'] else "None"
            lines.append(f"### {table_name}")
            lines.append(f"- **Rows:** {schema['row_count']:,}")
            lines.append(f"- **Primary Key:** {pk_str}")
            
            # Add foreign key info
            if schema['foreign_keys']:
                fk_list = [f"`{fk['column']}` → `{fk['references_table']}`" for fk in schema['foreign_keys']]
                lines.append(f"- **Foreign Keys:** {', '.join(fk_list)}")
            
            lines.append("")
            
            # Column list
            lines.append("| Column | Type | Nullable | Notes |")
            lines.append("|--------|------|----------|-------|")
            
            for col in schema['columns']:
                nullable = "Yes" if col['nullable'] else "No"
                
                notes = []
                if schema['primary_key'] == col['name']:
                    notes.append("Primary Key")
                
                for fk in schema['foreign_keys']:
                    if fk['column'] == col['name']:
                        notes.append(f"FK → {fk['references_table']}")
                
                if col['is_unique'] and schema['primary_key'] != col['name']:
                    notes.append("Unique")
                
                notes_str = ", ".join(notes) if notes else "-"
                
                lines.append(f"| {col['name']} | {col['type']} | {nullable} | {notes_str} |")
            
            lines.append("")
        
        # Relationships
        if self.analysis['relationships']:
            lines.append("---")
            lines.append("")
            lines.append("## Relationships")
            lines.append("")
            lines.append("| From | Type | To |")
            lines.append("|------|------|-----|")
            
            for rel in self.analysis['relationships']:
                lines.append(
                    f"| {rel['from_table']}.{rel['from_column']} | "
                    f"{rel['type']} | "
                    f"{rel['to_table']}.{rel['to_column']} |"
                )
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate database diagram from Parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display in terminal
  python generate_diagram.py data/parquet/
  
  # Generate Mermaid ERD
  python generate_diagram.py data/parquet/ --format mermaid
  
  # Export to markdown file
  python generate_diagram.py data/parquet/ --output schema.md
        """
    )
    
    parser.add_argument(
        'data_dir',
        type=Path,
        help='Directory containing parquet files'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['terminal', 'mermaid', 'summary', 'all'],
        default='terminal',
        help='Output format (default: terminal)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file (markdown format)'
    )
    
    args = parser.parse_args()
    
    try:
        # Analyze schema
        analyzer = ParquetSchemaAnalyzer(args.data_dir)
        analysis = analyzer.analyze()
        
        generator = DiagramGenerator(analysis)
        
        # Generate requested format
        if args.format == 'terminal' or args.format == 'all':
            generator.generate_terminal()
        
        # Generate output file
        if args.output or args.format in ['mermaid', 'summary', 'all']:
            output_lines = []
            
            if args.format in ['summary', 'all']:
                output_lines.append(generator.generate_summary())
            
            if args.format in ['mermaid', 'all']:
                if args.format == 'all':
                    output_lines.append("\n---\n")
                output_lines.append("## Entity Relationship Diagram\n")
                output_lines.append(generator.generate_mermaid())
            
            output_content = "\n".join(output_lines)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_content)
                console.print(f"[green]✓ Saved to: {args.output}[/green]")
            else:
                console.print(output_content)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

