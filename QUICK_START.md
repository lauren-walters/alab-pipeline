# A-Lab Pipeline Quick Start

## Installation

```bash
# Install dependencies (uses existing data/venv)
source data/venv/bin/activate
pip install -r data/requirements.txt
```

**Note**: The default schema (`data/products/schema/`) contains all Pydantic schemas that are auto-discovered.

## Run Tests

```bash
# Run comprehensive integration tests
./run_tests.sh

# Tests validate:
# - Auto-discovery (schemas, analyses)
# - Hook system functionality
# - Filter configurations
# - Edge cases and error handling
# - Full pipeline integration
```

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        A-Lab Pipeline Flow                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  run_product_pipeline.sh                                                 │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────┐     Uses shared primitive:                         │
│  │ 1. Filter       │     mongodb_to_parquet.py                          │
│  │ 2. Transform    │────────────────────────────────────────────────────│
│  │ 3. Analyze      │     Calls run_analysis.sh or built-in analyzers    │
│  │ 4. Validate     │     Uses auto-discovered Pydantic schemas          │
│  │ 5. Diagram      │     Generates SCHEMA_DIAGRAM.md for product        │
│  │ 6. Upload       │     S3 OpenData (parquet files)                    │
│  └─────────────────┘                                                     │
│                                                                          │
│  run_dashboard.sh  ◄──── Separate: operates on parquet files            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Create Your First Data Product

```bash
# Interactive product creation (auto-discovers schemas)
./run_product_pipeline.sh create

# Follow prompts to configure:
# - Product name (e.g., "reaction_genome")
# - Experiment types (NSC, Na, PG, etc.)
# - Analyses to run (XRD, powder statistics)
# - Schema (auto-loaded from data/products/schema/)
```

## Run the Pipeline

```bash
# Dry run (default) - see what would be uploaded
./run_product_pipeline.sh run --product reaction_genome

# Live upload to S3 OpenData
./run_product_pipeline.sh run --product reaction_genome --upload

# Run specific stages only
./run_product_pipeline.sh run --product reaction_genome --stages filter transform diagram
```

## Check Status

```bash
# List all products
./run_product_pipeline.sh list

# Check specific product status
./run_product_pipeline.sh status --product reaction_genome

# Validate configuration
./run_product_pipeline.sh validate --product reaction_genome
```

## Pipeline Stages

| Stage         | Description                       | Output                       |
| ------------- | --------------------------------- | ---------------------------- |
| **filter**    | Select experiments from MongoDB   | `experiments.txt`            |
| **transform** | Convert MongoDB → Parquet         | `parquet/*.parquet`          |
| **analyze**   | Run enabled analyses (XRD, etc.)  | `analysis_results/*.parquet` |
| **validate**  | Validate against Pydantic schemas | Errors/warnings logged       |
| **diagram**   | Generate schema documentation     | `SCHEMA_DIAGRAM.md`          |
| **upload**    | Upload to S3 OpenData             | Cloud storage                |

---

## 🔌 Extending the Pipeline

The pipeline uses **auto-discovery** for schemas and analyses. No code changes needed!

### Adding a New Schema (e.g., SEM data)

1. **Create the schema file**:

```bash
# Create: data/products/schema/sem_data.py
```

```python
"""SEM Data Schema"""
from pydantic import BaseModel, Field

class SEMData(BaseModel, extra="forbid"):
    """SEM measurement data for experiments"""

    # Set table name (optional - defaults to filename)
    __schema_table__ = "sem_data"

    experiment_id: str = Field(description="Experiment identifier")
    image_count: int = Field(description="Number of SEM images")
    magnification: float | None = Field(default=None, description="Magnification level")
    morphology_class: str | None = Field(default=None, description="Classified morphology")
```

2. **Update parquet transformer** (if extracting new data from MongoDB):

```python
# In mongodb_to_parquet.py, add extraction logic for SEM data
```

3. **Done!** Schema is auto-discovered on next pipeline run.

### Adding a New Analysis

1. **Create the analyzer file**:

```bash
# Create: data/analyses/sem_analyzer.py
```

```python
"""SEM Morphology Analysis"""
from pathlib import Path
import pandas as pd
from base_analyzer import BaseAnalyzer

class SEMAnalyzer(BaseAnalyzer):
    """Analyze SEM images for morphology patterns"""

    # Class attributes for discovery
    name = "sem_clustering"
    description = "Cluster SEM images by morphology"
    cli_flag = "--sem"

    def analyze(self, experiments_df: pd.DataFrame, parquet_dir: Path) -> pd.DataFrame:
        """Run SEM analysis"""
        results = []

        for _, exp in experiments_df.iterrows():
            # Your analysis logic here
            results.append({
                'experiment_name': exp['name'],
                'cluster_id': self._compute_cluster(exp),
                'morphology_score': self._compute_score(exp)
            })

        return pd.DataFrame(results)

    def get_output_schema(self):
        return {
            'cluster_id': {'type': 'int', 'required': True, 'description': 'Cluster assignment'},
            'morphology_score': {'type': 'float', 'required': False, 'description': 'Similarity score'}
        }

    def _compute_cluster(self, exp):
        # Implement clustering logic
        return 0

    def _compute_score(self, exp):
        # Implement scoring logic
        return 0.0
```

2. **Done!** Analysis is auto-discovered. Enable in product config:

```yaml
# In your product's config.yaml
analyses:
  - name: sem_clustering
    enabled: true
    config:
      num_clusters: 5
```

### Adding a New Filter Type

1. **Edit filter presets**:

```bash
# Edit: data/config/filters.yaml
```

```yaml
# Add your filter
my_custom_filter:
  description: 'Experiments with SEM data'
  status:
    - completed
  has_sem: true # Add new filter field
```

2. **Update filter logic** (if adding new filter field):

```python
# In data/products/base_product.py, update ExperimentFilter class
class ExperimentFilter(BaseModel):
    has_sem: Optional[bool] = Field(None, description="Must have SEM data")
```

---

## Configuration Files

| File                               | Purpose                             |
| ---------------------------------- | ----------------------------------- |
| `data/config/defaults.yaml`        | Global pipeline defaults            |
| `data/config/filters.yaml`         | Filter presets and experiment types |
| `data/config/analyses.yaml`        | Analysis documentation and defaults |
| `data/products/{name}/config.yaml` | Product-specific configuration      |

## Auto-Discovery Summary

| Component    | Directory                  | How to Add                            |
| ------------ | -------------------------- | ------------------------------------- |
| **Schemas**  | `data/products/schema/`    | Create `*.py` with Pydantic BaseModel |
| **Analyses** | `data/analyses/`           | Create `*.py` inheriting BaseAnalyzer |
| **Filters**  | `data/config/filters.yaml` | Add YAML entry                        |

---

## Scheduled Runs

Use cron for automation:

```bash
# Daily at 2 AM
0 2 * * * cd /path/to/A-Lab_Samples && ./run_product_pipeline.sh run --product reaction_genome --upload >> logs/pipeline.log 2>&1
```

## Troubleshooting

### Missing dependencies

```bash
source data/venv/bin/activate
pip install -r data/requirements.txt
```

### MongoDB not running

```bash
brew services start mongodb-community@7.0
```

### No experiments match filter

Check filter criteria in `data/products/<product>/config.yaml`

### Analysis not found

Run to see available analyzers:

```bash
python data/analyses/base_analyzer.py
```

### Schema not discovered

Run to see available schemas:

```bash
python data/products/schema_manager.py
```

## Documentation

- **PIPELINE_ARCHITECTURE.md** - Complete architecture overview
- **data/pipeline/README.md** - Pipeline component details
- **data/products/SCHEMA_SYSTEM.md** - Schema system documentation
- **data/config/analyses.yaml** - Analysis documentation
