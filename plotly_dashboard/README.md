# Berkeley DOE A-Lab Material Discovery Dashboard

## Plotly Dash Version

This is a Python-based Plotly Dash dashboard that replicates the functionality of the Next.js version. It provides interactive visualizations for material discovery data from Berkeley's DOE A-Lab.

## Features

- 🔐 Password-protected access
- 📊 Interactive Plotly charts and visualizations
- 🔬 Sample information and target composition analysis
- 🧪 Precursor dosing accuracy tracking
- 🌡️ Temperature profile visualization
- 📐 XRD characterization plots
- 🔬 SEM-EDS cluster analysis
- 🎨 Modern, responsive Bootstrap UI

## Quick Start

From the project root:

```bash
# 1. Generate data from MongoDB (first time or when updating)
./update_data.sh

# 2. Launch dashboard (auto-setup on first run)
./run_dashboard.sh
```

That's it! The dashboard will:

- Automatically create a virtual environment (first run)
- Install dependencies (first run)
- Launch at `http://localhost:8050`

## Options

```bash
./run_dashboard.sh           # Launch with password
./run_dashboard.sh --no-pass # Launch without password
./run_dashboard.sh --help    # Show help
```

**Default password:** `alab` (can be changed in `app.py` line 1158)

## Project Structure

```
plotly_dashboard/
├── app.py                     # Main Dash application
├── parquet_data_loader.py     # Parquet data loader
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── venv/                      # Virtual environment (auto-created)

Project root:
├── run_dashboard.sh           # Dashboard launcher (handles setup automatically)
└── update_data.sh             # Data update script
```

## Key Differences from Next.js Version

### Data Source:

- Loads data directly from Parquet files via `parquet_data_loader.py`
- No intermediate JSON files needed
- Fast, efficient data access
- Same password authentication
- Matching color schemes and layout

### Technical Differences:

- **Backend**: Python (Dash/Flask) vs Node.js (Next.js)
- **Frontend**: Dash HTML components vs React/JSX
- **Charts**: Plotly.js (via Dash) vs Recharts
- **Styling**: Bootstrap (dash-bootstrap-components) vs Tailwind CSS
- **Server**: Single-file Dash app vs Next.js app router

### Advantages of Dash Version:

- ✅ Native Python - better integration with scientific computing libraries
- ✅ Server-side rendering - no client-side JavaScript needed
- ✅ Simpler deployment for Python environments
- ✅ Built-in Plotly interactivity
- ✅ Easier to extend with Python data analysis tools (NumPy, Pandas, SciPy, etc.)

### Advantages of Next.js Version:

- ✅ Better performance for complex UIs
- ✅ More flexible component architecture
- ✅ Better SEO capabilities
- ✅ Richer ecosystem of UI libraries

## Customization

### Changing the Password

Edit line 848 in `app.py`:

```python
if password == "your_new_password":
```

### Modifying Colors

Color palettes are defined at the top of `app.py`:

```python
ELEMENT_COLORS = ['#FF6B6B', '#4ECDC4', ...]
POWDER_COLORS = ['#e74c3c', '#3498db', ...]
```

### Adding New Visualizations

1. Create a new function that returns a Plotly figure:

```python
def create_my_chart(data):
    fig = go.Figure(...)
    return fig
```

2. Add the chart to the layout:

```python
dcc.Graph(figure=create_my_chart(data))
```

## Data Sources

The dashboard reads from:

- `../public/data.json` - Main sample data
- `../public/sem_analysis_summary.json` - SEM-EDS analysis data

Make sure these files are accessible from the dashboard directory.

## Troubleshooting

### Port Already in Use

If port 8050 is already in use, change it in `app.py` (last line):

```python
app.run_server(debug=True, port=8051)
```

### Missing Data Files

Ensure the paths to data files are correct:

```python
with open('../public/data.json', 'r') as f:
```

### Images Not Loading

Make sure all images are in the `assets/` folder. Dash automatically serves files from this directory.

## Development

To run in development mode with hot reloading:

```bash
python app.py
```

The `debug=True` flag enables:

- Hot reloading on code changes
- Detailed error messages
- Developer tools

## Production Deployment

For production, consider:

1. **Use a production WSGI server**:

```bash
pip install gunicorn
gunicorn app:server -b 0.0.0.0:8050
```

2. **Set debug=False** in `app.py`

3. **Use environment variables** for sensitive data:

```python
import os
PASSWORD = os.getenv('DASHBOARD_PASSWORD', 'default')
```

4. **Deploy to**:
   - Heroku
   - AWS Elastic Beanstalk
   - Google Cloud Run
   - Azure App Service
   - DigitalOcean App Platform

## License

Same as the parent project.

## Support

For questions about:

- **The data**: Contact Berkeley DOE A-Lab team
- **The dashboard**: Check the Next.js version for reference implementation
- **Dash framework**: Visit https://dash.plotly.com/

## Credits

Converted from the original Next.js implementation to Plotly Dash.
