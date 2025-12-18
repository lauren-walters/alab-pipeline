"""
Berkeley DOE A-Lab Material Discovery Dashboard
Built with Plotly Dash

Data Source: Parquet files (loaded via ParquetDataLoader)
"""

import json
import base64
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import re

# Check for no-auth mode
NO_AUTH = os.environ.get('ALAB_NO_AUTH', '0') == '1'

# Import data loader from same directory
from parquet_data_loader import ParquetDataLoader

# Initialize data loader
try:
    data_loader = ParquetDataLoader()
    print(f"✓ Loaded Parquet data: {len(data_loader.get_experiment_list())} experiments")
except Exception as e:
    print(f"⚠ Error loading Parquet data: {e}")
    print("  Run ./update_data.sh to generate Parquet files")
    data_loader = None

# Initialize the Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Berkeley DOE A-Lab"

# Color palettes (matching Next.js version)
ELEMENT_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788']

POWDER_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                 '#1abc9c', '#34495e', '#e67e22', '#95a5a6']

CLUSTER_COLORS = ['#e74c3c', '#3498db', '#2ecc71']


def create_sem_not_available_alert():
    """Create a reusable alert for when SEM data is not available"""
    return dbc.Alert([
        html.Div([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("SEM Analysis Not Available")
        ], className="mb-2"),
        html.P([
            "SEM-EDS (Scanning Electron Microscopy with Energy Dispersive X-ray Spectroscopy) ",
            "data is not currently integrated into the automated data pipeline."
        ], className="mb-0")
    ], color="info")


def parse_formula(formula):
    """Parse chemical formula to element composition"""
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    composition = {}
    
    for element, count in matches:
        if element and element not in ['(', ')']:
            amount = float(count) if count else 1.0
            composition[element] = composition.get(element, 0) + amount
    
    return composition


def analyze_data(data):
    """Analyze sample data and extract key metrics"""
    metadata = data['metadata']
    target = metadata['target']
    powder_dosing = metadata['powderdosing_results']
    powders = powder_dosing['Powders']
    
    # Parse formula
    composition = parse_formula(target)
    total = sum(composition.values())
    percentages = {el: (amt / total) * 100 for el, amt in composition.items()}
    
    # Parse precursors
    precursors = []
    for powder in powders:
        doses = powder['Doses']
        actual_mass = sum(dose['Mass'] for dose in doses)
        precursors.append({
            'name': powder['PowderName'],
            'target_mass': powder['TargetMass'],
            'actual_mass': actual_mass,
            'num_doses': len(doses),
            'accuracy': (actual_mass / powder['TargetMass']) * 100
        })
    
    total_target = sum(p['target_mass'] for p in precursors)
    total_actual = sum(p['actual_mass'] for p in precursors)
    
    return {
        'sample_info': {
            'name': data['name'],
            'id': data['_id'],
            'last_updated': data['last_updated'],
            'target_formula': target
        },
        'target_composition': {
            'elements': composition,
            'total_atoms': total,
            'percentages': percentages
        },
        'precursors': precursors,
        'powder_totals': {
            'target_mass_g': total_target,
            'actual_mass_g': total_actual,
            'transfer_mass_mg': powder_dosing['ActualTransferMass']
        }
    }


def load_experiment_list():
    """Load list of all experiments from Parquet data"""
    if data_loader is None:
        return []
    
    try:
        exp_names = data_loader.get_experiment_list()
        # Convert to format expected by dashboard (list of dicts with 'name')
        return [{'name': name} for name in exp_names]
    except Exception as e:
        print(f"Error loading experiments: {e}")
        return []


def load_experiment_data(experiment_name):
    """Load data for a specific experiment from Parquet files
    
    Note: SEM-EDS analysis data is not included in the Parquet pipeline.
    SEM data must be accessed separately from individual experiment folders.
    The dashboard will show a notice when SEM data is not available.
    """
    if data_loader is None:
        return None, None, None
    
    try:
        # Load from parquet using ParquetDataLoader
        raw_data = data_loader.get_complete_experiment_data(experiment_name)
        
        if raw_data and 'error' not in raw_data:
            sample_data = analyze_data(raw_data)
            
            # SEM data is not part of the MongoDB → Parquet pipeline
            # It exists only in separate experiment folders (e.g., experiments/NSC_249/)
            sem_data = None
            
            # Return in correct order: sample_data, raw_data, sem_data
            return sample_data, raw_data, sem_data
        else:
            return None, None, None
    except Exception as e:
        print(f"Error loading experiment {experiment_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def load_data():
    """Load sample data and SEM analysis - backward compatibility"""
    # Try to load NSC_249 by default, or first experiment from index
    experiments = load_experiment_list()
    
    if experiments:
        # Use first experiment from index
        experiment_name = experiments[0]['name']
    else:
        # Default to NSC_249 or try loading old format
        experiment_name = 'NSC_249'
    
    return load_experiment_data(experiment_name)


# Load experiment list for dropdown
experiment_list = load_experiment_list()
default_experiment = experiment_list[0]['name'] if experiment_list else 'NSC_249'

# Load initial data
sample_data, raw_data, sem_data = load_experiment_data(default_experiment)


def create_element_pie_chart(sample_data):
    """Create pie chart for element composition"""
    elements = sample_data['target_composition']['elements']
    percentages = sample_data['target_composition']['percentages']
    
    sorted_elements = sorted(elements.items(), key=lambda x: x[1], reverse=True)
    labels = [el for el, _ in sorted_elements]
    values = [amt for _, amt in sorted_elements]
    colors = ELEMENT_COLORS[:len(labels)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hovertemplate='<b>%{label}</b><br>%{value:.3f} atoms<br>%{percent}<extra></extra>',
        texttemplate='%{label}: %{percent}'
    )])
    
    fig.update_layout(
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_powder_pie_chart(sample_data):
    """Create pie chart for powder composition"""
    precursors = sample_data['precursors']
    labels = [p['name'][:15] for p in precursors]
    values = [p['actual_mass'] for p in precursors]
    colors = POWDER_COLORS[:len(labels)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hovertemplate='<b>%{label}</b><br>%{value:.5f}g<extra></extra>'
    )])
    
    fig.update_layout(
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_temperature_chart(raw_data):
    """Create temperature profile chart"""
    metadata = raw_data.get('metadata', {})
    heating_results = metadata.get('heating_results', {})
    
    if 'temperature_log' not in heating_results:
        return None
    
    temp_log = heating_results['temperature_log']
    time_hours = [float(t) / 60 for t in temp_log['time_minutes']]
    temperatures = [float(t) for t in temp_log['temperature_celsius']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_hours,
        y=temperatures,
        mode='lines',
        line=dict(color='#f97316', width=3),
        name='Temperature'
    ))
    
    fig.update_layout(
        xaxis_title='Time (hours)',
        yaxis_title='Temperature (°C)',
        height=400,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,1)',
        margin=dict(t=30, b=50, l=50, r=30)
    )
    
    return fig


def create_xrd_chart(raw_data):
    """Create XRD pattern chart"""
    metadata = raw_data.get('metadata', {})
    diffraction_results = metadata.get('diffraction_results', {})
    
    if 'twotheta' not in diffraction_results or 'counts' not in diffraction_results:
        return None
    
    two_theta = [float(x) for x in diffraction_results['twotheta']]
    counts = [float(c) for c in diffraction_results['counts']]
    max_counts = max(counts)
    normalized_counts = [(c / max_counts) * 100 for c in counts]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=two_theta,
        y=normalized_counts,
        mode='lines',
        line=dict(color='#8b5cf6', width=2),
        name='XRD Pattern'
    ))
    
    fig.update_layout(
        xaxis_title='2θ (degrees)',
        yaxis_title='Intensity (%)',
        height=400,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,1)',
        margin=dict(t=30, b=50, l=50, r=30)
    )
    
    return fig


def create_xrd_analysis_section(raw_data):
    """Create XRD phase analysis section if DARA results are available"""
    metadata = raw_data.get('metadata', {})
    xrd_analysis = metadata.get('xrd_analysis')
    
    if not xrd_analysis:
        return None
    
    # Handle failed analyses
    if not xrd_analysis.get('success', True):  # Default True for backwards compat
        error = xrd_analysis.get('error', 'Unknown error')
        error_type = xrd_analysis.get('error_type', 'analysis_error')
        
        # Map error types to user-friendly messages
        error_messages = {
            'no_peaks_detected': 'No crystalline peaks detected - sample may be amorphous',
            'cif_download': 'Unable to download reference structures from COD database',
            'timeout': 'Analysis timed out - please retry',
            'no_chemical_system': 'Chemical system not defined for this experiment',
            'dara_internal_error': 'Internal DARA processing error',
            'analysis_error': 'Analysis failed'
        }
        friendly_error = error_messages.get(error_type, error)
        
        return dbc.Card([
            dbc.CardBody([
                html.H2("🔬 XRD Phase Analysis (DARA)", className="h4 mb-3"),
                dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    html.Strong("Analysis Failed: "),
                    friendly_error,
                    html.Br(),
                    html.Small(f"Error: {error}", className="text-muted")
                ], color="warning", className="mb-0")
            ])
        ], className="shadow-sm mb-4")
    
    phases = xrd_analysis.get('phases', [])
    rwp = xrd_analysis.get('rwp', 0)
    rp = xrd_analysis.get('rp', 0)
    
    # Determine quality badge color based on Rwp
    if rwp < 15:
        quality_color = "success"
        quality_text = "Excellent"
    elif rwp < 25:
        quality_color = "warning"
        quality_text = "Good"
    else:
        quality_color = "danger"
        quality_text = "Poor"
    
    return dbc.Card([
        dbc.CardBody([
            html.H2("🔬 XRD Phase Analysis (DARA)", className="h4 mb-4"),
            
            # Quality metrics row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.P("Rwp", className="small text-muted mb-1"),
                            html.H3(f"{rwp:.2f}%", className=f"h4 mb-0 text-{quality_color}")
                        ])
                    ], className="bg-light border-0")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.P("Rp", className="small text-muted mb-1"),
                            html.H3(f"{rp:.2f}%", className="h4 mb-0")
                        ])
                    ], className="bg-light border-0")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.P("Phases Found", className="small text-muted mb-1"),
                            html.H3(str(len(phases)), className="h4 mb-0 text-primary")
                        ])
                    ], className="bg-light border-0")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.P("Fit Quality", className="small text-muted mb-1"),
                            dbc.Badge(quality_text, color=quality_color, className="h5")
                        ])
                    ], className="bg-light border-0")
                ], md=3)
            ], className="mb-4"),
            
            # Phases table
            html.H3("Identified Phases", className="h6 mb-3"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Phase"),
                        html.Th("Spacegroup"),
                        html.Th("Weight %", className="text-end"),
                        html.Th("R-phase %", className="text-end")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            phase.get('phase_name', 'Unknown')[:40] + ('...' if len(phase.get('phase_name', '')) > 40 else ''),
                            className="font-monospace small"
                        ),
                        html.Td(phase.get('spacegroup', 'N/A')),
                        html.Td(
                            # For single phase, show 100%; otherwise use weight_fraction
                            f"{100.0 if len(phases) == 1 else (phase.get('weight_fraction') or 0) * 100:.1f}%",
                            className="text-end"
                        ),
                        html.Td(
                            f"{phase.get('r_phase', 0):.2f}%",
                            className="text-end"
                        )
                    ]) for phase in phases
                ])
            ], bordered=True, hover=True, responsive=True, size="sm"),
            
            # Info alert
            dbc.Alert([
                html.Strong("About Phase Analysis: "),
                "Rietveld refinement identifies crystalline phases and their proportions. ",
                f"Rwp (weighted profile R-factor) measures fit quality - lower is better. ",
                f"Analysis performed with DARA/BGMN."
            ], color="info", className="mt-3 small")
        ])
    ], className="shadow-sm mb-4")


def create_cluster_comparison_chart(sem_data, sample_data):
    """Create bar chart comparing clusters to target composition"""
    if not sem_data:
        return None
    
    elements = ['Na', 'Y', 'Hf', 'Zr', 'Nb', 'In', 'Sn', 'Si', 'P', 'O']
    target_percentages = sample_data['target_composition']['percentages']
    
    fig = go.Figure()
    
    # Add target composition
    target_values = [target_percentages.get(el, 0) for el in elements]
    fig.add_trace(go.Bar(
        name='Target',
        x=elements,
        y=target_values,
        marker_color='rgba(59, 130, 246, 0.7)'
    ))
    
    # Add cluster compositions
    for i, cluster in enumerate(sem_data['clusters']):
        cluster_values = [cluster['composition'].get(el, 0) for el in elements]
        fig.add_trace(go.Bar(
            name=f'Cluster {cluster["id"]}',
            x=elements,
            y=cluster_values,
            marker_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ))
    
    fig.update_layout(
        barmode='group',
        xaxis_title='Element',
        yaxis_title='Atomic %',
        height=500,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,1)',
        margin=dict(t=30, b=50, l=50, r=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def create_element_distribution_chart(sem_data):
    """Create bar chart showing element distribution across clusters"""
    if not sem_data or not sem_data.get('clusters'):
        return None
    
    # Get all elements from all clusters
    all_elements = set()
    for cluster in sem_data['clusters']:
        all_elements.update(cluster['composition'].keys())
    
    elements = sorted(all_elements)
    
    fig = go.Figure()
    
    for i, cluster in enumerate(sem_data['clusters']):
        values = [cluster['composition'].get(el, 0) for el in elements]
        fig.add_trace(go.Bar(
            name=f'Cluster {cluster["id"]}',
            x=elements,
            y=values,
            marker_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ))
    
    fig.update_layout(
        barmode='group',
        xaxis_title='Element',
        yaxis_title='Atomic %',
        height=400,
        title='Element Distribution by Cluster',
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,1)',
        margin=dict(t=50, b=50, l=50, r=30)
    )
    
    return fig


def create_composition_heatmap(sem_data):
    """Create heatmap of composition across clusters"""
    if not sem_data or not sem_data.get('clusters'):
        return None
    
    # Get all elements
    all_elements = set()
    for cluster in sem_data['clusters']:
        all_elements.update(cluster['composition'].keys())
    
    elements = sorted(all_elements)
    
    # Build matrix
    matrix = []
    cluster_labels = []
    for cluster in sem_data['clusters']:
        cluster_labels.append(f"Cluster {cluster['id']}")
        row = [cluster['composition'].get(el, 0) for el in elements]
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=elements,
        y=cluster_labels,
        colorscale='Viridis',
        text=[[f'{val:.1f}%' for val in row] for row in matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Atomic %")
    ))
    
    fig.update_layout(
        title='Composition Heatmap',
        xaxis_title='Element',
        yaxis_title='Cluster',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,1)',
        margin=dict(t=50, b=50, l=100, r=30)
    )
    
    return fig


def create_target_vs_actual_chart(sem_data, sample_data):
    """Create chart comparing target vs actual composition"""
    if not sem_data or not sem_data.get('target_comparison'):
        return None
    
    comparisons = sem_data['target_comparison']
    elements = [c['Element'] for c in comparisons]
    target = [c['Target_at%'] for c in comparisons]
    actual = [c['Cluster_0_at%'] for c in comparisons]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Target',
        x=elements,
        y=target,
        marker_color='rgba(59, 130, 246, 0.7)'
    ))
    
    fig.add_trace(go.Bar(
        name='Measured (Cluster 0)',
        x=elements,
        y=actual,
        marker_color='rgba(239, 68, 68, 0.7)'
    ))
    
    fig.update_layout(
        barmode='group',
        title='Target vs Measured Composition (Cluster 0)',
        xaxis_title='Element',
        yaxis_title='Atomic %',
        height=400,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,1)',
        margin=dict(t=50, b=50, l=50, r=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def create_variability_chart(sem_data):
    """Create chart showing composition variability within clusters"""
    if not sem_data or not sem_data.get('clusters'):
        return None
    
    # Calculate standard deviations (simplified - would need raw spectra data for true std dev)
    # For now, show the number of points per cluster as a proxy
    cluster_ids = [f"Cluster {c['id']}" for c in sem_data['clusters']]
    n_points = [c['n_points'] for c in sem_data['clusters']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=cluster_ids,
        y=n_points,
        marker_color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(cluster_ids))],
        text=n_points,
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Number of Spectra per Cluster',
        xaxis_title='Cluster',
        yaxis_title='Number of Spectra',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,1)',
        margin=dict(t=50, b=50, l=50, r=30),
        showlegend=False
    )
    
    return fig


# Layout
app.layout = dbc.Container([
    # Password Modal (only shown if auth is enabled)
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Berkeley DOE A-Lab")),
        dbc.ModalBody([
            html.P("Enter password to access data", className="text-muted mb-3"),
            dbc.Input(id="password-input", type="password", placeholder="Enter password"),
            html.Div(id="auth-error", className="text-danger mt-2")
        ]),
        dbc.ModalFooter(
            dbc.Button("Access Lab Data", id="login-button", color="primary", className="w-100")
        )
    ], id="password-modal", is_open=(not NO_AUTH), backdrop="static", keyboard=False),
    
    # Hidden div to store auth state
    dcc.Store(id='auth-state', data={'authenticated': NO_AUTH}),
    
    # Main content (shown immediately if NO_AUTH=True)
    html.Div(id='main-content', style={'display': 'block' if NO_AUTH else 'none'}, children=[
        # Header with experiment selector
        dbc.Row([
            dbc.Col([
                html.H1("Berkeley DOE A-Lab", className="display-4 fw-bold text-dark"),
                html.P("Material Discovery & Analysis Platform", className="lead text-muted")
            ], md=6, className="py-4"),
            dbc.Col([
                html.Div([
                    html.Label("Select Experiment:", className="fw-semibold mb-2"),
                    dcc.Dropdown(
                        id='experiment-selector',
                        options=[
                            {'label': f"{exp['name']} - {exp.get('target_formula', 'N/A')}", 
                             'value': exp['name']}
                            for exp in experiment_list
                        ],
                        value=default_experiment,
                        clearable=False,
                        searchable=True,
                        placeholder="Search experiments..."
                    )
            ], className="py-4")
            ], md=6)
        ]),
        
        # Dynamic content container
        html.Div(id='experiment-content')
    ])
], fluid=True, className="py-4", style={'backgroundColor': '#f9fafb'})


def generate_experiment_content(sample_data, raw_data, sem_data):
    """Generate dashboard content for an experiment"""
    if not sample_data or not raw_data:
        return html.Div([
            dbc.Alert("No data available for this experiment", color="warning", className="mt-4")
        ])
    
    return html.Div([
        # Sample Information Card
        dbc.Card([
            dbc.CardBody([
                html.H2("Sample Information", className="h4 mb-4"),
                dbc.Row([
                    dbc.Col([
                        html.P("Sample Name", className="small text-muted mb-1"),
                        html.P(sample_data['sample_info']['name'], className="h5 fw-semibold")
                    ], md=6),
                    dbc.Col([
                        html.P("Sample ID", className="small text-muted mb-1"),
                        html.P(sample_data['sample_info']['id'], className="h6 font-monospace")
                    ], md=6),
                    dbc.Col([
                        html.P("Target Formula", className="small text-muted mb-1"),
                        html.P(sample_data['sample_info']['target_formula'], 
                               className="h5 font-monospace text-break")
                    ], md=12, className="mt-3")
                ])
            ])
        ], className="shadow-sm mb-4"),
        
        # Target Element Composition
        dbc.Card([
            dbc.CardBody([
                html.H2("Target Element Composition", className="h4 mb-4"),
                dbc.Row([
                    dbc.Col([
                        html.H3("Elemental Distribution", className="h6 text-center mb-3"),
                        dcc.Graph(figure=create_element_pie_chart(sample_data), 
                                 config={'displayModeBar': False})
                    ], lg=6),
                    dbc.Col([
                        html.H3("Atomic Composition", className="h6 mb-3"),
                        html.Div([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.Div(style={
                                                    'width': '16px',
                                                    'height': '16px',
                                                    'backgroundColor': ELEMENT_COLORS[idx % len(ELEMENT_COLORS)],
                                                    'borderRadius': '4px',
                                                    'display': 'inline-block',
                                                    'marginRight': '12px'
                                                }),
                                                html.Span(el, className="fw-semibold")
                                            ], style={'display': 'flex', 'alignItems': 'center'})
                                        ], width='auto'),
                                        dbc.Col([
                                            html.P(f"{amt:.3f} atoms", 
                                                  className="small font-monospace mb-0 text-end"),
                                            html.P(f"{sample_data['target_composition']['percentages'][el]:.2f}%", 
                                                  className="small text-muted mb-0 text-end")
                                        ])
                                    ])
                                ], className="py-2")
                            ], className="mb-2")
                            for idx, (el, amt) in enumerate(
                                sorted(sample_data['target_composition']['elements'].items(), 
                                      key=lambda x: x[1], reverse=True)
                            )
                        ], style={'maxHeight': '350px', 'overflowY': 'auto'})
                    ], lg=6)
                ])
            ])
        ], className="shadow-sm mb-4"),
        
        # Available Precursors
        dbc.Card([
            dbc.CardBody([
                html.H2("Available Precursors", className="h4 mb-4"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Powder"),
                            html.Th("Target (g)", className="text-end"),
                            html.Th("Actual (g)", className="text-end"),
                            html.Th("Doses", className="text-end"),
                            html.Th("Accuracy", className="text-end")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(p['name'], className="fw-semibold"),
                            html.Td(f"{p['target_mass']:.5f}", className="font-monospace text-end"),
                            html.Td(f"{p['actual_mass']:.5f}", className="font-monospace text-end"),
                            html.Td(str(p['num_doses']), className="text-end"),
                            html.Td(
                                f"{p['accuracy']:.2f}%",
                                className=f"fw-semibold text-end {'text-success' if 99 <= p['accuracy'] <= 105 else 'text-warning'}"
                            )
                        ]) for p in sample_data['precursors']
                    ] + [
                        html.Tr([
                            html.Td("Total", className="fw-bold"),
                            html.Td(f"{sample_data['powder_totals']['target_mass_g']:.5f}", 
                                   className="font-monospace text-end fw-bold"),
                            html.Td(f"{sample_data['powder_totals']['actual_mass_g']:.5f}", 
                                   className="font-monospace text-end fw-bold"),
                            html.Td(str(sum(p['num_doses'] for p in sample_data['precursors'])), 
                                   className="text-end fw-bold"),
                            html.Td(
                                f"{(sample_data['powder_totals']['actual_mass_g'] / sample_data['powder_totals']['target_mass_g'] * 100):.2f}%",
                                className="text-success text-end fw-bold"
                            )
                        ], className="border-top border-2")
                    ])
                ], bordered=True, hover=True, responsive=True)
            ])
        ], className="shadow-sm mb-4"),
        
         # Sample Image (currently using placeholder - would need image storage for dynamic images)
         dbc.Card([
             dbc.CardBody([
                 html.H2("📷 Sample Image", className="h4 mb-4"),
                 html.Div([
                     dbc.Alert([
                         html.I(className="bi bi-image me-2"),
                         f"SEM images for {sample_data['sample_info']['name']} would be displayed here.",
                         html.Br(),
                         html.Small("Note: Dynamic image loading requires additional image storage configuration.", 
                                   className="text-muted")
                     ], color="light", className="text-center")
                 ])
             ])
         ], className="shadow-sm mb-4") if False else None,  # Disabled for now
        
        # Heating Profile
        dbc.Card([
            dbc.CardBody([
                html.H2("🔥 Heating Profile", className="h4 mb-4"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.P("Target Temp", className="small text-muted mb-1"),
                                    html.H3(f"{raw_data['metadata']['heating_results'].get('heating_temperature', 'N/A')}°C",
                                           className="h4 mb-0 text-danger")
                                ])
                            ], className="bg-light border-0")
                        ], md=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.P("Duration", className="small text-muted mb-1"),
                                    html.H3(f"{raw_data['metadata']['heating_results'].get('heating_time', 0) / 60:.1f} hrs",
                                           className="h4 mb-0 text-primary")
                                ])
                            ], className="bg-light border-0")
                        ], md=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.P("Data Points", className="small text-muted mb-1"),
                                    html.H3(str(len(raw_data['metadata']['heating_results'].get('temperature_log', {}).get('time_minutes', []))),
                                           className="h4 mb-0 text-success")
                                ])
                            ], className="bg-light border-0")
                        ], md=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.P("Max Temp", className="small text-muted mb-1"),
                                    html.H3(f"{max([float(t) for t in raw_data['metadata']['heating_results'].get('temperature_log', {}).get('temperature_celsius', [0])]):.0f}°C",
                                           className="h4 mb-0 text-warning")
                                ])
                            ], className="bg-light border-0")
                        ], md=3)
                    ], className="mb-4"),
                    dcc.Graph(figure=create_temperature_chart(raw_data), 
                             config={'displayModeBar': False}) if create_temperature_chart(raw_data) else 
                    html.P("Temperature data not available", className="text-muted")
                ])
            ])
        ], className="shadow-sm mb-4"),
        
        # Powder Composition Chart
        dbc.Card([
            dbc.CardBody([
                html.H2("Powder Composition by Mass", className="h4 mb-4"),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=create_powder_pie_chart(sample_data), 
                                 config={'displayModeBar': False})
                    ], lg=6),
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.Div(style={
                                        'width': '12px',
                                        'height': '12px',
                                        'backgroundColor': POWDER_COLORS[idx % len(POWDER_COLORS)],
                                        'borderRadius': '4px',
                                        'display': 'inline-block',
                                        'marginRight': '8px'
                                    }),
                                    html.Span(p['name'], className="small")
                                ], style={'display': 'flex', 'alignItems': 'center'}, 
                                   className="mb-2")
                                for idx, p in enumerate(sample_data['precursors'])
                            ], className="row row-cols-2")
                        ], style={'display': 'flex', 'alignItems': 'center', 'height': '100%'})
                    ], lg=6)
                ])
            ])
        ], className="shadow-sm mb-4"),
        
        # XRD Characterization (always show if data available)
        dbc.Card([
            dbc.CardBody([
                html.H2("📐 XRD Characterization", className="h4 mb-4"),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.P("Sample ID", className="small text-muted mb-1"),
                                        html.P(raw_data['metadata']['diffraction_results']['sampleid_in_aeris'].split('_')[0],
                                              className="h6 font-monospace mb-0")
                                    ])
                                ], className="bg-light border-0")
                            ], md=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.P("Data Points", className="small text-muted mb-1"),
                                        html.P(str(len(raw_data['metadata']['diffraction_results']['twotheta'])),
                                              className="h6 mb-0")
                                    ])
                                ], className="bg-light border-0")
                            ], md=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.P("2θ Range", className="small text-muted mb-1"),
                                        html.P(f"{min(raw_data['metadata']['diffraction_results']['twotheta']):.1f}° - {max(raw_data['metadata']['diffraction_results']['twotheta']):.1f}°",
                                              className="h6 mb-0")
                                    ])
                                ], className="bg-light border-0")
                            ], md=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.P("Max Intensity", className="small text-muted mb-1"),
                                        html.P(f"{max([float(c) for c in raw_data['metadata']['diffraction_results']['counts']]):.0f}",
                                              className="h6 mb-0")
                                    ])
                                ], className="bg-light border-0")
                            ], md=3)
                        ], className="mb-4"),
                        dcc.Graph(figure=create_xrd_chart(raw_data),
                                 config={'displayModeBar': False}) if create_xrd_chart(raw_data) else None,
                        dbc.Alert([
                            html.Strong("About XRD: "),
                            "X-ray diffraction pattern shows the crystalline structure of the material. ",
                            "Peaks correspond to specific crystal planes and can be used to identify phases ",
                            "and calculate lattice parameters."
                        ], color="info", className="mt-3")
                    ]) if raw_data['metadata'].get('diffraction_results', {}).get('twotheta') else
                    html.P("XRD data not available for this sample", className="text-muted text-center py-4")
            ])
        ], className="shadow-sm mb-4"),
        
        # XRD Phase Analysis (DARA) - only shown if analysis results available
        create_xrd_analysis_section(raw_data) if raw_data.get('metadata', {}).get('xrd_analysis') else None,
        
        # # SEM-EDS Analysis Section (all subsections always visible)
        # # SEM Analysis Header
        # dbc.Card([
        #     dbc.CardBody([
        #         html.H2("🔬 SEM-EDS Cluster Analysis", className="h4 mb-3"),
        #         html.Div([
        #             html.P("Scanning Electron Microscopy with Energy Dispersive X-ray Spectroscopy",
        #                   className="text-muted mb-3"),
        #             dbc.Row([
        #                 dbc.Col([
        #                     html.Div([
        #                         html.P("Spectra Analyzed", className="small mb-1 text-muted"),
        #                         html.H3(f"{sem_data['clustering']['n_spectra_used']}/{sem_data['clustering']['n_spectra_collected']}",
        #                                className="h5 mb-0")
        #                     ], className="bg-light rounded p-3")
        #                 ], md=3),
        #                 dbc.Col([
        #                     html.Div([
        #                         html.P("Clusters Found", className="small mb-1 text-muted"),
        #                         html.H3(str(sem_data['clustering']['n_clusters']),
        #                                className="h5 mb-0")
        #                     ], className="bg-light rounded p-3")
        #                 ], md=3),
        #                 dbc.Col([
        #                     html.Div([
        #                         html.P("Silhouette Score", className="small mb-1 text-muted"),
        #                         html.H3(f"{sem_data['clustering']['silhouette_score']:.3f}",
        #                                className="h5 mb-0")
        #                     ], className="bg-light rounded p-3")
        #                 ], md=3),
        #                 dbc.Col([
        #                     html.Div([
        #                         html.P("Method", className="small mb-1 text-muted"),
        #                         html.H3(sem_data['clustering']['method'].upper(),
        #                                className="h5 mb-0")
        #                     ], className="bg-light rounded p-3")
        #                 ], md=3)
        #             ])
        #         ]) if sem_data else create_sem_not_available_alert()
        #     ])
        # ], className="shadow-sm mb-4"),
        
        # # Cluster Comparison Chart
        # dbc.Card([
        #     dbc.CardBody([
        #         html.H2("Element Distribution by Cluster", className="h4 mb-4"),
        #         html.Div([
        #             dcc.Graph(figure=create_cluster_comparison_chart(sem_data, sample_data),
        #                      config={'displayModeBar': False})
        #         ]) if sem_data else create_sem_not_available_alert()
        #     ])
        # ], className="shadow-sm mb-4"),
        
        # # Target vs Actual Comparison
        # dbc.Card([
        #     dbc.CardBody([
        #         html.H2("Target vs Measured Composition (Cluster 0)", className="h4 mb-4"),
        #         html.Div([
        #             dbc.Table([
        #                 html.Thead([
        #                     html.Tr([
        #                         html.Th("Element"),
        #                         html.Th("Target (at%)", className="text-end"),
        #                         html.Th("Measured (at%)", className="text-end"),
        #                         html.Th("Difference", className="text-end"),
        #                         html.Th("Rel. Diff", className="text-end")
        #                     ])
        #                 ]),
        #                 html.Tbody([
        #                     html.Tr([
        #                         html.Td(comp['Element'], className="fw-semibold"),
        #                         html.Td(f"{comp['Target_at%']:.2f}", className="font-monospace text-end"),
        #                         html.Td(f"{comp['Cluster_0_at%']:.2f}", className="font-monospace text-end"),
        #                         html.Td(
        #                             f"{'+' if comp['Difference'] > 0 else ''}{comp['Difference']:.2f}",
        #                             className="font-monospace text-end"
        #                         ),
        #                         html.Td(
        #                             f"{'+' if comp['Rel_Diff_%'] > 0 else ''}{comp['Rel_Diff_%']:.1f}%",
        #                             className=f"fw-semibold text-end {'text-success' if abs(comp['Rel_Diff_%']) < 10 else 'text-warning' if abs(comp['Rel_Diff_%']) < 25 else 'text-danger'}"
        #                         )
        #                     ])
        #                     for comp in sem_data.get('target_comparison', [])
        #                 ])
        #             ], bordered=True, hover=True, responsive=True),
        #             dbc.Alert([
        #                 html.Strong("Note: "),
        #                 f"Cluster 0 represents the primary NASICON phase with {sem_data['clusters'][0]['n_points']} measurements. ",
        #                 "Green values indicate good agreement (±10%), orange indicates acceptable (±25%)."
        #             ], color="info", className="mt-3")
        #         ]) if sem_data and sem_data.get('target_comparison') else create_sem_not_available_alert()
        #     ])
        # ], className="shadow-sm mb-4"),
        
        # # SEM Composition Heatmap
        # dbc.Card([
        #     dbc.CardBody([
        #         html.H2("Composition Heatmap", className="h4 mb-4"),
        #         html.Div([
        #             dcc.Graph(figure=create_composition_heatmap(sem_data),
        #                      config={'displayModeBar': False}),
        #             html.P("Heatmap visualization of elemental composition for each cluster. Warmer colors indicate higher concentrations.",
        #                   className="small text-muted mt-2 text-center")
        #         ]) if sem_data else create_sem_not_available_alert()
        #     ])
        # ], className="shadow-sm mb-4"),
        
        # # SEM Spectra Distribution
        # dbc.Card([
        #     dbc.CardBody([
        #         html.H2("Spectra Distribution", className="h4 mb-4"),
        #         html.Div([
        #             dcc.Graph(figure=create_variability_chart(sem_data),
        #                      config={'displayModeBar': False}),
        #             html.P("Distribution of spectra across clusters, showing measurement coverage.",
        #                   className="small text-muted mt-2 text-center")
        #         ]) if sem_data else create_sem_not_available_alert()
        #     ])
        # ], className="shadow-sm mb-4"),
        
        # Footer message
        html.Div([
            dbc.Card([
                dbc.CardBody([
                    dbc.Alert([
                        html.Strong("Data Pipeline: "),
                        f"Loaded from Parquet files • {sample_data['sample_info']['name']} • ",
                        f"Last updated: {sample_data['sample_info'].get('last_updated', 'N/A')}"
                    ], color="light", className="mb-0 small text-center")
                ])
            ], className="border-0")
        ], className="mb-4")
    ])


# Experiment selection callback
@app.callback(
    Output('experiment-content', 'children'),
    [Input('experiment-selector', 'value')]
)
def update_experiment_content(experiment_name):
    """Load and display data for selected experiment"""
    if not experiment_name:
        return html.Div([
            dbc.Alert("Please select an experiment", color="info", className="mt-4")
        ])
    
    # Load experiment data
    sample_data, raw_data, sem_data = load_experiment_data(experiment_name)
    
    # Generate content
    return generate_experiment_content(sample_data, raw_data, sem_data)


# Authentication callback
@app.callback(
    [Output('password-modal', 'is_open'),
     Output('main-content', 'style'),
     Output('auth-error', 'children'),
     Output('auth-state', 'data')],
    [Input('login-button', 'n_clicks')],
    [State('password-input', 'value'),
     State('auth-state', 'data')]
)
def authenticate(n_clicks, password, auth_state):
    # If NO_AUTH mode is enabled, bypass authentication
    if NO_AUTH:
        return False, {'display': 'block'}, '', {'authenticated': True}
    
    if n_clicks is None:
        return True, {'display': 'none'}, '', auth_state
    
    # Simple password check (in production, use proper authentication)
    if password == "alab":  # Replace with your actual password
        return False, {'display': 'block'}, '', {'authenticated': True}
    else:
        return True, {'display': 'none'}, 'Incorrect password', auth_state


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

