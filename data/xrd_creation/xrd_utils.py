#!/usr/bin/env python3
"""
XRD Analysis Utilities - Extensible primitives for DARA integration

This module provides reusable functions for:
- Exporting XRD patterns from Parquet to XY format
- Running DARA refinement/phase search
- Extracting and structuring results
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

# Constants for CIF cache management
MAX_CACHE_FAILURES = 3  # Re-download if fewer than this many failures

# Paths
PARQUET_DIR = Path(__file__).parent.parent / "parquet"
RESULTS_DIR = Path(__file__).parent / "results"
CIF_CACHE_DIR = Path(__file__).parent / "cif_cache"

# DARA Configuration
INSTRUMENT_PROFILE = "Aeris-fds-Pixcel1d-Medipix3"
WAVELENGTH = "Cu"
DEFAULT_WMIN = 10
DEFAULT_WMAX = 80


@dataclass
class PhaseResult:
    """Single phase from refinement"""
    phase_name: str
    spacegroup: Optional[str] = None
    weight_fraction: Optional[float] = None
    weight_fraction_error: Optional[float] = None
    lattice_a_nm: Optional[float] = None
    lattice_b_nm: Optional[float] = None
    lattice_c_nm: Optional[float] = None
    lattice_alpha_deg: Optional[float] = None
    lattice_beta_deg: Optional[float] = None
    lattice_gamma_deg: Optional[float] = None
    r_phase: Optional[float] = None


@dataclass
class RefinementResult:
    """Complete refinement result for one experiment"""
    experiment_id: str
    experiment_name: str
    success: bool
    error: Optional[str] = None
    rwp: Optional[float] = None
    rp: Optional[float] = None
    rexp: Optional[float] = None
    num_phases: int = 0
    phases: List[PhaseResult] = None
    chemical_system: Optional[str] = None
    target_formula: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        return d


def load_experiment_data(experiment_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load XRD pattern and metadata for a single experiment
    
    Returns:
        Tuple of (pattern_df, metadata_dict)
    """
    experiments_df = pd.read_parquet(PARQUET_DIR / "experiments.parquet")
    elements_df = pd.read_parquet(PARQUET_DIR / "experiment_elements.parquet")
    xrd_points_df = pd.read_parquet(PARQUET_DIR / "xrd_data_points.parquet")
    
    # Find experiment
    exp_row = experiments_df[experiments_df['name'] == experiment_name]
    if exp_row.empty:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    exp = exp_row.iloc[0]
    exp_id = exp['experiment_id']
    
    # Get XRD pattern
    pattern = xrd_points_df[xrd_points_df['experiment_id'] == exp_id]
    pattern = pattern.sort_values('point_index')[['twotheta', 'counts']]
    
    if pattern.empty:
        raise ValueError(f"No XRD data for experiment '{experiment_name}'")
    
    # Get elements for chemical system
    elements = elements_df[elements_df['experiment_id'] == exp_id]['element_symbol'].tolist()
    chemsys = '-'.join(sorted(set(elements))) if elements else None
    
    metadata = {
        'experiment_id': exp_id,
        'experiment_name': experiment_name,
        'target_formula': exp.get('target_formula', ''),
        'experiment_type': exp.get('experiment_type', ''),
        'chemical_system': chemsys,
        'elements': elements,
        'num_points': len(pattern)
    }
    
    return pattern, metadata


def export_pattern_to_xy(pattern_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Export XRD pattern DataFrame to XY format (two-theta, intensity)
    
    Args:
        pattern_df: DataFrame with 'twotheta' and 'counts' columns
        output_path: Path to write XY file
    
    Returns:
        Path to written file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for _, row in pattern_df.iterrows():
            f.write(f"{row['twotheta']:.6f} {row['counts']:.2f}\n")
    
    logger.info(f"Exported {len(pattern_df)} points to {output_path}")
    return output_path


def _get_cache_meta_path(cache_dir: Path) -> Path:
    """Get path to cache metadata file"""
    return cache_dir / "_cache_meta.json"


def _read_cache_meta(cache_dir: Path) -> Dict[str, Any]:
    """Read cache metadata, return empty dict if not exists"""
    meta_path = _get_cache_meta_path(cache_dir)
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def _write_cache_meta(cache_dir: Path, meta: Dict[str, Any]):
    """Write cache metadata"""
    meta_path = _get_cache_meta_path(cache_dir)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def _is_cache_complete(cache_dir: Path) -> bool:
    """
    Check if cache is considered complete.
    Returns True if:
    - Cache meta shows 'complete' status, OR
    - We've failed MAX_CACHE_FAILURES times already
    """
    meta = _read_cache_meta(cache_dir)
    if meta.get('complete', False):
        return True
    if meta.get('download_attempts', 0) >= MAX_CACHE_FAILURES:
        return True
    return False


def get_reference_cifs(chemical_system: str, dest_dir: Path = None, force_redownload: bool = False) -> List[Path]:
    """
    Download reference CIF files for a chemical system from COD
    
    Args:
        chemical_system: e.g., "Na-P-Si-Zr"
        dest_dir: Directory to save CIFs (default: cif_cache/<chemsys>)
        force_redownload: Force re-download even if cached
    
    Returns:
        List of CIF file paths
    """
    from dara.structure_db import CODDatabase
    
    if dest_dir is None:
        dest_dir = CIF_CACHE_DIR / chemical_system.replace('-', '_')
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    existing_cifs = list(dest_dir.glob("*.cif"))
    meta = _read_cache_meta(dest_dir)
    
    # Check if we should use cache
    if not force_redownload and existing_cifs and _is_cache_complete(dest_dir):
        logger.info(f"Using {len(existing_cifs)} cached CIFs for {chemical_system}")
        if meta.get('failures'):
            logger.info(f"  (Note: {len(meta['failures'])} CIFs failed to download previously)")
        return existing_cifs
    
    # Download from COD
    logger.info(f"Downloading CIFs for {chemical_system} from COD...")
    cod_db = CODDatabase()
    
    # Track attempt
    meta['download_attempts'] = meta.get('download_attempts', 0) + 1
    meta['last_attempt'] = datetime.now().isoformat()
    
    try:
        # Clear existing CIFs if force_redownload
        if force_redownload:
            for cif in existing_cifs:
                cif.unlink()
        
        cod_db.get_cifs_by_chemsys(chemical_system, dest_dir=str(dest_dir))
        cif_files = list(dest_dir.glob("*.cif"))
        
        # Update metadata
        meta['success_count'] = len(cif_files)
        meta['complete'] = True  # Mark as complete (no failures detected at this level)
        meta['last_success'] = datetime.now().isoformat()
        _write_cache_meta(dest_dir, meta)
        
        logger.info(f"Downloaded {len(cif_files)} CIF files")
        return cif_files
        
    except Exception as e:
        logger.error(f"Failed to download CIFs: {e}")
        
        # Record failure
        if 'failures' not in meta:
            meta['failures'] = []
        meta['failures'].append({
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        })
        _write_cache_meta(dest_dir, meta)
        
        # Return any existing CIFs we might have
        existing = list(dest_dir.glob("*.cif"))
        if existing:
            logger.info(f"Using {len(existing)} partially cached CIFs")
            return existing
        return []


def run_phase_search(
    pattern_path: Path,
    cif_files: List[Path],
    wmin: float = DEFAULT_WMIN,
    wmax: float = DEFAULT_WMAX
) -> Dict[str, Any]:
    """
    Run DARA phase search on a pattern
    
    Args:
        pattern_path: Path to XY pattern file
        cif_files: List of reference CIF files
        wmin: Minimum two-theta angle
        wmax: Maximum two-theta angle
    
    Returns:
        Dictionary with search results
    """
    from dara import search_phases
    
    logger.info(f"Running phase search with {len(cif_files)} reference phases...")
    logger.info(f"  Pattern: {pattern_path}")
    logger.info(f"  2θ range: {wmin}° - {wmax}°")
    
    search_results = search_phases(
        pattern_path=str(pattern_path),
        phases=cif_files,
        wavelength=WAVELENGTH,
        instrument_profile=INSTRUMENT_PROFILE,
        refinement_params={
            "wmin": wmin,
            "wmax": wmax
        }
    )
    
    if not search_results:
        return {'success': False, 'error': 'No phases found'}
    
    return extract_search_result(search_results[0])


def run_refinement(
    pattern_path: Path,
    cif_files: List[Path],
    wmin: float = DEFAULT_WMIN,
    wmax: float = DEFAULT_WMAX
) -> Dict[str, Any]:
    """
    Run DARA refinement with known phases (faster than search)
    
    Args:
        pattern_path: Path to XY pattern file
        cif_files: List of CIF files to refine against
        wmin: Minimum two-theta angle
        wmax: Maximum two-theta angle
    
    Returns:
        Dictionary with refinement results
    """
    from dara.refine import do_refinement_no_saving
    
    logger.info(f"Running refinement with {len(cif_files)} phases...")
    
    refinement = do_refinement_no_saving(
        str(pattern_path),
        [str(c) for c in cif_files],
        wavelength=WAVELENGTH,
        instrument_profile=INSTRUMENT_PROFILE,
        phase_params={
            "lattice_range": 0.05,
            "b1": "0_0^0.005",
            "k1": "0_0^1",
            "k2": "fixed",
            "gewicht": "SPHAR2"
        },
        refinement_params={
            "wmin": wmin,
            "wmax": wmax
        }
    )
    
    return extract_refinement_result(refinement)


def extract_search_result(search_result) -> Dict[str, Any]:
    """Extract structured data from DARA SearchResult"""
    refinement = search_result.refinement_result
    return extract_refinement_result(refinement)


def _safe_get_value(val, index=0):
    """Safely extract value from DARA result (handles float or tuple)"""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return val[index] if len(val) > index else None
    # It's a plain float/int
    return val if index == 0 else None


def extract_refinement_result(refinement) -> Dict[str, Any]:
    """Extract structured data from DARA RefinementResult"""
    lst_data = refinement.lst_data
    
    phases = []
    for phase_name, pr in lst_data.phases_results.items():
        phases.append(PhaseResult(
            phase_name=phase_name,
            spacegroup=pr.hermann_mauguin,
            weight_fraction=_safe_get_value(pr.gewicht, 0),
            weight_fraction_error=_safe_get_value(pr.gewicht, 1),
            lattice_a_nm=_safe_get_value(pr.a, 0),
            lattice_b_nm=_safe_get_value(pr.b, 0),
            lattice_c_nm=_safe_get_value(pr.c, 0),
            lattice_alpha_deg=_safe_get_value(pr.alpha, 0),
            lattice_beta_deg=_safe_get_value(pr.beta, 0),
            lattice_gamma_deg=_safe_get_value(pr.gamma, 0),
            r_phase=pr.rphase
        ))
    
    return {
        'success': True,
        'rwp': lst_data.rwp,
        'rp': lst_data.rp,
        'rexp': lst_data.rexp,
        'num_phases': len(phases),
        'phases': [asdict(p) for p in phases]
    }


def save_visualization(refinement, output_path: Path) -> Path:
    """Save interactive HTML visualization"""
    fig = refinement.visualize()
    fig.write_html(str(output_path))
    logger.info(f"Saved visualization to {output_path}")
    return output_path


def list_available_experiments() -> List[str]:
    """List all experiment names in the Parquet data"""
    experiments_df = pd.read_parquet(PARQUET_DIR / "experiments.parquet")
    return sorted(experiments_df['name'].tolist())

