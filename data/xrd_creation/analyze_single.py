#!/usr/bin/env python3
"""
Analyze a single XRD experiment using DARA

Usage:
    python analyze_single.py NSC_249
    python analyze_single.py NSC_249 --mode refinement
    python analyze_single.py --list  # List available experiments
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import utilities
from xrd_utils import (
    load_experiment_data,
    export_pattern_to_xy,
    get_reference_cifs,
    run_phase_search,
    run_refinement,
    RefinementResult,
    RESULTS_DIR,
    list_available_experiments
)


def _save_result(result: RefinementResult, mode: str, wmin: float, wmax: float) -> Path:
    """Save result to JSON file. Returns path to saved file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_json = RESULTS_DIR / f"{result.experiment_name}_result.json"
    with open(result_json, 'w') as f:
        json.dump({
            **result.to_dict(),
            'analysis_timestamp': datetime.now().isoformat(),
            'mode': mode,
            'wmin': wmin,
            'wmax': wmax
        }, f, indent=2)
    return result_json


def analyze_experiment(
    experiment_name: str,
    mode: str = "phase_search",
    wmin: float = 10,
    wmax: float = 80,
    save_viz: bool = True
) -> RefinementResult:
    """
    Run full DARA analysis on a single experiment
    
    Args:
        experiment_name: e.g., "NSC_249"
        mode: "phase_search" (slower, finds phases) or "refinement" (faster, needs CIFs)
        wmin: Minimum two-theta angle
        wmax: Maximum two-theta angle
        save_viz: Save interactive HTML plot
    
    Returns:
        RefinementResult dataclass
    """
    # Create a prefix for all log messages for this experiment
    exp = experiment_name  # Short alias for log prefix
    
    logger.info("=" * 60)
    logger.info(f"[{exp}] DARA XRD Analysis")
    logger.info(f"[{exp}] Mode: {mode}")
    logger.info("=" * 60)
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load experiment data
    logger.info(f"\n[{exp}] [1/4] Loading experiment data from Parquet...")
    try:
        pattern_df, metadata = load_experiment_data(experiment_name)
        logger.info(f"[{exp}]   ✓ Loaded {metadata['num_points']} XRD data points")
        logger.info(f"[{exp}]   ✓ Target formula: {metadata['target_formula']}")
        logger.info(f"[{exp}]   ✓ Chemical system: {metadata['chemical_system']}")
        logger.info(f"[{exp}]   ✓ Elements: {metadata['elements']}")
    except Exception as e:
        logger.error(f"[{exp}]   ✗ Failed to load experiment: {e}")
        result = RefinementResult(
            experiment_id="",
            experiment_name=experiment_name,
            success=False,
            error=str(e)
        )
        result_path = _save_result(result, mode, wmin, wmax)
        logger.info(f"[{exp}] 📄 Failure saved to: {result_path}")
        return result
    
    # Step 2: Export to XY format
    logger.info(f"\n[{exp}] [2/4] Exporting XRD pattern to XY format...")
    xy_path = RESULTS_DIR / f"{experiment_name}.xy"
    export_pattern_to_xy(pattern_df, xy_path)
    logger.info(f"[{exp}]   ✓ Saved to {xy_path}")
    
    # Step 3: Get reference CIFs
    logger.info(f"\n[{exp}] [3/4] Getting reference CIF structures...")
    if not metadata['chemical_system']:
        logger.error(f"[{exp}]   ✗ No chemical system defined for this experiment")
        result = RefinementResult(
            experiment_id=metadata['experiment_id'],
            experiment_name=experiment_name,
            success=False,
            error="No chemical system defined",
            chemical_system=None,
            target_formula=metadata.get('target_formula')
        )
        result_path = _save_result(result, mode, wmin, wmax)
        logger.info(f"[{exp}] 📄 Failure saved to: {result_path}")
        return result
    
    cif_files = get_reference_cifs(metadata['chemical_system'])
    if not cif_files:
        logger.error(f"[{exp}]   ✗ No reference CIFs found")
        result = RefinementResult(
            experiment_id=metadata['experiment_id'],
            experiment_name=experiment_name,
            success=False,
            error="No reference CIFs found",
            chemical_system=metadata['chemical_system'],
            target_formula=metadata.get('target_formula')
        )
        result_path = _save_result(result, mode, wmin, wmax)
        logger.info(f"[{exp}] 📄 Failure saved to: {result_path}")
        return result
    logger.info(f"[{exp}]   ✓ Found {len(cif_files)} reference structures")
    
    # Step 4: Run DARA analysis
    logger.info(f"\n[{exp}] [4/4] Running DARA {mode}...")
    logger.info(f"[{exp}]   This may take 1-5 minutes...")
    
    try:
        if mode == "phase_search":
            result_dict = run_phase_search(xy_path, cif_files, wmin, wmax)
        else:
            result_dict = run_refinement(xy_path, cif_files[:10], wmin, wmax)  # Top 10 CIFs
    except Exception as e:
        logger.error(f"[{exp}]   ✗ DARA analysis failed: {e}")
        result = RefinementResult(
            experiment_id=metadata['experiment_id'],
            experiment_name=experiment_name,
            success=False,
            error=str(e),
            chemical_system=metadata['chemical_system'],
            target_formula=metadata.get('target_formula')
        )
        result_path = _save_result(result, mode, wmin, wmax)
        logger.info(f"[{exp}] 📄 Failure saved to: {result_path}")
        return result
    
    # Build result
    result = RefinementResult(
        experiment_id=metadata['experiment_id'],
        experiment_name=experiment_name,
        success=result_dict.get('success', False),
        error=result_dict.get('error'),
        rwp=result_dict.get('rwp'),
        rp=result_dict.get('rp'),
        rexp=result_dict.get('rexp'),
        num_phases=result_dict.get('num_phases', 0),
        phases=result_dict.get('phases', []),
        chemical_system=metadata['chemical_system'],
        target_formula=metadata['target_formula']
    )
    
    # Log results
    if result.success:
        logger.info("\n" + "=" * 60)
        logger.info(f"[{exp}] ✓ ANALYSIS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"[{exp}]   Rwp: {result.rwp:.2f}%")
        logger.info(f"[{exp}]   Rp:  {result.rp:.2f}%")
        logger.info(f"[{exp}]   Phases identified: {result.num_phases}")
        
        for i, phase in enumerate(result.phases or [], 1):
            wt = phase.get('weight_fraction', 0) or 0
            logger.info(f"[{exp}]     {i}. {phase['phase_name']}")
            logger.info(f"[{exp}]        Spacegroup: {phase.get('spacegroup', 'N/A')}")
            logger.info(f"[{exp}]        Weight fraction: {wt*100:.1f}%")
    else:
        logger.error(f"\n[{exp}] ✗ Analysis failed: {result.error}")
    
    # Save results (success or failure)
    result_path = _save_result(result, mode, wmin, wmax)
    logger.info(f"\n[{exp}] 📄 Results saved to: {result_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a single XRD experiment using DARA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_single.py NSC_249              # Phase search (slower, thorough)
  python analyze_single.py NSC_249 --mode refinement  # Refinement only (faster)
  python analyze_single.py --list               # List available experiments
        """
    )
    parser.add_argument(
        'experiment',
        nargs='?',
        help='Experiment name (e.g., NSC_249)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['phase_search', 'refinement'],
        default='phase_search',
        help='Analysis mode (default: phase_search)'
    )
    parser.add_argument(
        '--wmin',
        type=float,
        default=10,
        help='Minimum two-theta angle (default: 10)'
    )
    parser.add_argument(
        '--wmax',
        type=float,
        default=80,
        help='Maximum two-theta angle (default: 80)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available experiments'
    )
    
    args = parser.parse_args()
    
    if args.list:
        experiments = list_available_experiments()
        print(f"\nAvailable experiments ({len(experiments)} total):\n")
        for exp in experiments[:20]:
            print(f"  {exp}")
        if len(experiments) > 20:
            print(f"  ... and {len(experiments) - 20} more")
        print(f"\nUsage: python analyze_single.py <experiment_name>")
        return
    
    if not args.experiment:
        parser.print_help()
        sys.exit(1)
    
    result = analyze_experiment(
        experiment_name=args.experiment,
        mode=args.mode,
        wmin=args.wmin,
        wmax=args.wmax
    )
    
    sys.exit(0 if result.success else 1)


if __name__ == '__main__':
    main()

