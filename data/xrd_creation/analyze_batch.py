#!/usr/bin/env python3
"""
Batch XRD Analysis - Process multiple experiments with DARA

Usage:
    python analyze_batch.py                    # Incremental (skip existing)
    python analyze_batch.py --all              # Rerun all experiments
    python analyze_batch.py --experiment NSC_249  # Single experiment
    python analyze_batch.py --workers 4        # Parallel processing
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Paths
RESULTS_DIR = Path(__file__).parent / "results"
PARQUET_DIR = Path(__file__).parent.parent / "parquet"


def get_experiments_to_analyze(rerun_all: bool = False, 
                                single_experiment: Optional[str] = None,
                                limit: Optional[int] = None) -> List[str]:
    """
    Get list of experiments that need XRD analysis.
    
    Args:
        rerun_all: If True, return all experiments (ignore existing results)
        single_experiment: If provided, return only this experiment
        limit: If provided, limit to first N experiments
    
    Returns:
        List of experiment names to analyze
    """
    # Load all experiments
    experiments_df = pd.read_parquet(PARQUET_DIR / "experiments.parquet")
    all_experiments = set(experiments_df['name'].tolist())
    
    # Filter to only completed experiments
    completed = experiments_df[experiments_df['status'] == 'completed']['name'].tolist()
    logger.info(f"Total experiments: {len(all_experiments)}, Completed: {len(completed)}")
    
    if single_experiment:
        if single_experiment not in all_experiments:
            raise ValueError(f"Experiment '{single_experiment}' not found")
        return [single_experiment]
    
    if rerun_all:
        experiments = completed
    else:
        # Incremental: skip experiments with existing results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        existing_results = {p.stem.replace('_result', '') for p in RESULTS_DIR.glob('*_result.json')}
        
        experiments = [exp for exp in completed if exp not in existing_results]
        logger.info(f"Already analyzed: {len(existing_results)}, To analyze: {len(experiments)}")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        experiments = experiments[:limit]
        logger.info(f"Limited to first {limit} experiments")
    
    return experiments


def analyze_single_experiment(experiment_name: str) -> dict:
    """
    Analyze a single experiment (worker function for parallel processing).
    
    Returns:
        Dict with experiment_name, success, and result data
    """
    # Import here to avoid issues with multiprocessing
    from analyze_single import analyze_experiment
    
    try:
        result = analyze_experiment(
            experiment_name=experiment_name,
            mode="phase_search",
            wmin=10,
            wmax=80,
            save_viz=False
        )
        return {
            'experiment_name': experiment_name,
            'success': result.success,
            'rwp': result.rwp,
            'num_phases': result.num_phases,
            'error': result.error
        }
    except Exception as e:
        logger.error(f"Failed to analyze {experiment_name}: {e}")
        return {
            'experiment_name': experiment_name,
            'success': False,
            'rwp': None,
            'num_phases': 0,
            'error': str(e)
        }


def _categorize_error(error: str) -> str:
    """Categorize error message into error type"""
    if not error:
        return None
    error_lower = error.lower()
    if 'no peaks' in error_lower:
        return 'no_peaks_detected'
    if 'cif' in error_lower or 'download' in error_lower:
        return 'cif_download'
    if 'timeout' in error_lower:
        return 'timeout'
    if 'chemical system' in error_lower:
        return 'no_chemical_system'
    if 'not found' in error_lower:
        return 'experiment_not_found'
    if 'subscriptable' in error_lower or 'attribute' in error_lower:
        return 'dara_internal_error'
    return 'analysis_error'


def export_results_to_parquet():
    """
    Export all JSON results to Parquet files for dashboard use.
    Creates two tables:
    - xrd_refinements: All analysis results (success AND failures)
    - xrd_phases: Individual phases from successful analyses only
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_files = list(RESULTS_DIR.glob('*_result.json'))
    
    if not result_files:
        logger.warning("No result files found to export")
        return
    
    refinements = []
    phases = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            exp_id = result.get('experiment_id', '')
            exp_name = result.get('experiment_name', '')
            success = result.get('success', False)
            error = result.get('error')
            error_type = _categorize_error(error) if error else None
            
            # Refinement summary (includes ALL results - success and failure)
            refinements.append({
                'experiment_id': exp_id,
                'experiment_name': exp_name,
                'success': success,
                'error': error,
                'error_type': error_type,
                'rwp': result.get('rwp'),
                'rp': result.get('rp'),
                'rexp': result.get('rexp'),
                'num_phases': result.get('num_phases', 0),
                'chemical_system': result.get('chemical_system'),
                'target_formula': result.get('target_formula'),
                'analysis_timestamp': result.get('analysis_timestamp'),
                'mode': result.get('mode'),
                'wmin': result.get('wmin'),
                'wmax': result.get('wmax')
            })
            
            # Skip phase extraction for failed analyses
            if not success:
                continue
            
            # Individual phases (only for successful analyses)
            for phase in result.get('phases', []):
                phases.append({
                    'experiment_id': exp_id,
                    'experiment_name': exp_name,
                    'phase_name': phase.get('phase_name'),
                    'spacegroup': phase.get('spacegroup'),
                    'weight_fraction': phase.get('weight_fraction'),
                    'weight_fraction_error': phase.get('weight_fraction_error'),
                    'lattice_a_nm': phase.get('lattice_a_nm'),
                    'lattice_b_nm': phase.get('lattice_b_nm'),
                    'lattice_c_nm': phase.get('lattice_c_nm'),
                    'r_phase': phase.get('r_phase')
                })
                
        except Exception as e:
            logger.error(f"Error processing {result_file}: {e}")
            continue
    
    # Export to Parquet
    output_dir = PARQUET_DIR
    
    if refinements:
        refinements_df = pd.DataFrame(refinements)
        refinements_df.to_parquet(output_dir / 'xrd_refinements.parquet', index=False)
        success_count = refinements_df['success'].sum()
        fail_count = len(refinements_df) - success_count
        logger.info(f"✓ Exported {len(refinements_df)} refinements to xrd_refinements.parquet ({success_count} success, {fail_count} failed)")
        
        # Log failure breakdown by type (from refinements where success=False)
        if fail_count > 0:
            failed_df = refinements_df[refinements_df['success'] == False]
            type_counts = failed_df['error_type'].value_counts().to_dict()
            logger.info(f"  Failure breakdown:")
            for err_type, count in type_counts.items():
                logger.info(f"    - {err_type}: {count}")
    
    if phases:
        phases_df = pd.DataFrame(phases)
        phases_df.to_parquet(output_dir / 'xrd_phases.parquet', index=False)
        logger.info(f"✓ Exported {len(phases_df)} phases to xrd_phases.parquet")


def run_batch_analysis(experiments: List[str], workers: int = 1) -> dict:
    """
    Run batch analysis on multiple experiments.
    
    Args:
        experiments: List of experiment names to analyze
        workers: Number of parallel workers (1 = sequential)
    
    Returns:
        Summary statistics
    """
    results = []
    
    if workers == 1:
        # Sequential processing (easier debugging)
        for exp_name in tqdm(experiments, desc="Analyzing experiments"):
            result = analyze_single_experiment(exp_name)
            results.append(result)
    else:
        # Parallel processing
        # Note: Each DARA analysis uses Ray internally, so don't use too many workers
        logger.info(f"Running with {workers} parallel workers")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(analyze_single_experiment, exp): exp 
                for exp in experiments
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc="Analyzing experiments"):
                try:
                    result = future.result(timeout=600)  # 10 min timeout per experiment
                    results.append(result)
                except Exception as e:
                    exp_name = futures[future]
                    logger.error(f"Worker failed for {exp_name}: {e}")
                    results.append({
                        'experiment_name': exp_name,
                        'success': False,
                        'error': str(e)
                    })
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    successful_names = [r['experiment_name'] for r in results if r.get('success', False)]
    
    summary = {
        'total': len(results),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(results) * 100 if results else 0,
        'successful_experiments': successful_names  # NEW
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Batch XRD analysis using DARA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_batch.py                      # Incremental (skip existing)
  python analyze_batch.py --limit 10           # First 10 experiments only
  python analyze_batch.py --all                # Rerun all experiments
  python analyze_batch.py --all --limit 20     # Rerun first 20 experiments
  python analyze_batch.py --experiment NSC_249 # Single experiment
  python analyze_batch.py --workers 4          # Use 4 parallel workers
  python analyze_batch.py --export-only        # Just export existing results to Parquet
        """
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Rerun analysis on all experiments (ignore existing results)'
    )
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        help='Analyze a single experiment'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1, sequential)'
    )
    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Only export existing results to Parquet (no new analysis)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit to first N experiments (useful for testing)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be analyzed without running'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("DARA XRD Batch Analysis")
    logger.info("=" * 60)
    
    if args.export_only:
        logger.info("Export-only mode: exporting existing results to Parquet")
        export_results_to_parquet()
        return
    
    # Get experiments to analyze
    experiments = get_experiments_to_analyze(
        rerun_all=args.all,
        single_experiment=args.experiment,
        limit=args.limit
    )
    
    if not experiments:
        logger.info("No experiments to analyze")
        export_results_to_parquet()
        return
    
    if args.dry_run:
        logger.info(f"Dry run: would analyze {len(experiments)} experiments:")
        for exp in experiments[:10]:
            logger.info(f"  - {exp}")
        if len(experiments) > 10:
            logger.info(f"  ... and {len(experiments) - 10} more")
        return
    
    # Estimate time
    est_time_min = len(experiments) * 0.75 / max(args.workers, 1)  # ~45 sec per experiment
    logger.info(f"Estimated time: {est_time_min:.0f} minutes ({est_time_min/60:.1f} hours)")
    logger.info("")
    
    # Run analysis
    start_time = datetime.now()
    summary = run_batch_analysis(experiments, workers=args.workers)
    elapsed = datetime.now() - start_time
    
    # Export results
    logger.info("")
    logger.info("Exporting results to Parquet...")
    export_results_to_parquet()
    
    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total experiments: {summary['total']}")
    logger.info(f"  Successful: {summary['successful']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Success rate: {summary['success_rate']:.1f}%")
    if summary['successful_experiments']:
        display_list = summary['successful_experiments'][:5]  # Show up to 5
        logger.info(f"  Completed: {', '.join(display_list)}")
        if len(summary['successful_experiments']) > 5:
            logger.info(f"             ... and {len(summary['successful_experiments']) - 5} more")
    logger.info(f"  Elapsed time: {elapsed}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

