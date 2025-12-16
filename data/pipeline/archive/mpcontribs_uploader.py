#!/usr/bin/env python3
"""
MPContribs Uploader for A-Lab Experiments

Uploads experiment data to Materials Project Contributions portal.
Follows MPContribs schema requirements with parquet files as attachments.

API key should be set in .env file at project root:
    MPCONTRIBS_API_KEY=your_key_here
"""

import os
import io
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import logging

# Load .env file if present
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv not installed

from pipeline_state import PipelineStateManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PARQUET_DATA_DIR = Path(__file__).parent.parent / "parquet"
MPCONTRIBS_PROJECT = "alab_synthesis"


class MPContribsUploader:
    """Upload A-Lab experiment data to MPContribs"""
    
    def __init__(self, api_key: str = None, project: str = MPCONTRIBS_PROJECT):
        self.api_key = api_key or os.environ.get('MPCONTRIBS_API_KEY')
        self.project = project
        self.state = PipelineStateManager()
        self.client = None
        
        if not self.api_key:
            logger.warning("MPContribs API key not set. Set MPCONTRIBS_API_KEY in .env file.")
    
    def _get_client(self):
        """Lazy load MPContribs client"""
        if self.client is None:
            try:
                from mpcontribs.client import Client
                self.client = Client(self.api_key)
                logger.info(f"Connected to MPContribs")
            except ImportError:
                logger.error("mpcontribs-client not installed. Run: pip install mpcontribs-client")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to MPContribs: {e}")
                raise
        return self.client
    
    def _load_experiment_data(self, experiment_name: str) -> Optional[Dict]:
        """Load all data for an experiment from parquet files"""
        experiments_file = PARQUET_DATA_DIR / "experiments.parquet"
        if not experiments_file.exists():
            return None
        
        experiments_df = pd.read_parquet(experiments_file)
        exp_row = experiments_df[experiments_df['name'] == experiment_name]
        
        if exp_row.empty:
            return None
        
        exp = exp_row.iloc[0]
        exp_id = exp['experiment_id']
        
        # Load related data
        heating = None
        heating_file = PARQUET_DATA_DIR / "heating_sessions.parquet"
        if heating_file.exists():
            heating_df = pd.read_parquet(heating_file)
            h = heating_df[heating_df['experiment_id'] == exp_id]
            if len(h) > 0:
                heating = h.iloc[0]
        
        recovery = None
        recovery_file = PARQUET_DATA_DIR / "powder_recovery.parquet"
        if recovery_file.exists():
            recovery_df = pd.read_parquet(recovery_file)
            r = recovery_df[recovery_df['experiment_id'] == exp_id]
            if len(r) > 0:
                recovery = r.iloc[0]
        
        # XRD analysis results
        refinement = None
        refinements_file = PARQUET_DATA_DIR / "xrd_refinements.parquet"
        if refinements_file.exists():
            ref_df = pd.read_parquet(refinements_file)
            ref_row = ref_df[ref_df['experiment_name'] == experiment_name]
            if not ref_row.empty:
                refinement = ref_row.iloc[0]
        
        phases = pd.DataFrame()
        phases_file = PARQUET_DATA_DIR / "xrd_phases.parquet"
        if phases_file.exists():
            phases_df = pd.read_parquet(phases_file)
            phases = phases_df[phases_df['experiment_name'] == experiment_name]
        
        return {
            'experiment': exp,
            'heating': heating,
            'recovery': recovery,
            'refinement': refinement,
            'phases': phases
        }
    
    def _create_contribution(self, experiment_name: str, data: Dict, dry_run: bool = True) -> Dict:
        """
        Create MPContribs contribution structure
        
        Following the schema:
        - identifier: target formula (for linking to MP materials)
        - data: hierarchical key-value data
        - tables: phases as DataFrame
        - attachments: parquet files
        """
        exp = data['experiment']
        heating = data.get('heating')
        recovery = data.get('recovery')
        refinement = data.get('refinement')
        phases = data.get('phases')
        
        # Build data dictionary with safe access
        synthesis_data = {}
        if heating is not None:
            if pd.notna(heating.get('heating_temperature')):
                synthesis_data['heating_temperature_C'] = float(heating['heating_temperature'])
            if pd.notna(heating.get('heating_time')):
                synthesis_data['heating_time_min'] = float(heating['heating_time'])
            if pd.notna(heating.get('atmosphere')):
                synthesis_data['atmosphere'] = str(heating['atmosphere'])
        
        if recovery is not None and pd.notna(recovery.get('recovery_yield_percent')):
            synthesis_data['recovery_yield_pct'] = float(recovery['recovery_yield_percent'])
        
        xrd_data = {'success': False}
        if refinement is not None:
            xrd_data['success'] = bool(refinement.get('success', False))
            if pd.notna(refinement.get('rwp')):
                xrd_data['rwp'] = float(refinement['rwp'])
            if pd.notna(refinement.get('num_phases')):
                xrd_data['num_phases'] = int(refinement['num_phases'])
            if pd.notna(refinement.get('error')):
                xrd_data['error'] = str(refinement['error'])
        
        contribution = {
            'project': self.project,
            'identifier': str(exp['target_formula']),
            'data': {
                'experiment': {
                    'name': experiment_name,
                    'type': str(exp['experiment_type']),
                    'status': str(exp['status']),
                    'last_updated': str(exp['last_updated'])
                },
                'synthesis': synthesis_data if synthesis_data else {'note': 'No synthesis data'},
                'xrd_analysis': xrd_data
            }
        }
        
        # Add phases as table if available
        if phases is not None and len(phases) > 0:
            phases_table = phases[['phase_name', 'spacegroup', 'weight_fraction']].copy()
            phases_table.attrs['name'] = 'identified_phases'
            phases_table.attrs['title'] = 'XRD Phase Identification'
            contribution['tables'] = [phases_table]
        
        if dry_run:
            logger.info(f"[DRY RUN] Would upload: {experiment_name}")
            logger.info(f"  Formula: {contribution['identifier']}")
            logger.info(f"  XRD success: {xrd_data.get('success', False)}")
            if phases is not None:
                logger.info(f"  Phases: {len(phases)}")
            return {'dry_run': True, 'experiment': experiment_name, 'contribution': contribution}
        
        return contribution
    
    def upload_experiment(self, experiment_name: str, dry_run: bool = True) -> Optional[str]:
        """
        Upload a single experiment to MPContribs
        
        Returns:
            contribution_id if successful, 'dry_run' for dry runs, None if failed
        """
        data = self._load_experiment_data(experiment_name)
        if not data:
            logger.warning(f"No data found for experiment: {experiment_name}")
            return None
        
        contribution = self._create_contribution(experiment_name, data, dry_run=dry_run)
        
        if dry_run:
            return 'dry_run'
        
        if not self.api_key:
            logger.error("Cannot upload: MPContribs API key not set")
            return None
        
        try:
            client = self._get_client()
            
            # Check if contribution already exists
            existing = client.contributions.get_entries(
                project=self.project,
                query={'data__experiment__name': experiment_name},
                _fields=['id']
            ).result()
            
            if existing.get('data'):
                # Update existing
                contrib_id = existing['data'][0]['id']
                client.contributions.update_entry(
                    pk=contrib_id,
                    contribution=contribution
                ).result()
                logger.info(f"✓ Updated {experiment_name} (ID: {contrib_id})")
                return contrib_id
            else:
                # Create new
                result = client.contributions.create_entry(
                    contribution=contribution
                ).result()
                logger.info(f"✓ Created {experiment_name} (ID: {result['id']})")
                return result['id']
                
        except Exception as e:
            logger.error(f"✗ Failed to upload {experiment_name}: {e}")
            return None
    
    def upload_all_experiments(self, dry_run: bool = True, limit: Optional[int] = None) -> Dict:
        """
        Upload all experiments that haven't been uploaded yet
        
        Returns:
            Summary dict with success/failed counts
        """
        to_upload = self.state.get_experiments_to_upload()
        
        if limit:
            to_upload = to_upload[:limit]
        
        if not to_upload:
            logger.info("No experiments to upload")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Uploading {len(to_upload)} experiments")
        
        run_id = self.state.record_run(
            run_type='mpcontribs_upload',
            phases=['mpcontribs_upload'],
            experiments=to_upload,
            dry_run=dry_run
        )
        
        success = 0
        failed = 0
        
        for exp_name in to_upload:
            start = datetime.now()
            contrib_id = self.upload_experiment(exp_name, dry_run=dry_run)
            duration = (datetime.now() - start).total_seconds()
            
            if contrib_id:
                success += 1
                self.state.update_experiment_status(
                    run_id=run_id,
                    experiment_name=exp_name,
                    phase='mpcontribs_upload',
                    status='success',
                    duration_seconds=duration,
                    mpcontribs_id=contrib_id if contrib_id != 'dry_run' else None
                )
            else:
                failed += 1
                self.state.update_experiment_status(
                    run_id=run_id,
                    experiment_name=exp_name,
                    phase='mpcontribs_upload',
                    status='failed',
                    duration_seconds=duration
                )
        
        logger.info("=" * 50)
        logger.info(f"Upload complete: {success} success, {failed} failed")
        if dry_run:
            logger.info("This was a DRY RUN. Use --submit-to-mpcontribs to upload.")
        
        return {'total': len(to_upload), 'success': success, 'failed': failed, 'dry_run': dry_run}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Upload A-Lab experiments to MPContribs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python mpcontribs_uploader.py                    # Dry run (default)
  python mpcontribs_uploader.py --submit-to-mpcontribs  # Actually upload
  python mpcontribs_uploader.py -e NSC_249 --submit-to-mpcontribs  # Single experiment
  python mpcontribs_uploader.py --limit 5          # First 5 experiments (dry run)
        '''
    )
    
    parser.add_argument('--submit-to-mpcontribs', action='store_true',
                       help='Actually submit to MPContribs (default is dry run)')
    parser.add_argument('--experiment', '-e', type=str,
                       help='Upload single experiment')
    parser.add_argument('--limit', '-l', type=int,
                       help='Limit number of experiments to upload')
    parser.add_argument('--project', '-p', type=str, default=MPCONTRIBS_PROJECT,
                       help=f'MPContribs project name (default: {MPCONTRIBS_PROJECT})')
    
    args = parser.parse_args()
    dry_run = not args.submit_to_mpcontribs
    
    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No data will be uploaded")
        print("Use --submit-to-mpcontribs to actually upload")
        print("=" * 60)
        print()
    
    uploader = MPContribsUploader(project=args.project)
    
    if args.experiment:
        result = uploader.upload_experiment(args.experiment, dry_run=dry_run)
        if result:
            print(f"\n✓ {'Would upload' if dry_run else 'Uploaded'}: {args.experiment}")
        else:
            print(f"\n✗ Failed: {args.experiment}")
    else:
        uploader.upload_all_experiments(dry_run=dry_run, limit=args.limit)


if __name__ == '__main__':
    main()

