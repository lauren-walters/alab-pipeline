#!/usr/bin/env python3
"""
Parquet Data Loader for A-Lab Dashboard
Loads experiment data from consolidated Parquet files for fast analytical queries

Schema v2 (Consolidated):
- experiments.parquet: Main table with ALL 1:1 data (~45 columns)
- experiment_elements.parquet: Elements per experiment (1:N)
- powder_doses.parquet: Individual powder doses (1:N)
- temperature_logs.parquet: Temperature readings (1:N, optional)
- xrd_data_points.parquet: Raw XRD patterns (1:N, optional)
- workflow_tasks.parquet: Task execution history (1:N, optional)

Note: This loader provides data from the MongoDB → Parquet pipeline.
SEM-EDS analysis data is NOT included in this pipeline.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Default parquet data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "parquet"

# XRD analysis results directory (from DARA)
XRD_RESULTS_DIR = Path(__file__).parent.parent / "data" / "xrd_creation" / "results"


class ParquetDataLoader:
    """Load experiment data from consolidated Parquet files"""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing parquet files
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info(f"Parquet data directory: {self.data_dir}")
        
        # Lazy-load dataframes (only load when needed)
        self._experiments = None
        self._experiment_elements = None
        self._powder_doses = None
        self._temperature_logs = None
        self._xrd_data_points = None
        self._workflow_tasks = None
        self._pipeline_runs = None
    
    @property
    def experiments(self) -> pd.DataFrame:
        """Main experiments table with all 1:1 data merged"""
        if self._experiments is None:
            self._experiments = self._load_parquet('experiments')
        return self._experiments
    
    @property
    def experiment_elements(self) -> pd.DataFrame:
        if self._experiment_elements is None:
            self._experiment_elements = self._load_parquet('experiment_elements')
        return self._experiment_elements
    
    @property
    def powder_doses(self) -> pd.DataFrame:
        if self._powder_doses is None:
            self._powder_doses = self._load_parquet('powder_doses')
        return self._powder_doses
    
    @property
    def temperature_logs(self) -> pd.DataFrame:
        if self._temperature_logs is None:
            self._temperature_logs = self._load_parquet('temperature_logs')
        return self._temperature_logs
    
    @property
    def xrd_data_points(self) -> pd.DataFrame:
        if self._xrd_data_points is None:
            self._xrd_data_points = self._load_parquet('xrd_data_points')
        return self._xrd_data_points
    
    @property
    def workflow_tasks(self) -> pd.DataFrame:
        if self._workflow_tasks is None:
            self._workflow_tasks = self._load_parquet('workflow_tasks')
        return self._workflow_tasks
    
    @property
    def pipeline_runs(self) -> pd.DataFrame:
        """Load pipeline run history for dashboard status display"""
        if self._pipeline_runs is None:
            pipeline_file = self.data_dir.parent / "pipeline" / "pipeline_runs.parquet"
            if pipeline_file.exists():
                self._pipeline_runs = pd.read_parquet(pipeline_file)
            else:
                self._pipeline_runs = pd.DataFrame()
        return self._pipeline_runs
    
    # ========================================
    # Backward compatibility properties
    # These extract data from consolidated experiments table
    # ========================================
    
    @property
    def heating_sessions(self) -> pd.DataFrame:
        """Extract heating data from consolidated experiments table (backward compat)"""
        heating_cols = [c for c in self.experiments.columns if c.startswith('heating_') or c == 'experiment_id']
        df = self.experiments[heating_cols].copy()
        # Rename columns to remove prefix for backward compatibility
        df.columns = [c.replace('heating_', '') if c != 'experiment_id' else c for c in df.columns]
        return df
    
    @property
    def powder_recovery(self) -> pd.DataFrame:
        """Extract recovery data from consolidated experiments table (backward compat)"""
        recovery_cols = [c for c in self.experiments.columns if c.startswith('recovery_') or c == 'experiment_id']
        df = self.experiments[recovery_cols].copy()
        df.columns = [c.replace('recovery_', '') if c != 'experiment_id' else c for c in df.columns]
        return df
    
    @property
    def xrd_measurements(self) -> pd.DataFrame:
        """Extract XRD measurement data from consolidated experiments table (backward compat)"""
        xrd_cols = [c for c in self.experiments.columns if c.startswith('xrd_') or c == 'experiment_id']
        df = self.experiments[xrd_cols].copy()
        df.columns = [c.replace('xrd_', '') if c != 'experiment_id' else c for c in df.columns]
        return df
    
    @property
    def sample_finalization(self) -> pd.DataFrame:
        """Extract finalization data from consolidated experiments table (backward compat)"""
        final_cols = [c for c in self.experiments.columns if c.startswith('finalization_') or c == 'experiment_id']
        df = self.experiments[final_cols].copy()
        df.columns = [c.replace('finalization_', '') if c != 'experiment_id' else c for c in df.columns]
        return df
    
    @property
    def dosing_sessions(self) -> pd.DataFrame:
        """Extract dosing session data from consolidated experiments table (backward compat)"""
        dosing_cols = [c for c in self.experiments.columns if c.startswith('dosing_') or c == 'experiment_id']
        df = self.experiments[dosing_cols].copy()
        df.columns = [c.replace('dosing_', '') if c != 'experiment_id' else c for c in df.columns]
        return df
    
    def get_pipeline_status(self) -> Dict:
        """
        Get current pipeline status for dashboard display.
        
        Returns:
            Dict with pipeline statistics
        """
        # Count from refinements file
        refinements_file = self.data_dir / "xrd_refinements.parquet"
        analyzed = 0
        if refinements_file.exists():
            ref_df = pd.read_parquet(refinements_file)
            analyzed = len(ref_df[ref_df['success'] == True])
        
        # Count uploaded from pipeline state
        uploaded = 0
        last_run = None
        total_runs = 0
        
        if len(self.pipeline_runs) > 0:
            uploaded = len(self.pipeline_runs[
                (self.pipeline_runs['phase'] == 'mpcontribs_upload') & 
                (self.pipeline_runs['status'] == 'success') &
                (self.pipeline_runs['dry_run'] == False)
            ]['experiment_name'].unique())
            
            last_run = self.pipeline_runs['run_timestamp'].max()
            total_runs = self.pipeline_runs['run_id'].nunique()
        
        return {
            'total_experiments': len(self.experiments),
            'analyzed_experiments': analyzed,
            'uploaded_experiments': uploaded,
            'pending_upload': analyzed - uploaded,
            'last_run': str(last_run) if last_run else None,
            'total_runs': total_runs
        }
    
    def _load_parquet(self, name: str) -> pd.DataFrame:
        """Load a parquet file"""
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            logger.warning(f"Parquet file not found: {path}")
            return pd.DataFrame()
        
        return pd.read_parquet(path)
    
    def get_experiment_list(self) -> List[str]:
        """Get list of all experiment names"""
        return sorted(self.experiments['name'].tolist())
    
    def get_experiment_info(self, experiment_name: str) -> Dict:
        """Get basic experiment information"""
        exp = self.experiments[self.experiments['name'] == experiment_name]
        
        if exp.empty:
            return {}
        
        exp_row = exp.iloc[0]
        exp_id = exp_row['experiment_id']
        
        # Get elements for this experiment
        elements = self.experiment_elements[
            self.experiment_elements['experiment_id'] == exp_id
        ]['element_symbol'].tolist()
        
        return {
            'experiment_id': exp_id,
            'name': exp_row['name'],
            'external_id': exp_id,
            'target_formula': exp_row['target_formula'],
            'last_updated': exp_row['last_updated'],
            'status': exp_row['status'],
            'elements': elements
        }
    
    def get_dosing_data(self, experiment_name: str) -> Dict:
        """Get powder dosing data"""
        info = self.get_experiment_info(experiment_name)
        if not info:
            return {}
        
        exp_id = info['experiment_id']
        exp_row = self.experiments[self.experiments['experiment_id'] == exp_id].iloc[0]
        
        # Get powder doses
        doses = self.powder_doses[
            self.powder_doses['experiment_id'] == exp_id
        ].sort_values('dose_sequence')
        
        # Group doses by powder
        powders_dict = {}
        for _, dose in doses.iterrows():
            powder_name = dose['powder_name']
            if powder_name not in powders_dict:
                powders_dict[powder_name] = {
                    'PowderName': powder_name,
                    'TargetMass': float(dose['target_mass']) if pd.notna(dose['target_mass']) else 0.0,
                    'Doses': []
                }
            
            powders_dict[powder_name]['Doses'].append({
                'Mass': float(dose['actual_mass']) if pd.notna(dose['actual_mass']) else 0.0,
                'TimeStamp': str(dose['dose_timestamp']) if pd.notna(dose['dose_timestamp']) else None
            })
        
        return {
            'crucible_position': exp_row.get('dosing_crucible_position'),
            'crucible_sub_rack': exp_row.get('dosing_crucible_sub_rack'),
            'actual_transfer_mass': exp_row.get('dosing_actual_transfer_mass'),
            'Powders': list(powders_dict.values()),
            'doses': doses.to_dict('records')
        }
    
    def get_heating_data(self, experiment_name: str) -> Dict:
        """Get heating session data from consolidated experiments table"""
        info = self.get_experiment_info(experiment_name)
        if not info:
            return {}
        
        exp_id = info['experiment_id']
        exp_row = self.experiments[self.experiments['experiment_id'] == exp_id].iloc[0]
        
        return {
            'heating_temperature': exp_row.get('heating_temperature'),
            'heating_time': exp_row.get('heating_time'),
            'cooling_rate': exp_row.get('heating_cooling_rate'),
            'atmosphere': exp_row.get('heating_atmosphere'),
            'flow_rate': exp_row.get('heating_flow_rate_ml_min'),
            'low_temperature_calcination': exp_row.get('heating_low_temp_calcination')
        }
    
    def get_temperature_profile(self, experiment_name: str) -> List[Dict]:
        """Get temperature log data"""
        info = self.get_experiment_info(experiment_name)
        if not info:
            return []
        
        exp_id = info['experiment_id']
        
        temp_logs = self.temperature_logs[
            self.temperature_logs['experiment_id'] == exp_id
        ].sort_values('sequence_number')
        
        if temp_logs.empty:
            return []
        
        return temp_logs[['time_minutes', 'temperature_celsius']].to_dict('records')
    
    def get_recovery_data(self, experiment_name: str) -> Dict:
        """Get powder recovery data from consolidated experiments table"""
        info = self.get_experiment_info(experiment_name)
        if not info:
            return {}
        
        exp_id = info['experiment_id']
        exp_row = self.experiments[self.experiments['experiment_id'] == exp_id].iloc[0]
        
        return {
            'initial_crucible_weight_mg': exp_row.get('recovery_initial_crucible_weight_mg'),
            'weight_collected_mg': exp_row.get('recovery_weight_collected_mg'),
            'recovery_yield_percent': exp_row.get('recovery_yield_percent')
        }
    
    def get_xrd_metadata(self, experiment_name: str) -> Dict:
        """Get XRD measurement metadata from consolidated experiments table"""
        info = self.get_experiment_info(experiment_name)
        if not info:
            return {}
        
        exp_id = info['experiment_id']
        exp_row = self.experiments[self.experiments['experiment_id'] == exp_id].iloc[0]
        
        return {
            'sampleid_in_aeris': exp_row.get('xrd_sampleid_in_aeris'),
            'xrd_holder_index': exp_row.get('xrd_holder_index'),
            'total_mass_dispensed_mg': exp_row.get('xrd_total_mass_dispensed_mg'),
            'met_target_mass': exp_row.get('xrd_met_target_mass')
        }
    
    def get_xrd_pattern(self, experiment_name: str, 
                       max_points: Optional[int] = None) -> Dict:
        """Get XRD diffraction pattern data"""
        metadata = self.get_xrd_metadata(experiment_name)
        if not metadata:
            return {}
        
        info = self.get_experiment_info(experiment_name)
        exp_id = info['experiment_id']
        
        # Get data points
        xrd_points = self.xrd_data_points[
            self.xrd_data_points['experiment_id'] == exp_id
        ].sort_values('point_index')
        
        if max_points:
            xrd_points = xrd_points.head(max_points)
        
        if not xrd_points.empty:
            metadata['twotheta'] = xrd_points['twotheta'].tolist()
            metadata['counts'] = xrd_points['counts'].tolist()
        
        return metadata
    
    def get_xrd_analysis(self, experiment_name: str) -> Optional[Dict]:
        """
        Get DARA XRD refinement/phase analysis results if available.
        Checks Parquet first (faster), falls back to JSON.
        """
        # Try Parquet first (more efficient for dashboard)
        refinements_file = self.data_dir / "xrd_refinements.parquet"
        phases_file = self.data_dir / "xrd_phases.parquet"
        
        if refinements_file.exists():
            try:
                refinements_df = pd.read_parquet(refinements_file)
                ref_row = refinements_df[refinements_df['experiment_name'] == experiment_name]
                
                if not ref_row.empty:
                    ref = ref_row.iloc[0]
                    
                    success = ref.get('success', True)
                    
                    if not success:
                        return {
                            'success': False,
                            'error': ref.get('error', 'Unknown error'),
                            'error_type': ref.get('error_type'),
                            'chemical_system': ref.get('chemical_system'),
                            'target_formula': ref.get('target_formula')
                        }
                    
                    phases = []
                    if phases_file.exists():
                        phases_df = pd.read_parquet(phases_file)
                        exp_phases = phases_df[phases_df['experiment_name'] == experiment_name]
                        for _, p in exp_phases.iterrows():
                            phases.append({
                                'phase_name': p['phase_name'],
                                'spacegroup': p['spacegroup'],
                                'weight_fraction': p['weight_fraction'],
                                'r_phase': p['r_phase']
                            })
                    
                    return {
                        'success': True,
                        'rwp': ref['rwp'],
                        'rp': ref['rp'],
                        'rexp': ref.get('rexp'),
                        'num_phases': ref['num_phases'],
                        'phases': phases,
                        'chemical_system': ref.get('chemical_system'),
                        'target_formula': ref.get('target_formula')
                    }
            except Exception as e:
                logger.debug(f"Error loading from Parquet: {e}, trying JSON")
        
        # Fall back to JSON file
        result_file = XRD_RESULTS_DIR / f"{experiment_name}_result.json"
        
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            return result
        except Exception as e:
            logger.error(f"Error loading XRD analysis for {experiment_name}: {e}")
            return None
    
    def get_complete_experiment_data(self, experiment_name: str) -> Dict:
        """Get all data for an experiment in dashboard format"""
        try:
            info = self.get_experiment_info(experiment_name)
            if not info:
                return {'name': experiment_name, 'error': 'Experiment not found'}
            
            data = {
                '_id': info['external_id'],
                'name': experiment_name,
                'last_updated': str(info.get('last_updated', '')),
                'metadata': {
                    'target': info.get('target_formula', ''),
                    'elements_present': info.get('elements', [])
                }
            }
            
            # Dosing data
            dosing = self.get_dosing_data(experiment_name)
            if dosing:
                data['metadata']['powderdosing_results'] = {
                    'CruciblePosition': dosing.get('crucible_position'),
                    'CrucibleSubRack': dosing.get('crucible_sub_rack'),
                    'ActualTransferMass': float(dosing.get('actual_transfer_mass', 0) or 0),
                    'Powders': dosing.get('Powders', [])
                }
            
            # Heating data
            heating = self.get_heating_data(experiment_name)
            if heating:
                data['metadata']['heating_results'] = {
                    'heating_temperature': heating.get('heating_temperature'),
                    'heating_time': heating.get('heating_time'),
                    'cooling_rate': heating.get('cooling_rate'),
                    'low_temperature_calcination': heating.get('low_temperature_calcination', False)
                }
                
                # Temperature profile
                temp_profile = self.get_temperature_profile(experiment_name)
                if temp_profile:
                    data['metadata']['heating_results']['temperature_log'] = {
                        'time_minutes': [p['time_minutes'] for p in temp_profile],
                        'temperature_celsius': [p['temperature_celsius'] for p in temp_profile]
                    }
            
            # Recovery data
            recovery = self.get_recovery_data(experiment_name)
            if recovery:
                data['metadata']['recoverpowder_results'] = recovery
            
            # XRD data (limited points for performance)
            xrd = self.get_xrd_pattern(experiment_name, max_points=2000)
            if xrd:
                data['metadata']['diffraction_results'] = xrd
            
            # DARA XRD analysis results (if available)
            xrd_analysis = self.get_xrd_analysis(experiment_name)
            if xrd_analysis:
                data['metadata']['xrd_analysis'] = xrd_analysis
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading experiment {experiment_name}: {e}")
            return {'name': experiment_name, 'error': str(e)}
    
    def get_analysis_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Get all dataframes for analysis.
        Useful for generating statistics, plots, etc.
        """
        return {
            'experiments': self.experiments,
            'experiment_elements': self.experiment_elements,
            'powder_doses': self.powder_doses,
            'workflow_tasks': self.workflow_tasks
        }


if __name__ == '__main__':
    # Test the loader
    loader = ParquetDataLoader()
    
    experiments = loader.get_experiment_list()
    print(f"Found {len(experiments)} experiments")
    print(f"Experiments table columns: {len(loader.experiments.columns)}")
    
    if experiments:
        # Test with first experiment
        exp_name = experiments[0]
        print(f"\nTesting with: {exp_name}")
        data = loader.get_complete_experiment_data(exp_name)
        print(f"  Target: {data.get('metadata', {}).get('target', 'N/A')}")
        print(f"  Elements: {data.get('metadata', {}).get('elements_present', [])}")
