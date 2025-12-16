#!/usr/bin/env python3
"""
MPContribs Project Manager

Handles the full lifecycle of MPContribs projects:
1. Create project with proper metadata
2. Initialize columns with units
3. Format contributions with units in values
4. Submit contributions

This follows the official MPContribs API patterns.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

# Load .env file if present
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MPContribsProjectManager:
    """Manages MPContribs project lifecycle"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('MPCONTRIBS_API_KEY')
        
        if not self.api_key:
            logger.warning("No MPContribs API key set")
            return
        
        # Import MPContribs clients
        try:
            from mp_api.client import MPRester
            from mpcontribs.client import Client as ContribsClient
            
            self.mpr = MPRester(api_key=self.api_key)
            self.client = None  # Will be set per project
            
        except ImportError:
            logger.error("mpcontribs-client not installed. Run: pip install mpcontribs-client mp-api")
            raise
    
    def ensure_project_exists(self, project_name: str, config: Dict) -> bool:
        """
        Create project if it doesn't exist
        
        Args:
            project_name: MPContribs project identifier
            config: Product configuration with metadata
        
        Returns:
            True if project exists or was created
        """
        from mpcontribs.client import Client as ContribsClient
        
        try:
            # Check if project exists
            self.client = ContribsClient(project=project_name, apikey=self.api_key)
            project = self.client.get_project()
            
            if project:
                logger.info(f"Project '{project_name}' already exists")
                return True
                
        except Exception:
            # Project doesn't exist, create it
            pass
        
        # Create new project
        metadata = config.get('metadata', {})
        
        try:
            self.mpr.contribs.create_project(
                name=project_name,
                title=metadata.get('title', f"A-Lab {project_name} Dataset"),
                authors=metadata.get('authors', 'A-Lab Team, Lawrence Berkeley National Laboratory'),
                description=metadata.get('description', 
                    'Automated materials synthesis and characterization data from A-Lab'),
                url=metadata.get('references', [{}])[0].get('url', 
                    'https://github.com/CederGroupHub/alab_data')
            )
            
            logger.info(f"✓ Created MPContribs project: {project_name}")
            
            # Initialize client for the new project
            self.client = ContribsClient(project=project_name, apikey=self.api_key)
            
            # Update with additional references if provided
            if metadata.get('references'):
                self.client.update_project({
                    'references': metadata['references']
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return False
    
    def initialize_columns(self, columns_with_units: Dict[str, Optional[str]]) -> bool:
        """
        Initialize MPContribs columns with units
        
        Args:
            columns_with_units: Dict of column_name -> unit
                - None for dimensionless numeric/boolean
                - "" (empty string) for text fields
                - "degC", "%", etc. for values with units
        
        Returns:
            True if successful
        """
        if not self.client:
            logger.error("No active MPContribs client")
            return False
        
        try:
            self.client.init_columns(columns_with_units)
            logger.info(f"✓ Initialized {len(columns_with_units)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize columns: {e}")
            return False
    
    def set_column_descriptions(self, descriptions: Dict[str, str]) -> bool:
        """
        Set human-readable descriptions for columns
        
        Args:
            descriptions: Dict of column_name -> description
        
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            self.client.update_project({'other': descriptions})
            logger.info(f"✓ Set descriptions for {len(descriptions)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set descriptions: {e}")
            return False
    
    def format_contribution(self, 
                           experiment_data: Dict,
                           schema: Dict,
                           project_name: str) -> Dict:
        """
        Format experiment data as MPContribs contribution
        
        Args:
            experiment_data: Raw experiment data
            schema: Schema definition with units
            project_name: MPContribs project name
        
        Returns:
            Properly formatted contribution dict
        """
        # Convert field names to camelCase
        def to_camel_case(snake_str: str) -> str:
            components = snake_str.split('_')
            return components[0] + ''.join(x.title() for x in components[1:])
        
        # Build data dict with units embedded in values
        data = {}
        
        for field_name, field_def in schema.items():
            camel_name = to_camel_case(field_name)
            value = experiment_data.get(field_name)
            
            if value is not None:
                # Format value with unit
                if field_def.get('unit') is not None:
                    # Numeric with unit
                    data[camel_name] = f"{value} {field_def['unit']}"
                elif field_def.get('unit') == "":
                    # String value
                    data[camel_name] = str(value)
                else:
                    # Dimensionless number or boolean
                    data[camel_name] = str(value)
        
        # Build contribution structure
        contribution = {
            'project': project_name,
            'identifier': experiment_data.get('target_formula', experiment_data.get('name', 'unknown')),
            'formula': experiment_data.get('target_formula'),
            'data': data
        }
        
        # Add structures if available (e.g., from CIF files)
        if 'structure' in experiment_data:
            contribution['structures'] = [experiment_data['structure']]
        
        # Add tables if available (e.g., XRD phases)
        tables = []
        
        # Check for XRD phases
        if 'xrd_phases' in experiment_data and experiment_data['xrd_phases']:
            phases_df = pd.DataFrame(experiment_data['xrd_phases'])
            phases_df.attrs['name'] = 'xrdPhases'
            phases_df.attrs['title'] = 'XRD Phase Identification'
            tables.append(phases_df)
        
        if tables:
            contribution['tables'] = tables
        
        return contribution
    
    def submit_contributions(self, contributions: List[Dict], dry_run: bool = True) -> int:
        """
        Submit contributions to MPContribs
        
        Args:
            contributions: List of formatted contribution dicts
            dry_run: If True, only log what would be uploaded
        
        Returns:
            Number of successful uploads
        """
        if dry_run:
            logger.info(f"[DRY RUN] Would upload {len(contributions)} contributions")
            for i, contrib in enumerate(contributions[:3]):  # Show first 3
                logger.info(f"  [{i+1}] {contrib.get('identifier', 'unknown')}")
                logger.info(f"      Fields: {list(contrib.get('data', {}).keys())}")
            if len(contributions) > 3:
                logger.info(f"  ... and {len(contributions) - 3} more")
            return len(contributions)
        
        if not self.client:
            logger.error("No active MPContribs client")
            logger.error("Set MPCONTRIBS_API_KEY in .env file to enable uploads")
            return 0
        
        success_count = 0
        
        try:
            # Submit in batches (MPContribs recommends max 500 at a time)
            batch_size = 100
            
            for i in range(0, len(contributions), batch_size):
                batch = contributions[i:i + batch_size]
                
                try:
                    result = self.client.submit_contributions(batch)
                    success_count += len(batch)
                    logger.info(f"✓ Uploaded batch {i//batch_size + 1}: {len(batch)} contributions")
                    
                except Exception as e:
                    logger.error(f"Failed to upload batch {i//batch_size + 1}: {e}")
            
            logger.info(f"Successfully uploaded {success_count}/{len(contributions)} contributions")
            return success_count
            
        except Exception as e:
            logger.error(f"Failed to submit contributions: {e}")
            return success_count
    
    def delete_all_contributions(self) -> bool:
        """Delete all contributions from the project (use with caution!)"""
        if not self.client:
            return False
        
        try:
            self.client.delete_contributions()
            logger.info("✓ Deleted all contributions from project")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete contributions: {e}")
            return False
    
    def make_project_public(self) -> bool:
        """Make the project publicly visible"""
        if not self.client:
            return False
        
        try:
            self.client.make_public()
            logger.info("✓ Project is now public")
            return True
            
        except Exception as e:
            logger.error(f"Failed to make project public: {e}")
            return False
    
    def make_project_private(self) -> bool:
        """Make the project private"""
        if not self.client:
            return False
        
        try:
            self.client.make_private()
            logger.info("✓ Project is now private")
            return True
            
        except Exception as e:
            logger.error(f"Failed to make project private: {e}")
            return False
