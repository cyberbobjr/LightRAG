#!/usr/bin/env python3
"""
Script de gestion des versions pour LightRAG
Usage: python version_manager.py [patch|minor|major]
"""

import re
import sys
from pathlib import Path

class VersionManager:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.init_file = self.root_dir / "lightrag" / "__init__.py"
        self.api_init_file = self.root_dir / "lightrag" / "api" / "__init__.py"
        
    def get_current_version(self):
        """Récupère la version actuelle depuis __init__.py"""
        with open(self.init_file, 'r') as f:
            content = f.read()
            
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
        else:
            raise ValueError("Version non trouvée dans __init__.py")
    
    def parse_version(self, version_str):
        """Parse une version au format: 1.4.9"""
        # Pattern pour: major.minor.patch
        pattern = r'^(\d+)\.(\d+)\.(\d+)$'
        match = re.match(pattern, version_str)
        
        if not match:
            raise ValueError(f"Format de version invalide: {version_str}")
            
        return {
            'major': int(match.group(1)),
            'minor': int(match.group(2)), 
            'patch': int(match.group(3))
        }
    
    def increment_version(self, version_parts, bump_type):
        """Incrémente la version selon le type demandé"""
        if bump_type == "major":
            version_parts['major'] += 1
            version_parts['minor'] = 0
            version_parts['patch'] = 0
        elif bump_type == "minor":
            version_parts['minor'] += 1
            version_parts['patch'] = 0
        elif bump_type == "patch":
            version_parts['patch'] += 1
        else:
            raise ValueError("Type de bump invalide. Utilisez: major, minor, ou patch")
            
        return f"{version_parts['major']}.{version_parts['minor']}.{version_parts['patch']}"
    
    def update_version_files(self, new_version):
        """Met à jour les fichiers avec la nouvelle version"""
        # Mise à jour de lightrag/__init__.py
        with open(self.init_file, 'r') as f:
            content = f.read()
        
        content = re.sub(
            r'(__version__\s*=\s*["\'])[^"\']+(["\'])',
            f'\\g<1>{new_version}\\g<2>',
            content
        )
        
        with open(self.init_file, 'w') as f:
            f.write(content)
        
        # Optionnel: mettre à jour l'API version avec timestamp
        from datetime import datetime
        api_version = datetime.now().strftime("%m%d")
        
        with open(self.api_init_file, 'r') as f:
            api_content = f.read()
        
        api_content = re.sub(
            r'(__api_version__\s*=\s*["\'])[^"\']+(["\'])',
            f'\\g<1>{api_version}\\g<2>',
            api_content
        )
        
        with open(self.api_init_file, 'w') as f:
            f.write(api_content)
            
        return api_version
    
    def bump_version(self, bump_type="patch"):
        """Fonction principale pour incrémenter la version"""
        try:
            current_version = self.get_current_version()
            print(f"Version actuelle: {current_version}")
            
            version_parts = self.parse_version(current_version)
            new_version = self.increment_version(version_parts, bump_type)
            
            api_version = self.update_version_files(new_version)
            
            print(f"Nouvelle version: {new_version}")
            print(f"Version API: {api_version}")
            print("\nFichiers mis à jour:")
            print(f"  - {self.init_file}")
            print(f"  - {self.api_init_file}")
            
            return new_version
            
        except Exception as e:
            print(f"Erreur: {e}")
            return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python version_manager.py [major|minor|patch]")
        print("  major: 1.4.9 -> 2.0.0")
        print("  minor: 1.4.9 -> 1.5.0") 
        print("  patch: 1.4.9 -> 1.4.10")
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    if bump_type not in ['major', 'minor', 'patch']:
        print("Type de bump invalide. Utilisez: major, minor, ou patch")
        sys.exit(1)
    
    manager = VersionManager()
    new_version = manager.bump_version(bump_type)
    
    if new_version:
        print(f"\n✅ Version mise à jour avec succès: {new_version}")
        print("\nProchaines étapes recommandées:")
        print("1. git add .")
        print(f"2. git commit -m 'chore: bump version to {new_version}'")
        print(f"3. git tag v{new_version}")
        print("4. git push origin HEAD --tags")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()