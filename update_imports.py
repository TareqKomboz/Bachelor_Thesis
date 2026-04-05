import os
import re

packages = [
    'agents', 'common', 'environments', 'evaluation', 
    'objective_functions', 'training', 'analysis', 'definitions'
]

def update_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    new_content = content
    
    # from <package> -> from l2o.<package>
    for pkg in packages:
        # Use word boundaries and ensure it's not already prefixed with l2o.
        pattern = rf'(?<!l2o\.)\bfrom {pkg}\b'
        new_content = re.sub(pattern, f'from l2o.{pkg}', new_content)
    
    # Handle the typo 'definitons'
    new_content = re.sub(rf'(?<!l2o\.)\bfrom definitons\b', 'from l2o.definitions', new_content)
    
    # import definitions -> import l2o.definitions as definitions
    # check for 'import definitions' that is NOT already 'import l2o.definitions'
    new_content = re.sub(r'(?<!l2o\.)\bimport definitions\b', 'import l2o.definitions as definitions', new_content)

    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

def main():
    files_updated = 0
    for root, dirs, files in os.walk('.'):
        # Skip some directories if needed, but here we want to cover everything.
        if '.git' in root or '.idea' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_file(file_path):
                    print(f"Updated: {file_path}")
                    files_updated += 1
    print(f"Total files updated: {files_updated}")

if __name__ == "__main__":
    main()
