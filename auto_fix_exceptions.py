import re
from pathlib import Path


def auto_fix_broad_except(file_path: Path):
    content = file_path.read_text(encoding='utf-8')
    
    # Heuristic: Find 'except Exception as e:' blocks that don't have logger.* or raise
    # We look for the pattern and check the following block
    
    def replacement(match):
        full_block = match.group(0)
        
        # Check if it already has logging or raise
        if re.search(r'logger\.(error|critical|exception)|raise', full_block):
            return full_block
            
        # Add logging and re-raise
        new_block = re.sub(
            r'(except Exception as e:)', 
            r'\1\n            self.logger.error(f"Виникла помилка: {e}", exc_info=True)\n            raise', 
            full_block
        )
        return new_block

    # This is a basic regex-based approach; complex blocks might require AST parsing.
    # For now, targeting simple cases.
    new_content = re.sub(r'except Exception as e:.*?(?=\n\s+(?:except|finally|def|class)|$)', replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        file_path.write_text(new_content, encoding='utf-8')
        return True
    return False

# Test on one file
if __name__ == "__main__":
    target = Path("src/analytics/analyzers/causal_event_finder.py")
    if auto_fix_broad_except(target):
        print(f"Fixed {target}")
    else:
        print("No changes needed or fix failed.")
