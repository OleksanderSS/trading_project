import ast
import pathlib

def check_broad_except(tree):
    """
    Analyzes an AST tree to find 'except Exception' blocks.
    Returns a list of lineno where potential silent exceptions are found.
    
    A finding is considered a potential 'silent' exception if the block:
    1. Has no 'raise' statement.
    2. Has no 'logger.error' or 'logger.warning' (or equivalent).
    """
    findings = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            # Check if it's 'except Exception'
            if isinstance(node.type, ast.Name) and node.type.id == 'Exception':
                
                # Analyze the body of the except block
                has_raise = False
                has_logging = False
                
                for stmt in node.body:
                    # Look for raise statements
                    if isinstance(stmt, ast.Raise):
                        has_raise = True
                        break
                    
                    # Look for logger.error/warning/etc.
                    # This is a heuristic: check if any Attribute contains 'logger'
                    for subnode in ast.walk(stmt):
                        if isinstance(subnode, ast.Attribute) and isinstance(subnode.value, ast.Name):
                            if subnode.value.id == 'logger':
                                has_logging = True
                                break
                
                if not has_raise and not has_logging:
                    findings.append(node.lineno)
                    
    return findings

# Test with a dummy tree
code = """
try:
    pass
except Exception:
    pass # Should be flagged

try:
    pass
except Exception:
    logger.error("error") # Should NOT be flagged

try:
    pass
except Exception:
    raise RuntimeError("error") # Should NOT be flagged
"""
tree = ast.parse(code)
print(f"Findings at lines: {check_broad_except(tree)}")
