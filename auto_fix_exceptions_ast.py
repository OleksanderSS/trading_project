import ast
from pathlib import Path
import astor

class ExceptFixer(ast.NodeTransformer):
    def __init__(self, filename, dry_run=True):
        self.filename = filename
        self.dry_run = dry_run
        self.changed_blocks = 0
        self.changed = False

    def visit_ExceptHandler(self, node):
        is_exception = (node.type is None or 
                        (isinstance(node.type, ast.Name) and node.type.id == 'Exception'))
        
        if is_exception:
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(ast.Module(body=node.body, type_ignores=[])))
            has_log = any(isinstance(n, ast.Call) and 
                          isinstance(n.func, ast.Attribute) and 
                          n.func.attr in ('error', 'exception', 'critical', 'warning', 'info') 
                          for n in ast.walk(ast.Module(body=node.body, type_ignores=[])))
            
            if not has_raise:
                exc_var = node.name if node.name else "e"
                
                if not has_log:
                    log_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='logger', ctx=ast.Load()), attr='error', ctx=ast.Load()),
                        args=[ast.JoinedStr(values=[
                            ast.Constant(value='Виникла помилка: '), 
                            ast.FormattedValue(value=ast.Name(id=exc_var, ctx=ast.Load()), conversion=-1)
                        ])],
                        keywords=[ast.keyword(arg='exc_info', value=ast.Constant(value=True))]
                    ))
                    node.body.insert(0, log_call)
                
                raise_node = ast.Raise(exc=None, cause=None)
                node.body.append(raise_node)
                self.changed = True
                self.changed_blocks += 1
        
        return node

def analyze_or_apply(file_path: Path, dry_run=True):
    # Handle files with BOM
    source = file_path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    fixer = ExceptFixer(file_path.name, dry_run=dry_run)
    new_tree = fixer.visit(tree)
    
    if fixer.changed:
        if not dry_run:
            fixed_source = astor.to_source(new_tree)
            file_path.write_text(fixed_source, encoding='utf-8')
        return fixer.changed_blocks
    return 0

if __name__ == "__main__":
    import sys
    dry_run = True
    if len(sys.argv) > 1 and sys.argv[1] == "--apply":
        dry_run = False
        
    # Restrict to algorithms for now
    src_dir = Path("src/algorithms")
    total_files_changed = 0
    total_blocks_fixed = 0
    
    for p in src_dir.rglob("*.py"):
        try:
            blocks = analyze_or_apply(p, dry_run=dry_run)
            if blocks > 0:
                print(f"{'[DRY RUN] ' if dry_run else ''}Fixed {blocks} blocks in {p}")
                total_files_changed += 1
                total_blocks_fixed += blocks
        except Exception as e:
            print(f"Failed to process {p}: {e}")
            
    print(f"\nSummary: {total_files_changed} files, {total_blocks_fixed} blocks {'would be ' if dry_run else ''}fixed.")
