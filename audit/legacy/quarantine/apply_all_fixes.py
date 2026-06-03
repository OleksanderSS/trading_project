import ast
from pathlib import Path

import astor


class ExceptFixer(ast.NodeTransformer):
    def __init__(self, filename):
        self.filename = filename
        self.changed = False

    def visit_ExceptHandler(self, node):
        is_exception = (node.type is None or 
                        (isinstance(node.type, ast.Name) and node.type.id == 'Exception'))
        
        if is_exception:
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(ast.Module(body=node.body, type_ignores=[])))
            has_log = any(isinstance(n, ast.Call) and 
                          isinstance(n.func, ast.Attribute) and 
                          n.func.attr in ('error', 'exception', 'critical') 
                          for n in ast.walk(ast.Module(body=node.body, type_ignores=[])))
            
            if not has_raise and not has_log:
                exc_var = node.name if node.name else "e"
                
                log_call = ast.Expr(value=ast.Call(
                    func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='logger', ctx=ast.Load()), attr='error', ctx=ast.Load()),
                    args=[ast.JoinedStr(values=[
                        ast.Constant(value='Виникла помилка: '), 
                        ast.FormattedValue(value=ast.Name(id=exc_var, ctx=ast.Load()), conversion=-1)
                    ])],
                    keywords=[ast.keyword(arg='exc_info', value=ast.Constant(value=True))]
                ))
                
                raise_node = ast.Raise(exc=None, cause=None)
                
                node.body.insert(0, log_call)
                node.body.append(raise_node)
                self.changed = True
        
        return node

def apply_fix(file_path: Path):
    try:
        source = file_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        fixer = ExceptFixer(file_path.name)
        new_tree = fixer.visit(tree)
        
        if fixer.changed:
            fixed_source = astor.to_source(new_tree)
            file_path.write_text(fixed_source, encoding='utf-8')
            return True
    except Exception as e:
        print(f"Failed to fix {file_path}: {e}")
    return False

if __name__ == "__main__":
    count = 0
    for file_path in Path("src").rglob("*.py"):
        if apply_fix(file_path):
            print(f"Fixed: {file_path}")
            count += 1
    print(f"Total fixed: {count}")
