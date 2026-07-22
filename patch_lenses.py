import os
import glob

lens_dir = 'src/agents/modular_pipeline/lenses/'
lenses = glob.glob(os.path.join(lens_dir, '*_lens.py'))

for lens in lenses:
    with open(lens, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'self.llm_client' in content or 'base_lens' not in content:
        continue
        
    print(f'Patching {lens}...')
    
    # 1. Fix imports
    content = content.replace('from src.agents.modular_pipeline.base_lens import BaseLens', 'from src.agents.modular_pipeline.base_lens import BaseLens, LensAnalysisResult\nfrom src.agents.archive.cognitive_extractor import get_cognitive_prompt')
    
    # 2. Find async def analyze
    analyze_idx = content.find('async def analyze(self, source_text: str, current_state: Dict[str, Any]) -> Dict[str, Any]:')
    if analyze_idx == -1:
        continue
        
    insert_idx = content.find('\"\"\"', analyze_idx)
    if insert_idx != -1:
        insert_idx = content.find('\"\"\"', insert_idx + 3) + 3
    else:
        insert_idx = content.find(':', analyze_idx) + 1
        
    # Find indentation
    next_line_idx = content.find('\n', insert_idx) + 1
    spaces = 0
    while content[next_line_idx + spaces] == ' ':
        spaces += 1
    
    indent = ' ' * spaces
    
    llm_code = f'''
{indent}if getattr(self, "llm_client", None):
{indent}    prompt = f"Analyze the following news from this lens perspective:\\n\\n{{source_text}}"
{indent}    system_prompt = get_cognitive_prompt()
{indent}    result = await self.llm_client.generate_structured(prompt=prompt, response_model=LensAnalysisResult, system_prompt=system_prompt)
{indent}    if result: return result.model_dump()
'''
    
    content = content[:next_line_idx] + llm_code + content[next_line_idx:]
    
    with open(lens, 'w', encoding='utf-8') as f:
        f.write(content)
    
print('Done!')
