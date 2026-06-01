import asyncio
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.config.unified_config_manager import UnifiedConfigManager
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
import logging

# Disable logging to keep output clean
logging.getLogger().setLevel(logging.WARNING)

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

async def audit_stages_with_timeout(orchestrator, timeout_seconds=30):
    """Audit stages with timeout to prevent hangs"""
    async def _audit_stages():
        # Handle both list and dict based storage for stages
        stages = orchestrator.stages 
        stages_dict = {f"{i}_{s.__class__.__name__}": s for i, s in enumerate(stages)} if isinstance(stages, list) else stages

        for name, obj in stages_dict.items():
            print(f"\nStage: {name}")
            keywords = ['manager', 'enricher', 'guard', 'monitor', 'analyzer', 'orchestrator']
            attrs = [a for a in dir(obj) if any(k in a.lower() for k in keywords)]
            found_any = False
            for attr in attrs:
                try:
                    val = getattr(obj, attr)
                    if val is not None and not isinstance(val, (type, str, int, float, bool)):
                        print(f"  - {attr}: {type(val).__name__}")
                        found_any = True
                except Exception as e:
                    logging.warning(f"Error accessing {attr} on {name}: {e}")
            if not found_any:
                print("  - No obvious tool components found.")
    
    try:
        await asyncio.wait_for(_audit_stages(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print("  ❌ Stage audit timed out after 30 seconds")
    except Exception as e:
        print(f"  ❌ Stage audit failed: {e}")

async def run_audit():
    print("=== PIPELINE OPERATIONAL COVERAGE AUDIT ===")
    config_manager = UnifiedConfigManager()
    
    # 1. Pipeline Stages Audit (with timeout)
    print_section("STAGE COMPONENTS")
    orchestrator = PipelineOrchestrator(config_manager)
    await audit_stages_with_timeout(orchestrator, timeout_seconds=30)

    # 2. Analytics Engine Audit
    print_section("ANALYTICS ENGINE")
    try:
        engine = UnifiedAnalyticsEngine(config_manager=config_manager)
        print(f"Total Analyzers Active: {len(engine.analyzers)}")
        for name in sorted(engine.analyzers.keys()):
            print(f"  - {name}")
    except Exception as e:
        print(f"  ❌ AnalyticsEngine failed to initialize: {e}")

if __name__ == "__main__":
    asyncio.run(run_audit())
