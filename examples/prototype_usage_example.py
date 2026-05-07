"""
Example: Using Prototypes and Enhanced Factory

Demonstrates how to use ModelPrototype, PrototypeRegistry, and EnhancedModelFactory
in a practical pipeline.
"""

import asyncio
from pathlib import Path

from src.models.prototypes.prototype import ModelPrototype
from src.models.prototypes.registry import get_prototype_registry
from src.models.enhanced_factory import EnhancedModelFactory
from src.models.tree.catboost_model import CatBoostModel
from src.models.tree.lightgbm_model import LightGBMModel
from src.models.tree.xgboost_model import XGBoostModel
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def setup_prototypes():
    """Setup model prototypes"""
    logger.info("🔧 Setting up prototypes...")

    # Initialize registry
    EnhancedModelFactory.initialize_registry("data/prototypes/registry.json")

    # Create prototypes for tree-based models
    prototypes = [
        ModelPrototype(
            model_id="catboost_v1",
            model_class=CatBoostModel,
            version="1.0.0",
            dependencies=["catboost", "numpy"],
            metadata={
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "verbose": False,
            },
        ),
        ModelPrototype(
            model_id="lightgbm_v1",
            model_class=LightGBMModel,
            version="1.0.0",
            dependencies=["lightgbm", "numpy"],
            metadata={
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "verbose": -1,
            },
        ),
        ModelPrototype(
            model_id="xgboost_v1",
            model_class=XGBoostModel,
            version="1.0.0",
            dependencies=["xgboost", "numpy"],
            metadata={
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "verbosity": 0,
            },
        ),
    ]

    # Register prototypes
    for proto in prototypes:
        EnhancedModelFactory.register_prototype(proto)
        logger.info(f"✅ Registered: {proto.model_id}")

    return prototypes


def example_basic_cloning():
    """Example 1: Basic prototype cloning"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 1: Basic Prototype Cloning")
    logger.info("=" * 60)

    registry = get_prototype_registry()

    # Get prototype
    proto = registry.get("catboost_v1")
    if not proto:
        logger.error("Prototype not found!")
        return

    logger.info(f"📦 Prototype: {proto}")
    logger.info(f"   Version: {proto.version}")
    logger.info(f"   Metadata: {proto.metadata}")

    # Clone with default metadata
    logger.info("\n🔄 Cloning with default metadata...")
    model1 = proto.clone()
    logger.info(f"✅ Model 1 created: {model1}")

    # Clone with overrides
    logger.info("\n🔄 Cloning with parameter overrides...")
    model2 = proto.clone(iterations=200, depth=8)
    logger.info(f"✅ Model 2 created: {model2}")

    model3 = proto.clone(learning_rate=0.05)
    logger.info(f"✅ Model 3 created: {model3}")

    logger.info(f"\n📊 Prototype stats: {proto.get_info()}")


def example_registry_operations():
    """Example 2: Registry operations"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Registry Operations")
    logger.info("=" * 60)

    registry = get_prototype_registry()

    # List all prototypes
    logger.info("\n📋 All registered prototypes:")
    for model_id in registry.list_all():
        logger.info(f"   - {model_id}")

    # Filter by type
    logger.info("\n🔍 Tree-based models:")
    tree_models = registry.get_by_type("catboost")
    for proto in tree_models:
        logger.info(f"   - {proto.model_id} (v{proto.version})")

    # Clone from registry
    logger.info("\n🔄 Cloning from registry...")
    model = registry.clone("lightgbm_v1", n_estimators=500)
    logger.info(f"✅ Model created: {model}")

    # Get statistics
    logger.info("\n📊 Registry statistics:")
    stats = registry.get_stats()
    logger.info(f"   Total prototypes: {stats['total_prototypes']}")
    logger.info(f"   Total clones: {stats['total_clones']}")
    logger.info(f"   Avg clones per prototype: {stats['avg_clones_per_prototype']:.1f}")


def example_enhanced_factory():
    """Example 3: Enhanced factory usage"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Enhanced Factory Usage")
    logger.info("=" * 60)

    # Get model via prototype
    logger.info("\n🏭 Getting model via prototype...")
    model1 = EnhancedModelFactory.get_model("catboost_v1", iterations=300)
    logger.info(f"✅ Model 1: {model1}")

    # Get model via legacy factory (backward compatible)
    logger.info("\n🏭 Getting model via legacy factory...")
    model2 = EnhancedModelFactory.get_model("catboost", iterations=100)
    logger.info(f"✅ Model 2: {model2}")

    # Get available models
    logger.info("\n📋 Available models:")
    models = EnhancedModelFactory.get_available_models()
    for model_id in sorted(models):
        logger.info(f"   - {model_id}")

    # Get factory statistics
    logger.info("\n📊 Factory statistics:")
    stats = EnhancedModelFactory.get_factory_stats()
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")


def example_performance_comparison():
    """Example 4: Performance comparison"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Performance Comparison")
    logger.info("=" * 60)

    import time

    registry = get_prototype_registry()
    proto = registry.get("catboost_v1")

    if not proto:
        logger.error("Prototype not found!")
        return

    # Measure cloning speed
    logger.info("\n⏱️ Measuring cloning speed...")
    n_clones = 100

    start = time.time()
    for _ in range(n_clones):
        model = proto.clone()
    elapsed = time.time() - start

    logger.info(f"   Cloned {n_clones} models in {elapsed:.3f}s")
    logger.info(f"   Average: {elapsed/n_clones*1000:.2f}ms per clone")
    logger.info(f"   Throughput: {n_clones/elapsed:.0f} clones/sec")


def example_export_summary():
    """Example 5: Export summary"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Export Summary")
    logger.info("=" * 60)

    # Export factory summary
    logger.info("\n📤 Factory summary:")
    summary = EnhancedModelFactory.export_summary()

    logger.info(f"   Local prototypes: {summary['local_prototypes']}")
    logger.info(f"   Available models: {len(summary['available_models'])}")
    logger.info(f"   Stats: {summary['stats']}")

    # Export registry summary
    registry = get_prototype_registry()
    logger.info("\n📤 Registry summary:")
    reg_summary = registry.export_summary()
    logger.info(f"   Total prototypes: {reg_summary['total_prototypes']}")
    logger.info(f"   Registry path: {reg_summary['registry_path']}")


async def main():
    """Main example execution"""
    logger.info("🚀 Starting Prototype Usage Examples")
    logger.info("=" * 60)

    # Setup prototypes
    setup_prototypes()

    # Run examples
    example_basic_cloning()
    example_registry_operations()
    example_enhanced_factory()
    example_performance_comparison()
    example_export_summary()

    logger.info("\n" + "=" * 60)
    logger.info("✅ All examples completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
