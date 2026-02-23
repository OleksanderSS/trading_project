# Experiments Module

## 🎯 **Purpose**
This module provides a framework for running experiments to optimize trading strategies, feature combinations, and model performance.

## 📁 **Structure**

### 📦 **Core Files:**
- **`experiment_base.py`** - Base class for all experiments
- **`experiment_config.py`** - Centralized configuration
- **`experiment_utils.py`** - Utility functions and visualizers
- **`compare_layers.py`** - Original layer comparison experiment
- **`improved_compare_layers.py`** - Enhanced version with parallel processing

## 🚀 **Key Features**

### 📊 **Base Experiment Class:**
- **Abstract interface** for consistent experiment structure
- **Automatic result saving** (CSV, JSON)
- **Report generation** with markdown output
- **Performance tracking** with timing and system stats
- **Error handling** and logging

### 🔧 **Configuration Management:**
- **Centralized settings** for all experiments
- **Output directory management** with timestamps
- **Parallel processing** configuration
- **Visualization settings** and styling

### 📈 **Enhanced Capabilities:**
- **Parallel execution** for faster results
- **Progress tracking** with checkpoints
- **Automatic visualizations** (plots, heatmaps)
- **Comprehensive reporting** with analysis
- **Performance monitoring** (CPU, memory usage)

## 🎯 **Usage Examples**

### 📦 **Basic Experiment:**
```python
from experiments import BaseExperiment

class MyExperiment(BaseExperiment):
    def run_experiment(self, **kwargs):
        # Your experiment logic here
        return results
    
    def get_metrics(self):
        return ["metric1", "metric2"]

# Run experiment
exp = MyExperiment()
results = exp.run(save_results=True, save_report=True)
```

### 🚀 **Enhanced Layer Comparison:**
```bash
# Run with parallel processing
python experiments/improved_compare_layers.py --days 365 --parallel --workers 8

# Generate visualizations
python experiments/improved_compare_layers.py --days 90 --visualize

# Custom output directory
python experiments/improved_compare_layers.py --output-dir experiments/my_results
```

### 📊 **Performance Tracking:**
```python
from experiments import PerformanceTracker

tracker = PerformanceTracker()
tracker.start()

# Your experiment code here
tracker.checkpoint("data_loaded")
tracker.checkpoint("processing_complete")

summary = tracker.get_summary()
print(f"Total time: {summary['total_time_formatted']}")
```

## 📈 **Output Structure**

### 📁 **Results Directory:**
```
experiments/results/
├── CompareLayers_20240108_120000/
│   ├── CompareLayers_20240108_120000.csv
│   ├── CompareLayers_20240108_120000.json
│   ├── CompareLayers_20240108_120000_enhanced_report.md
│   └── plots/
│       ├── CompareLayers_20240108_120000_performance.png
│       └── CompareLayers_20240108_120000_heatmap.png
└── experiment_summary.json
```

### 📊 **Generated Reports:**
- **CSV results** for data analysis
- **JSON results** for programmatic access
- **Markdown reports** for human reading
- **Visualization plots** (PNG, PDF)
- **Performance summaries** with system stats

## 🎯 **Best Practices**

### 📦 **Experiment Design:**
1. **Inherit from BaseExperiment** for consistency
2. **Implement required methods** (`run_experiment`, `get_metrics`)
3. **Use configuration** from `ExperimentConfig`
4. **Track performance** with `PerformanceTracker`

### 🔧 **Execution:**
1. **Use parallel processing** for large experiments
2. **Set appropriate timeouts** for long-running tasks
3. **Monitor system resources** during execution
4. **Save intermediate results** for long experiments

### 📊 **Analysis:**
1. **Generate visualizations** for better insights
2. **Compare multiple metrics** for comprehensive analysis
3. **Track best configurations** for production use
4. **Document findings** in reports

## 🚀 **Advanced Features**

### 📈 **Parallel Processing:**
- **Automatic CPU detection** for optimal worker count
- **Chunk-based processing** for memory efficiency
- **Progress tracking** with checkpoints
- **Error handling** for worker failures

### 📊 **Visualizations:**
- **Performance plots** by metric and configuration
- **Correlation heatmaps** for metric relationships
- **Best results charts** for top performers
- **Custom styling** with seaborn integration

### 📋 **Reporting:**
- **Enhanced markdown** with statistical analysis
- **Automatic insights** generation
- **Best configuration** identification
- **Performance statistics** with system metrics

## 🎯 **Integration with Main System**

### 📦 **Dependencies:**
- **core.pipeline_helpers** for pipeline execution
- **utils.metrics** for metric calculation
- **config.feature_layers** for layer definitions
- **utils.logger** for consistent logging

### 🔄 **Workflow:**
1. **Configure experiment** parameters
2. **Run experiment** with tracking
3. **Generate results** and visualizations
4. **Analyze findings** and identify best configurations
5. **Apply insights** to production system

## 📈 **Future Enhancements**

### 🚀 **Planned Features:**
- **Experiment scheduling** and automation
- **Result comparison** across multiple runs
- **Statistical significance** testing
- **Hyperparameter optimization** integration
- **Cloud execution** support for large experiments

### 📊 **Advanced Analytics:**
- **Time series analysis** of performance
- **Monte Carlo simulations** for strategy testing
- **Bayesian optimization** for parameter tuning
- **Ensemble methods** for combining results

## 🎯 **Getting Started**

### 📦 **Quick Start:**
```bash
# Run basic layer comparison
python experiments/compare_layers.py

# Run enhanced version
python experiments/improved_compare_layers.py --parallel --visualize

# Check configuration
python -c "from experiments import ExperimentConfig; print(ExperimentConfig.get_output_dir('test'))"
```

### 🔧 **Custom Experiment:**
1. **Create new class** inheriting from `BaseExperiment`
2. **Implement experiment logic** in `run_experiment`
3. **Define metrics** in `get_metrics`
4. **Run with tracking** and reporting

**The experiments module provides a complete framework for systematic trading strategy optimization!** 🎯
