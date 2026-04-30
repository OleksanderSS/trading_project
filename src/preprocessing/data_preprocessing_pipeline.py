# DataPreprocessingPipeline Implementation
# Reusable preprocessing with fit/transform separation for training/inference

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import pickle
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DataPreprocessingPipeline")


class DataPreprocessingPipeline:
    """
    Implements a scikit-learn-like fit/transform pipeline for data preprocessing.
    
    Advantages:
    - Logic separation: fit() during training, transform() during inference
    - State retention of preprocessing parameters (scalers, encoders, etc.)
    - Export/Import capabilities for accurate reproducibility
    - Sequential application of execution steps (fit_transform)
    """

    def __init__(self, project_path=None, name="default"):
        """
        Args:
            project_path: Base directory for saving artifacts
            name: Identifier for the preprocessing pipeline
        """
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.name = name
        self.logger = ProjectLogger.get_logger(f"DataPreprocessingPipeline.{name}")
        self.steps = []
        self.is_fitted = False
        
        # Artifacts (Cached during fit)
        self.numeric_features = []
        self.categorical_features = []
        self.numeric_scaler = None  # StandardScaler
        self.categorical_encoders = {}  # Dict of LabelEncoders
        self.feature_names = []
        self.preprocessing_config = {}
        
    def add_step(self, step_name, step_config):
        """
        Appends a preprocessing step to the execution pipeline.
        
        Args:
            step_name: Name of the step ('drop_na', 'scale_numeric', 'encode_categorical', etc.)
            step_config: Configuration dict mapping operational parameters
        """
        self.steps.append({
            'name': step_name,
            'config': step_config,
            'fitted': False
        })
        return self

    def fit(self, X):
        """
        Fits the pipeline utilizing the training dataset context.
        
        Args:
            X: Training features (DataFrame or structured array)
            y: Target variable (optional)
        
        Returns:
            self
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        self.logger.info(f"📊 Fitting preprocessing pipeline '{self.name}'...")
        self.logger.info(f"   Input shape: {X.shape}")
        
        # Step 1: Isolate numeric and categorical feature columns
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.feature_names = X.columns.tolist()
        
        self.logger.info(f"   Numeric features: {len(self.numeric_features)}")
        self.logger.info(f"   Categorical features: {len(self.categorical_features)}")
        
        # Step 2: Initialize numeric scaling
        if self.numeric_features:
            try:
                from sklearn.preprocessing import StandardScaler
                self.numeric_scaler = StandardScaler()
                X[self.numeric_features] = self.numeric_scaler.fit_transform(X[self.numeric_features])
                self.logger.info(f"   ✅ Scaled {len(self.numeric_features)} numeric features")
            except Exception as e:
                self.logger.warning(f"   ⚠️ Numeric scaling failed: {e}")
        
        # Step 3: Initialize categorical encoding parameters
        if self.categorical_features:
            try:
                from sklearn.preprocessing import LabelEncoder
                for cat_col in self.categorical_features:
                    le = LabelEncoder()
                    X[cat_col] = le.fit_transform(X[cat_col].astype(str))
                    self.categorical_encoders[cat_col] = le
                self.logger.info(f"   ✅ Encoded {len(self.categorical_features)} categorical features")
            except Exception as e:
                self.logger.warning(f"   ⚠️ Categorical encoding failed: {e}")
        
        # Step 4: Neutralize NaN vectors
        X = X.fillna(0)
        self.logger.info("   ✅ Filled NaN values")
        
        # Step 5: Clean critical inf vectors
        X = X.replace([np.inf, -np.inf], 0)
        self.logger.info("   ✅ Replaced inf values")
        
        self.is_fitted = True
        self.preprocessing_config = {
            'fit_timestamp': datetime.now().isoformat(),
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'n_numeric': len(self.numeric_features),
            'n_categorical': len(self.categorical_features)
        }
        
        self.logger.info("   ✅ Pipeline fitted successfully")
        return self

    def _validate_fitted_state(self) -> None:
        """Validate pipeline is fitted before transformation."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted yet. Execute fit() prior to calling transform().")
    
    def _convert_to_dataframe(self, X) -> pd.DataFrame:
        """Convert input to DataFrame if needed."""
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        return X.copy()
    
    def _apply_numeric_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply numeric scaling to features."""
        if not (self.numeric_features and self.numeric_scaler):
            return X
            
        try:
            X[self.numeric_features] = self.numeric_scaler.transform(X[self.numeric_features])
            return X
        except Exception as e:
            self.logger.warning(f"⚠️ Numeric scaling during transform failed: {e}")
            return X
    
    def _apply_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding to features."""
        if not self.categorical_features:
            return X
            
        try:
            for cat_col in self.categorical_features:
                if cat_col in self.categorical_encoders:
                    encoder = self.categorical_encoders[cat_col]
                    X[cat_col] = X[cat_col].astype(str)
                    try:
                        X[cat_col] = encoder.transform(X[cat_col])
                    except ValueError:
                        # Safely handle unobserved categorical dimensions
                        current_encoder = encoder
                        X[cat_col] = X[cat_col].apply(
                            lambda x, enc=current_encoder: enc.transform([x])[0] if x in enc.classes_ 
                            else enc.transform([enc.classes_[0]])[0]
                        )
        except Exception as e:
            self.logger.warning(f"⚠️ Categorical encoding transformation failed: {e}")
        
        return X
    
    def _clean_final_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean NaN and infinite values from final data."""
        X = X.fillna(0)
        return X.replace([np.inf, -np.inf], 0)
    
    def transform(self, X):
        """
        Applies the parameters of the fitted pipeline to an unobserved dataset.
        
        Args:
            X: Input data to be dimensionally and categorically transformed
        
        Returns:
            Transformed features DataFrame
        """
        self._validate_fitted_state()
        X = self._convert_to_dataframe(X)
        X = self._apply_numeric_scaling(X)
        X = self._apply_categorical_encoding(X)
        X = self._clean_final_data(X)
        return X

    def fit_transform(self, X, y=None):
        """
        Executes sequential fit and transform procedures.
        
        Args:
            X: Input features
            y: Target variable (optional)
        
        Returns:
            Transformed data arrays
        """
        return self.fit(X, y).transform(X)

    def save(self, filepath=None):
        """
        Persists the fitted pipeline state and parameters to disk.
        
        Args:
            filepath: Destination file path (defaults to project_path + artifact name)
        """
        if filepath is None:
            filepath = self.project_path / f"preprocessing_pipeline_{self.name}.pkl"
        
        pipeline_data = {
            'name': self.name,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'numeric_scaler': self.numeric_scaler,
            'categorical_encoders': self.categorical_encoders,
            'feature_names': self.feature_names,
            'preprocessing_config': self.preprocessing_config,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        self.logger.info(f"💾 Pipeline artifact secured at {filepath}")
        return filepath

    def load(self, filepath):
        """
        Restores a fitted pipeline state from a static file path.
        
        Args:
            filepath: Target file path locating the saved pipeline configuration
        """
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.name = pipeline_data['name']
        self.numeric_features = pipeline_data['numeric_features']
        self.categorical_features = pipeline_data['categorical_features']
        self.numeric_scaler = pipeline_data['numeric_scaler']
        self.categorical_encoders = pipeline_data['categorical_encoders']
        self.feature_names = pipeline_data['feature_names']
        self.preprocessing_config = pipeline_data['preprocessing_config']
        self.is_fitted = pipeline_data['is_fitted']
        
        self.logger.info(f"✅ Pipeline structural state restored from: {filepath}")
        return self

    def get_config(self):
        """Returns the configuration mapping for the current pipeline state."""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'config': self.preprocessing_config,
            'n_features': len(self.feature_names)
        }

    def save_config(self, filepath: str = None) -> None:
        """
        Exports the current configuration payload to JSON.
        
        Args:
            filepath: Destination file path (defaults to project_path + artifact name)
        """
        if filepath is None:
            filepath = self.project_path / f"preprocessing_config_{self.name}.json"
        
        config = {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'preprocessing_config': self.preprocessing_config,
            'feature_names': self.feature_names,
            'n_numeric': len(self.numeric_features),
            'n_categorical': len(self.categorical_features),
            'saved_time': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"💾 Pipeline JSON config artifacts exported to {filepath}")
        return filepath


# EXAMPLE USAGE
if __name__ == "__main__":
    from src.core.logging.logger import ProjectLogger
    from src.config.unified_config_manager import get_current_config
    
    logger = ProjectLogger.get_logger("PreprocessingExample")
    logger.info("=== DataPreprocessingPipeline Example ===\n")
    
    # Internal validation logic
    config = get_current_config()
    seed = config.get('performance.random_seed', 42)
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame({
        'numeric_1': rng.standard_normal(100) * 100,
        'numeric_2': rng.standard_normal(100) * 50,
        'categorical_1': rng.choice(['A', 'B', 'C'], 100),
        'categorical_2': rng.choice(['X', 'Y'], 100)
    })
    
    X_test = pd.DataFrame({
        'numeric_1': rng.standard_normal(20) * 100,
        'numeric_2': rng.standard_normal(20) * 50,
        'categorical_1': rng.choice(['A', 'B', 'C'], 20),
        'categorical_2': rng.choice(['X', 'Y'], 20)
    })
    
    logger.info("Original training data:")
    logger.info(f"\n{X_train.head(3)}")
    logger.info(f"Shape: {X_train.shape}\n")
    
    # Initialization and fitting process
    pipeline = DataPreprocessingPipeline(name="test_pipeline")
    X_train_transformed = pipeline.fit_transform(X_train)
    
    logger.info("\nTransformed training data:")
    logger.info(f"\n{X_train_transformed.head(3)}")
    
    # Transform test set
    X_test_transformed = pipeline.transform(X_test)
    
    logger.info("\nTransformed test data:")
    logger.info(f"\n{X_test_transformed.head(3)}")
    
    # Preservation routines
    logger.info("\n" + "="*60)
    pipeline.save_config()
    logger.info(f"Pipeline config: {pipeline.get_config()}")
