{ pkgs, ... }: {
  channel = "unstable";

  packages = [
    pkgs.gcc
    pkgs.gfortran
    pkgs.pkg-config
    pkgs.git
    
    (pkgs.python312.withPackages (ps: with ps; [
      # Базові пакети
      pandas
      numpy

      # Основні бібліотеки
      "scikit-learn"
      lightgbm
      xgboost
      matplotlib
      seaborn
      requests
      joblib

      # Джерела даних
      yfinance
      "pandas-datareader"
      "google-cloud-bigquery"
      "pandas-gbq"

      # Залежності, що виявились
      holidays
      transformers
      torch
      "python-dotenv"
      "feedparser"
      "fredapi"

      pip
    ]))
  ];

  idx = {
    extensions = [ "ms-python.python" ];
    workspace = {};
  };
}
