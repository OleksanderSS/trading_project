{ pkgs, ... }: {
  channel = "unstable";

  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.gcc
    pkgs.gfortran
    pkgs.pkg-config
    pkgs.git
  ];

  idx = {
    extensions = [ "ms-python.python" ];
    workspace = {
      # Use onCreate for initial setup
      onCreate = {
        # Install ONLY light dependencies for development
        install-deps = "pip install pandas numpy requests feedparser fredapi yfinance scikit-learn";
      };
    };
  };
}
