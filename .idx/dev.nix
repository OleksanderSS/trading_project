{ pkgs, ... }: {
  channel = "stable-24.05";

  # Keep only essential system packages here
  packages = [
    pkgs.python312
    pkgs.gcc
    pkgs.google-cloud-sdk
  ];

  idx = {
    extensions = [ "ms-python.python" ];
    
    workspace = {
      # This script runs every time the workspace starts
      onStart = {
        setup = ''
          # Create a virtual environment if it doesn't exist
          if [ ! -d ".venv" ]; then
            ${pkgs.python312.interpreter} -m venv .venv
          fi
          
          # Activate the venv and install all python packages
          source .venv/bin/activate
          pip install --upgrade pip
          
          # Install all dependencies from requirements.txt
          pip install -r requirements.txt
        '';
      };
    };
  };
}
