Run SonarSource/sonarqube-scan-action@7006c4492b2e0ee0f816d36501671557c97f5995
Installing Sonar Scanner CLI 8.1.0.6389 for windows-x64...
Downloading from: https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-8.1.0.6389-windows-x64.zip
Downloading signature from: https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-8.1.0.6389-windows-x64.zip.asc
Importing SonarSource public key from hkps://keyserver.ubuntu.com...
"C:\Program Files\Git\usr\bin\gpg.exe" --homedir /d/a/_temp/gpg-home-1780768943132-3784 --batch --keyserver hkps://keyserver.ubuntu.com --recv-keys 679F1EE92B19609DE816FDE81DB198F93525EC1A
gpg: keybox '/d/a/_temp/gpg-home-1780768943132-3784/pubring.kbx' created
gpg: /d/a/_temp/gpg-home-1780768943132-3784/trustdb.gpg: trustdb created
gpg: key 1DB198F93525EC1A: public key "SonarSource S.A. <infra@sonarsource.com>" imported
gpg: Total number processed: 1
gpg:               imported: 1
Successfully imported key from hkps://keyserver.ubuntu.com
✓ SonarSource public key imported successfully
Verifying GPG signature...
"C:\Program Files\Git\usr\bin\gpg.exe" --homedir /d/a/_temp/gpg-home-1780768943132-3784 --batch --verify /d/a/_temp/7d38a20c-a20b-4378-92f4-8d8c0dbcddad /d/a/_temp/e4309cdc-d2c1-41be-8825-41ec29de8aa4
gpg: Signature made Tue Apr 21 07:20:27 2026 CUT
gpg:                using RSA key D1436C0DBACEA48702AF97C363F1DD7753B8B315
gpg: Good signature from "SonarSource S.A. <infra@sonarsource.com>" [unknown]
gpg: WARNING: This key is not certified with a trusted signature!
gpg:          There is no indication that the signature belongs to the owner.
Primary key fingerprint: 679F 1EE9 2B19 609D E816  FDE8 1DB1 98F9 3525 EC1A
     Subkey fingerprint: D143 6C0D BACE A487 02AF  97C3 63F1 DD77 53B8 B315
✓ GPG signature verification passed
"C:\Program Files\PowerShell\7\pwsh.exe" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Unrestricted -Command "$ErrorActionPreference = 'Stop' ; try { Add-Type -AssemblyName System.IO.Compression.ZipFile } catch { } ; try { [System.IO.Compression.ZipFile]::ExtractToDirectory('D:\a\_temp\e4309cdc-d2c1-41be-8825-41ec29de8aa4', 'D:\a\_temp\9df0be46-f5e5-4281-a891-19e1d13d3771', $true) } catch { if (($_.Exception.GetType().FullName -eq 'System.Management.Automation.MethodException') -or ($_.Exception.GetType().FullName -eq 'System.Management.Automation.RuntimeException') ){ Expand-Archive -LiteralPath 'D:\a\_temp\e4309cdc-d2c1-41be-8825-41ec29de8aa4' -DestinationPath 'D:\a\_temp\9df0be46-f5e5-4281-a891-19e1d13d3771' -Force } else { throw $_ } } ;"
Sonar Scanner CLI cached to: C:\hostedtoolcache\windows\sonar-scanner-cli\8.1.0.6389\windows-x64
C:\Windows\system32\cmd.exe /D /S /C "C:\hostedtoolcache\windows\sonar-scanner-cli\8.1.0.6389\windows-x64\bin\sonar-scanner.bat "-Dsonar.projectBaseDir=.""
18:02:39.039 INFO  Scanner configuration file: C:\hostedtoolcache\windows\sonar-scanner-cli\8.1.0.6389\windows-x64\bin\..\conf\sonar-scanner.properties
18:02:39.044 INFO  Project root configuration file: D:\a\trading_project\trading_project\sonar-project.properties
18:02:39.063 INFO  SonarScanner CLI 8.1.0.6389
18:02:39.075 INFO  Windows Server 2025 10.0 amd64
18:02:48.267 INFO  Communicating with SonarQube Cloud
18:02:48.267 INFO  JRE provisioning: os[windows], arch[amd64]
18:02:54.768 INFO  Starting SonarScanner Engine...
18:02:54.770 INFO  Java 21.0.9 Eclipse Adoptium (64-bit)
18:02:59.300 INFO  Load global settings
18:03:00.064 INFO  Load global settings (done) | time=752ms
18:03:00.178 INFO  Server id: 1BD809FA-AWHW8ct9-T_TB3XqouNu
18:03:01.023 INFO  Loading required plugins
18:03:01.025 INFO  Load plugins index
18:03:01.176 INFO  Load plugins index (done) | time=152ms
18:03:01.178 INFO  Load/download plugins
18:03:01.908 INFO  Load/download plugins (done) | time=731ms
18:03:02.162 INFO  Loaded core extensions: architecture, a3s, sca
18:03:02.571 INFO  Process project properties
18:03:02.606 INFO  Project key: OleksanderSS_trading_project
18:03:02.606 INFO  Base dir: D:\a\trading_project\trading_project
18:03:02.607 INFO  Working dir: D:\a\trading_project\trading_project\.scannerwork
18:03:02.619 INFO  Load project settings for component key: 'OleksanderSS_trading_project'
18:03:02.988 INFO  Load project settings for component key: 'OleksanderSS_trading_project' (done) | time=368ms
18:03:03.005 INFO  Found an active CI vendor: 'Github Actions'
18:03:04.056 INFO  Check ALM binding of project 'OleksanderSS_trading_project'
18:03:04.282 INFO  Detected project binding: BOUND
18:03:04.283 INFO  Check ALM binding of project 'OleksanderSS_trading_project' (done) | time=225ms
18:03:04.288 INFO  Load branch configuration
18:03:04.289 INFO  Github event: push
18:03:04.295 INFO  Auto-configuring branch production
18:03:04.296 INFO  Load branch configuration (done) | time=7ms
18:03:04.302 INFO  Create analysis
18:03:04.671 INFO  Branch name: production, type: long
18:03:04.671 INFO  Create analysis (done) | time=368ms
18:03:04.937 INFO  Load quality profiles
18:03:05.397 INFO  Load quality profiles (done) | time=459ms
18:03:06.156 INFO  Load active rules
18:03:07.474 INFO  Load active rules (done) | time=1319ms
18:03:08.019 INFO  Organization key: oleksanderss
18:03:08.067 INFO  Preprocessing files...
18:03:09.956 INFO  3 languages detected in 653 preprocessed files (done) | time=1887ms
18:03:09.956 INFO  23 files ignored because of inclusion/exclusion patterns
18:03:09.956 INFO  0 files ignored because of scm ignore settings
18:03:12.328 INFO  Loading plugins for detected languages
18:03:12.329 INFO  Load/download plugins
18:03:13.934 INFO  Load/download plugins (done) | time=1605ms
18:03:14.262 INFO  Load project repositories
18:03:15.703 INFO  Load project repositories (done) | time=1440ms
18:03:15.728 INFO  Indexing files...
18:03:15.729 INFO  Project configuration:
18:03:15.729 INFO    Excluded sources: **/__pycache__/**, **/*.pyc, **/.ipynb_checkpoints/**, **/*.ipynb, **/.gemini/**, **/venv/**, **/.venv/**, src/colab/**, src/devtools/**, src/trained_models/**, **/build-wrapper-dump.json
18:03:15.796 INFO  653 files indexed (done) | time=67ms
18:03:15.799 INFO  Quality profile for json: Sonar way
18:03:15.800 INFO  Quality profile for py: Sonar way
18:03:15.800 INFO  Quality profile for yaml: Sonar way
18:03:15.801 INFO  ------------- Run sensors on module OleksanderSS_trading_project
18:03:15.896 INFO  Load metrics repository
18:03:16.020 INFO  Load metrics repository (done) | time=123ms
18:03:16.033 INFO  Sensor cache enabled
18:03:16.040 INFO  Load sensor cache
18:03:17.831 INFO  Load sensor cache (11 MB) | time=1791ms
18:03:19.450 INFO  Sensor Python Sensor [python]
18:03:21.006 WARN  Access to the multi-values/property set property 'sonar.test.inclusions' should be made using 'getStringArray' method. The SonarQube plugin using this property should be updated.
18:03:21.006 WARN  Access to the multi-values/property set property 'sonar.test.exclusions' should be made using 'getStringArray' method. The SonarQube plugin using this property should be updated.
18:03:21.015 WARN  Access to the multi-values/property set property 'sonar.test.inclusions' should be made using 'getStringArray' method. The SonarQube plugin using this property should be updated.
18:03:21.016 WARN  Access to the multi-values/property set property 'sonar.test.exclusions' should be made using 'getStringArray' method. The SonarQube plugin using this property should be updated.
18:03:21.053 INFO  Starting global symbols computation
18:03:21.059 INFO  587 source files to be analyzed
18:03:30.540 INFO  587/587 source files have been analyzed
18:03:30.542 INFO  Finished step global symbols computation in 9483ms
18:03:30.978 INFO  Starting rules execution
18:03:30.980 INFO  587 source files to be analyzed
18:03:37.231 INFO  No boundary descriptors defined
18:03:40.998 INFO  23/587 files analyzed, current files: diary_engine.py, elite_risk_metrics.py, config.py, ...
18:03:51.013 INFO  141/587 files analyzed, current files: constraint_engine.py, news_quality_enricher.py, heavy_light_comparator.py, ...
18:03:57.399 WARN  SonarPython detected files that look like test code but 'sonar.tests' is not configured. Rules targeting production code were not executed on these files. Configure 'sonar.tests' in your project properties for a more accurate analysis.
18:04:01.033 INFO  297/587 files analyzed, current files: loader.py, correlation_visualizer.py, local_file_collector.py, ...
18:04:09.773 ERROR Unable to parse file: src/pipeline/stages/stage_7_evaluation.py
18:04:09.785 ERROR Parse error at line 193 column 20:

  187: 
  188: 
  189:             if 'total_return_pct' in financial_metrics:
  190:                 stress_results['scenarios']['high_volatility'] = {
  191:                     'description': 'Portfolio performance under high volatility conditions',
  192:                     'impact': financial_metrics['total_return_pct'] * 0.5
  -->                      'status': 'passed' if financial_metrics['total_return_pct'] > 0 else 'failed'
  194:                 }
  195: 
  196: 
  197:             if 'max_drawdown_pct' in financial_metrics:
  198:                 stress_results['scenarios']['market_crash'] =

18:04:09.844 ERROR Working directory is null, cannot save UDG
18:04:11.076 INFO  490/587 files analyzed, current files: data_generator.py, data_manager.py, risk_decomposition_analyzer.py, ...
18:04:14.542 INFO  587/587 source files have been analyzed
18:04:14.543 INFO  Finished step rules execution in 43563ms
18:04:14.543 INFO  The Python analyzer was able to leverage cached data from previous analyses for 0 out of 587 files. These files were not parsed.
18:04:14.547 INFO  Sensor Python Sensor [python] (done) | time=55095ms
18:04:14.549 INFO  Sensor Cobertura Sensor for Python coverage [python]
18:04:15.369 INFO  Sensor Cobertura Sensor for Python coverage [python] (done) | time=820ms
18:04:15.370 INFO  Sensor PythonXUnitSensor [python]
18:04:16.054 INFO  Sensor PythonXUnitSensor [python] (done) | time=683ms
18:04:16.055 INFO  Sensor Python Dependency Sensor [python]
18:04:16.061 INFO  Sensor Python Dependency Sensor [python] (done) | time=6ms
18:04:16.062 INFO  Sensor SecurityPythonTelemetrySensor [securitypythonfrontend]
18:04:16.062 INFO  Sensor SecurityPythonTelemetrySensor [securitypythonfrontend] (done) | time=0ms
18:04:16.062 INFO  Sensor Python HTML templates processing [securitypythonfrontend]
18:04:16.086 INFO  HTML files are not indexed : you may want to add them in the scanned files of this project to detect Python XSS vulnerabilities
18:04:16.086 INFO  Sensor Python HTML templates processing [securitypythonfrontend] (done) | time=23ms
18:04:16.086 INFO  Sensor IaC CloudFormation Sensor [iac]
18:04:16.121 INFO  There are no files to be analyzed for the CloudFormation language
18:04:16.122 INFO  Sensor IaC CloudFormation Sensor [iac] (done) | time=34ms
18:04:16.125 INFO  Sensor IaC cfn-lint report Sensor [iac]
18:04:16.126 INFO  Sensor IaC cfn-lint report Sensor [iac] (done) | time=1ms
18:04:16.127 INFO  Sensor IaC Kustomization Sensor [iac]
18:04:16.131 INFO  Sensor IaC Kustomization Sensor [iac] (done) | time=3ms
18:04:16.132 INFO  Sensor IaC hadolint report Sensor [iac]
18:04:16.132 INFO  Sensor IaC hadolint report Sensor [iac] (done) | time=0ms
18:04:16.136 INFO  Sensor IaC Azure Resource Manager Sensor [iac]
18:04:16.145 INFO  There are no files to be analyzed for the Azure Resource Manager language
18:04:16.145 INFO  Sensor IaC Azure Resource Manager Sensor [iac] (done) | time=8ms
18:04:16.146 INFO  Sensor Java Config Sensor [iac]
18:04:16.159 INFO  There are no files to be analyzed for the Java language
18:04:16.159 INFO  Sensor Java Config Sensor [iac] (done) | time=13ms
18:04:16.159 INFO  Sensor IaC Docker Sensor [iac]
18:04:16.177 INFO  There are no files to be analyzed for the Docker language
18:04:16.178 INFO  Sensor IaC Docker Sensor [iac] (done) | time=17ms
18:04:16.178 INFO  Sensor IaC Ansible Sensor [iac]
18:04:16.218 INFO  There are no files to be analyzed for the Ansible language
18:04:16.218 INFO  Sensor IaC Ansible Sensor [iac] (done) | time=40ms
18:04:16.218 INFO  Sensor IaC ansible-lint report Sensor [iac]
18:04:16.218 INFO  Sensor IaC ansible-lint report Sensor [iac] (done) | time=1ms
18:04:16.219 INFO  Sensor IaC spectral report Sensor [iac]
18:04:16.220 INFO  Sensor IaC spectral report Sensor [iac] (done) | time=1ms
18:04:16.221 INFO  Sensor IaC GitHub Actions Sensor [iac]
18:04:16.222 INFO  There are no files to be analyzed for the GitHub Actions language
18:04:16.222 INFO  Sensor IaC GitHub Actions Sensor [iac] (done) | time=0ms
18:04:16.223 INFO  Sensor IaC actionlint report Sensor [iac]
18:04:16.224 INFO  Sensor IaC actionlint report Sensor [iac] (done) | time=0ms
18:04:16.225 INFO  Sensor IaC Azure Pipelines Sensor [iac]
18:04:16.239 INFO  There are no files to be analyzed for the Azure Pipelines language
18:04:16.239 INFO  Sensor IaC Azure Pipelines Sensor [iac] (done) | time=13ms
18:04:16.243 INFO  Sensor IaC Shell Sensor [iac]
18:04:16.245 INFO  There are no files to be analyzed for the Shell language
18:04:16.249 INFO  Sensor IaC Shell Sensor [iac] (done) | time=1ms
18:04:16.249 INFO  Sensor JavaScript/TypeScript/CSS analysis [javascript]
18:04:16.290 INFO  No input files found for analysis
18:04:16.293 INFO  Hit the cache for 0 out of 0
18:04:16.295 INFO  Miss the cache for 0 out of 0
18:04:16.295 INFO  Sensor JavaScript/TypeScript/CSS analysis [javascript] (done) | time=49ms
18:04:16.296 INFO  Sensor IaC Kubernetes Sensor [iac]
18:04:16.375 INFO  There are no files to be analyzed for the Kubernetes language
18:04:16.375 INFO  Sensor IaC Kubernetes Sensor [iac] (done) | time=78ms
18:04:16.375 INFO  Sensor IaC YAML Sensor [iac]
18:04:16.376 INFO  Sensor for language "YAML" is enabled by a feature flag. You can disable it by setting "sonar.yaml.activate" to false.
18:04:16.390 INFO  27 source files to be analyzed
18:04:16.649 INFO  27/27 source files have been analyzed
18:04:16.650 INFO  Sensor IaC YAML Sensor [iac] (done) | time=275ms
18:04:16.651 INFO  Sensor IaC JSON Sensor [iac]
18:04:16.651 INFO  Sensor for language "JSON" is enabled by a feature flag. You can disable it by setting "sonar.json.activate" to false.
18:04:16.656 INFO  3 source files to be analyzed
18:04:16.664 INFO  3/3 source files have been analyzed
18:04:16.664 INFO  Sensor IaC JSON Sensor [iac] (done) | time=13ms
18:04:16.665 INFO  Sensor Serverless configuration file sensor [security]
18:04:16.666 INFO  0 Serverless function entries were found in the project
18:04:16.674 INFO  0 Serverless function handlers were kept as entrypoints
18:04:16.675 INFO  Sensor Serverless configuration file sensor [security] (done) | time=8ms
18:04:16.676 INFO  Sensor AWS SAM template file sensor [security]
18:04:16.690 INFO  Sensor AWS SAM template file sensor [security] (done) | time=16ms
18:04:16.691 INFO  Sensor Generic Coverage Report
18:04:16.692 INFO  Parsing D:\a\trading_project\trading_project\coverage.xml
18:04:17.333 ERROR Error during parsing of the generic coverage report 'D:\a\trading_project\trading_project\coverage.xml'. Look at SonarQube documentation to know the expected XML format.
18:04:17.742 INFO  EXECUTION FAILURE
18:04:17.742 INFO  Total time: 1:38.706s
Error: Action failed: The process 'C:\hostedtoolcache\windows\sonar-scanner-cli\8.1.0.6389\windows-x64\bin\sonar-scanner.bat' failed with exit code 1