# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with proper module organization
- Configuration system via `config.py`
- README with comprehensive documentation
- MIT License
- GitHub Actions CI/CD workflow
- Requirements and project metadata files
- Changelog

### Changed
- Reorganized code into `src/dielectrics/` package structure
- Renamed `sensitivty_phantom4layer.py` to `sensitivity_analysis.py`
- Moved matplotlib backend from hardcoded to configurable
- Moved output paths to centralized configuration

### Fixed
- Filename typo: "sensitivty" → "sensitivity"

## [0.1.0] - 2025-06-09

### Added
- Transfer matrix method implementation for layered dielectrics
- Cole-Cole dispersion model support
- TE and TM polarization analysis
- APD (Absorbed Power Density) calculations
- Skin and phantom material models
- Broadband phantom analysis scripts
- Sensitivity and uncertainty analysis
