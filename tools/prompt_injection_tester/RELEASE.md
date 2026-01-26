# Prompt Injection Tester - Release Guide

**Version**: 2.0.0
**Release Date**: 2026-01-26
**Architecture**: Sequential 4-Phase Pipeline

## Table of Contents

- [Overview](#overview)
- [Installation Methods](#installation-methods)
- [Docker Deployment](#docker-deployment)
- [Development Setup](#development-setup)
- [CI/CD Pipeline](#cicd-pipeline)
- [Release Checklist](#release-checklist)
- [Distribution](#distribution)
- [Upgrading](#upgrading)

## Overview

The Prompt Injection Tester (PIT) v2.0.0 is a production-ready security assessment tool featuring:

✅ Sequential 4-Phase Pipeline architecture
✅ 20+ built-in attack patterns
✅ Multi-format reporting (JSON, YAML, HTML)
✅ Docker containerization support
✅ CI/CD integration with GitHub Actions
✅ Comprehensive documentation

## Installation Methods

### Method 1: Install from Source (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/your-org/ai-llm-red-team-handbook.git
cd ai-llm-red-team-handbook/tools/prompt_injection_tester

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .

# Verify installation
pit --version
```

### Method 2: Install from PyPI (Future Release)

```bash
# Once published to PyPI
pip install prompt-injection-tester

# Verify installation
pit --version
```

### Method 3: Docker (Recommended for Production)

```bash
# Pull pre-built image (future)
docker pull ghcr.io/your-org/pit:2.0.0

# Or build locally
cd tools/prompt_injection_tester
docker build -t pit:2.0.0 .

# Run
docker run --rm pit:2.0.0 --version
```

## Docker Deployment

### Quick Start with Docker Compose

```bash
cd tools/prompt_injection_tester

# Start services (PIT + Ollama for testing)
docker-compose up -d

# Pull Ollama model
docker-compose exec ollama ollama pull llama3:latest

# Run scan
docker-compose run --rm pit scan http://ollama:11434/api/chat --auto --model llama3:latest

# View reports in browser
docker-compose --profile ui up -d nginx
# Open http://localhost:8080
```

### Production Deployment

#### Single Container

```bash
# Run against external LLM
docker run --rm \
  -v $(pwd)/reports:/reports \
  pit:2.0.0 scan https://api.openai.com/v1/chat/completions \
  --auto \
  --output /reports/report.html
```

#### Docker Compose with Custom Config

```bash
# Create config directory
mkdir -p config

# Create config.yaml
cat > config/config.yaml <<EOF
target:
  url: "https://api.openai.com/v1/chat/completions"
  token: "\${OPENAI_API_KEY}"
  model: "gpt-4"
  timeout: 30

attack:
  patterns:
    - direct_instruction_override
    - role_manipulation
    - delimiter_confusion
  rate_limit: 0.5

reporting:
  format: "html"
  output: "/reports/security_assessment.html"
EOF

# Run with config
docker-compose run --rm \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  pit scan https://api.openai.com/v1/chat/completions \
  --config /config/config.yaml
```

#### Kubernetes Deployment

```yaml
# pit-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pit-security-scan
spec:
  template:
    spec:
      containers:
      - name: pit
        image: pit:2.0.0
        command: ["pit", "scan"]
        args:
          - "http://llm-service:8080/api/chat"
          - "--auto"
          - "--output"
          - "/reports/report.html"
        volumeMounts:
        - name: reports
          mountPath: /reports
        env:
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-credentials
              key: api-key
      volumes:
      - name: reports
        persistentVolumeClaim:
          claimName: pit-reports-pvc
      restartPolicy: Never
```

Apply with:

```bash
kubectl apply -f pit-job.yaml
kubectl logs -f job/pit-security-scan
```

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Docker (optional)
- Ollama or compatible LLM endpoint

### Setup Steps

```bash
# 1. Clone and enter directory
git clone <repository-url>
cd tools/prompt_injection_tester

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install with dev dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks (optional)
pre-commit install

# 5. Run tests
pytest tests/ -v

# 6. Run formatters
black pit/ tests/
ruff check pit/ tests/ --fix

# 7. Type checking
mypy pit/ --ignore-missing-imports
```

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes to pit/ directory

# 3. Run tests
pytest tests/ -v --cov=pit

# 4. Run report tests
python tests/test_reports.py

# 5. Run E2E tests (requires Ollama)
python tests/e2e_test.py --target http://localhost:11434/api/chat --quick

# 6. Format and lint
black pit/ tests/
ruff check pit/ tests/ --fix

# 7. Commit changes
git add .
git commit -m "feat: your feature description"

# 8. Push and create PR
git push origin feature/your-feature
```

## CI/CD Pipeline

### GitHub Actions Workflow

The repository includes a comprehensive CI/CD pipeline at [.github/workflows/pit-test.yml](../../.github/workflows/pit-test.yml).

#### Pipeline Stages

1. **Lint and Test**
   - Python 3.10, 3.11, 3.12
   - Ruff linting
   - Black formatting check
   - MyPy type checking
   - Integration tests with coverage

2. **Docker Build**
   - Build Docker image
   - Test Docker container

3. **End-to-End Test**
   - Start Ollama service
   - Run full pipeline test
   - Generate reports

4. **Security Scan**
   - Safety dependency check
   - Bandit security analysis

#### Triggering CI/CD

```bash
# Automatic triggers:
# - Push to main/develop
# - Pull request to main/develop
# - Changes to tools/prompt_injection_tester/**

# Manual trigger:
# - Go to Actions tab in GitHub
# - Select "PIT - Continuous Integration"
# - Click "Run workflow"
```

#### Viewing Results

```bash
# Check workflow status
gh workflow view "PIT - Continuous Integration"

# Download artifacts
gh run download <run-id> --name e2e-reports
gh run download <run-id> --name security-reports
```

## Release Checklist

### Pre-Release

- [ ] All tests passing (local and CI)
- [ ] Documentation updated
  - [ ] USER_GUIDE.md
  - [ ] PATTERN_DEVELOPMENT.md
  - [ ] ARCHITECTURE.md
  - [ ] CHANGELOG.md
- [ ] Version bumped in pyproject.toml
- [ ] Docker image builds successfully
- [ ] E2E tests pass against real LLM
- [ ] Security scans clean
- [ ] Release notes prepared

### Release Process

```bash
# 1. Update version in pyproject.toml
# [project]
# version = "2.0.0"

# 2. Update CHANGELOG.md
cat >> CHANGELOG.md <<EOF

## [2.0.0] - 2026-01-26

### Added
- Sequential 4-Phase Pipeline architecture
- Multi-format reporting (JSON, YAML, HTML)
- Docker containerization support
- Comprehensive documentation

### Changed
- Complete re-architecture from v1.x
- Type-safe configuration with Pydantic v2
- Modern CLI with Typer and Rich

### Fixed
- Tool use concurrency errors eliminated
EOF

# 3. Commit version bump
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 2.0.0"

# 4. Create git tag
git tag -a v2.0.0 -m "Release version 2.0.0"

# 5. Push changes and tag
git push origin main
git push origin v2.0.0

# 6. Build and push Docker image
docker build -t pit:2.0.0 .
docker tag pit:2.0.0 ghcr.io/your-org/pit:2.0.0
docker tag pit:2.0.0 ghcr.io/your-org/pit:latest
docker push ghcr.io/your-org/pit:2.0.0
docker push ghcr.io/your-org/pit:latest

# 7. Create GitHub release
gh release create v2.0.0 \
  --title "PIT v2.0.0 - Sequential Pipeline Architecture" \
  --notes-file RELEASE_NOTES.md \
  --latest
```

### Post-Release

- [ ] Verify Docker image pulls correctly
- [ ] Update documentation website
- [ ] Announce release (if applicable)
- [ ] Monitor for issues
- [ ] Update example repositories

## Distribution

### PyPI Publishing (Future)

```bash
# Build distribution packages
python -m build

# Upload to PyPI (test first)
twine upload --repository testpypi dist/*

# Verify installation
pip install --index-url https://test.pypi.org/simple/ prompt-injection-tester

# Upload to production PyPI
twine upload dist/*
```

### Docker Registry

```bash
# GitHub Container Registry
docker login ghcr.io -u YOUR_USERNAME
docker push ghcr.io/your-org/pit:2.0.0

# Docker Hub
docker login
docker tag pit:2.0.0 your-username/pit:2.0.0
docker push your-username/pit:2.0.0
```

## Upgrading

### From v1.x to v2.0.0

**Breaking Changes:**

- Complete re-architecture with new sequential pipeline
- CLI interface changed (now using Typer)
- Configuration format updated (Pydantic v2)
- Import paths changed

**Migration Guide:**

```bash
# 1. Backup existing configs
cp config.yaml config.yaml.v1.backup

# 2. Update configuration format
# Old format (v1.x):
# target: "http://localhost:11434"
# patterns: ["pattern1", "pattern2"]

# New format (v2.0.0):
cat > config.yaml <<EOF
target:
  url: "http://localhost:11434/api/chat"
  model: "llama3:latest"
  timeout: 30

attack:
  patterns:
    - direct_instruction_override
    - role_manipulation
  rate_limit: 1.0

reporting:
  format: "html"
  output: "report.html"
EOF

# 3. Update CLI commands
# Old: pit test http://localhost:11434
# New: pit scan http://localhost:11434/api/chat --auto

# 4. Update custom patterns (if any)
# See PATTERN_DEVELOPMENT.md for new pattern interface
```

### Upgrading Within v2.x

```bash
# Standard upgrade
pip install --upgrade prompt-injection-tester

# Or with Docker
docker pull ghcr.io/your-org/pit:latest
```

## Troubleshooting

### Common Issues

**Issue: Docker image won't build**

```bash
# Clear Docker cache
docker builder prune --all

# Rebuild without cache
docker build --no-cache -t pit:2.0.0 .
```

**Issue: Permission denied in Docker**

```bash
# Run as current user
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/reports:/reports \
  pit:2.0.0 scan <target> --auto
```

**Issue: CI/CD tests failing**

```bash
# Run CI/CD locally with act
act -j lint-and-test

# Check specific test
pytest tests/integration/test_pipeline.py -v -k test_pipeline_phases_sequential
```

## Support

For issues, questions, or contributions:

1. Check [USER_GUIDE.md](USER_GUIDE.md) for usage help
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. Search [GitHub Issues](https://github.com/your-org/ai-llm-red-team-handbook/issues)
4. Create new issue with:
   - PIT version (`pit --version`)
   - Installation method (pip, Docker, source)
   - Error messages or unexpected behavior
   - Steps to reproduce

---

**Version**: 2.0.0
**Last Updated**: 2026-01-26
**License**: CC BY-SA 4.0
