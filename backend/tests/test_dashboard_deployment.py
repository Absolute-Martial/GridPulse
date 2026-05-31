from pathlib import Path


def test_dashboard_docker_assets_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert (repo_root / "frontend" / "index.html").exists()
    assert (repo_root / "frontend" / "app.js").exists()
    assert (repo_root / "frontend" / "styles.css").exists()
    assert (repo_root / "frontend" / "Dockerfile").exists()
    assert (repo_root / "frontend" / "nginx.conf").exists()

    compose_text = (repo_root / "docker-compose.yml").read_text(encoding="utf-8")
    frontend_dockerfile = (repo_root / "frontend" / "Dockerfile").read_text(encoding="utf-8")
    assert "frontend:" in compose_text
    assert "backend:" in compose_text
    assert "nginx" in frontend_dockerfile.lower()


def test_streamlit_dashboard_is_present_and_self_contained() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    streamlit_app = repo_root / "frontend" / "streamlit_app.py"
    streamlit_runtime = repo_root / "frontend" / "streamlit_runtime.py"
    requirements = repo_root / "requirements.txt"
    research_reference = repo_root / "docs" / "research-reference.md"

    assert streamlit_app.exists()
    assert streamlit_runtime.exists()
    assert requirements.exists()
    assert research_reference.exists()

    app_source = streamlit_app.read_text(encoding="utf-8")
    runtime_source = streamlit_runtime.read_text(encoding="utf-8")
    requirements_text = requirements.read_text(encoding="utf-8")
    research_text = research_reference.read_text(encoding="utf-8")

    assert "streamlit_autorefresh" in app_source
    assert "load_artifact" in app_source
    assert "Actual generated load" in app_source
    assert "Research reference" in app_source
    assert "build_streamlit_forecast" in runtime_source
    assert "available_model_paths" in runtime_source
    assert "load_initial_history" in runtime_source
    assert "streamlit" in requirements_text
    assert "scikit-learn==1.8.0" in requirements_text
    assert "plotly" in requirements_text
    assert "GridPulse_Modular_Prototype_Implementation_Paper.docx" in research_text
    assert "GridPulse_Technical_Architecture.docx" in research_text


def test_frontend_serves_forecasting_labels() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    html = (repo_root / "frontend" / "index.html").read_text(encoding="utf-8")
    script = (repo_root / "frontend" / "app.js").read_text(encoding="utf-8")

    assert "GridPulse Forecasting Dashboard" in html
    assert "Substation Forecast" in script
    assert "Forecast Explanation" in script
