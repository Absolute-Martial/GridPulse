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


def test_streamlit_dashboard_is_removed() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert not (repo_root / "frontend" / "streamlit_app.py").exists()


def test_frontend_serves_forecasting_labels() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    html = (repo_root / "frontend" / "index.html").read_text(encoding="utf-8")
    script = (repo_root / "frontend" / "app.js").read_text(encoding="utf-8")

    assert "GridPulse Forecasting Dashboard" in html
    assert "Substation Forecast" in script
    assert "Forecast Explanation" in script
