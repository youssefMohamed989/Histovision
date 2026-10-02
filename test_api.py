from fastapi.testclient import TestClient

from liver_histo_ai.api.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "device" in body


def test_get_default_config():
    resp = client.get("/pipeline/config")
    assert resp.status_code == 200
    body = resp.json()
    assert "tiling" in body
    assert body["tiling"]["tile_size"] == 512


def test_run_pipeline_missing_slide_returns_404(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVER_HISTO_DATA_ROOT", str(tmp_path))
    resp = client.post("/pipeline/run", json={"slide_path": "does_not_exist.svs"})
    assert resp.status_code == 404


def test_run_pipeline_rejects_paths_outside_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVER_HISTO_DATA_ROOT", str(tmp_path))
    outside = tmp_path.parent / "secret.svs"
    outside.write_bytes(b"x")
    for bad in ("../secret.svs", str(outside), "/etc/passwd"):
        resp = client.post("/pipeline/run", json={"slide_path": bad})
        assert resp.status_code == 400, bad


def test_run_pipeline_rejects_client_chosen_config(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVER_HISTO_DATA_ROOT", str(tmp_path))
    resp = client.post("/pipeline/run", json={"slide_path": "a.svs", "config_path": "/etc/hostname"})
    assert resp.status_code == 422


def test_error_detail_does_not_leak_internals(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVER_HISTO_DATA_ROOT", str(tmp_path))
    (tmp_path / "bad.svs").write_bytes(b"not a slide")
    resp = client.post("/pipeline/run", json={"slide_path": "bad.svs"})
    assert resp.status_code == 500
    assert str(tmp_path) not in resp.text


def test_cors_disabled_by_default():
    resp = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in resp.headers
