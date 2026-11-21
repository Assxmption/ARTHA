"""API health endpoint tests."""

import anyio
import httpx

from app.main import app


def test_health_endpoint_is_real_api_response():
    async def _test():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/health")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "ok"
            assert body["service"] == "artha"
            assert "optional_dependencies" in body

    anyio.run(_test)


def test_health_endpoint_is_in_openapi():
    assert "/api/health" in app.openapi()["paths"]


def test_mcx_read_only_endpoints_are_in_openapi():
    paths = app.openapi()["paths"]
    assert "/api/quant/mcx/contracts" in paths
    assert "/api/quant/mcx/ohlcv/{commodity}" in paths
