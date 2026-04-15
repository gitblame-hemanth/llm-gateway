"""Integration tests for API routes via FastAPI TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestModelsEndpoint:
    def test_models_endpoint_lists_models(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        model_ids = [m["id"] for m in data["data"]]
        assert "mock-model" in model_ids


class TestChatCompletion:
    def test_chat_completion_success(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello from mock provider!"
        assert data["usage"]["total_tokens"] == 30

    def test_chat_completion_openai_format(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
                "temperature": 0.7,
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Verify OpenAI-compatible structure
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]

    def test_invalid_model_404(self, client: TestClient) -> None:
        # Swap in a router with no providers so unknown models fail
        from src.core.config import RoutingConfig
        from src.providers.registry import ProviderRegistry
        from src.routing.router import Router

        empty_registry = ProviderRegistry()
        empty_config = RoutingConfig(default_provider="nonexistent")
        client.app.state.router = Router(empty_registry, empty_config)  # type: ignore[union-attr]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "does-not-exist",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 404


class TestEmbedding:
    def test_embedding_success(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "mock-embed",
                "input": ["Hello world"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"
        assert len(data["data"][0]["embedding"]) == 3
        assert data["usage"]["prompt_tokens"] == 5

    def test_embedding_multiple_inputs(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "mock-embed",
                "input": ["Hello", "World"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2
        assert data["data"][0]["index"] == 0
        assert data["data"][1]["index"] == 1
