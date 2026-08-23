"""
Smoke tests for the models API.
Run with: python models_api/scripts/smoke_test_models.py
These tests check that all 3 models load correctly and produce non-empty answers.
"Smoke test" = a quick sanity check to make sure nothing is obviously broken.
"""
from __future__ import annotations  # allows modern type hint syntax on older Python

import sys          # for modifying Python's module search path
from pathlib import Path  # for file system path operations

# ── Fix the import path ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
# __file__ = this script: models_api/scripts/smoke_test_models.py
# .parents[0] = scripts/ folder
# .parents[1] = models_api/ folder ← ROOT

sys.path.insert(0, str(ROOT))
# insert models_api/ at the FRONT of Python's path
# this means "import api.main" will look in models_api/api/main.py
# without this, Python wouldn't know where to find "api.main"

from fastapi.testclient import TestClient  # noqa: E402
# TestClient simulates HTTP requests WITHOUT starting a real server
# used for testing FastAPI apps programmatically
# noqa: E402 = tells the linter to ignore "import not at top of file" warning

from api.main import app  # noqa: E402
# import our FastAPI app from models_api/api/main.py


def main() -> None:
    cases = [
        (
            "/python",                               # which endpoint to test
            "What is a list comprehension in Python?",  # test query
        ),
        (
            "/organic",
            "What are the properties of benzene?",
        ),
        (
            "/gita",
            "What does Krishna say about duty in chapter 2?",
        ),
    ]

    # Context manager required so startup loads models into MODEL_REGISTRY.
    with TestClient(app) as client:
        # 'with TestClient(app) as client:' triggers the FastAPI startup event
        # this loads all 3 models into MODEL_REGISTRY before tests run
        # without 'with', models wouldn't be loaded and all requests would return 503

        for path, query in cases:
            # loop through all 3 test cases
            r = client.post(path, json={"query": query})
            # send a POST request to the endpoint with the test query as JSON body
            # client.post() returns an httpx.Response object

            assert r.status_code == 200, f"{path}: {r.status_code} {r.text}"
            # assert = check that this condition is True; if False → raise AssertionError with message
            # we expect HTTP 200 (OK); if we get anything else, the test fails with a clear error

            data = r.json()
            # parse the JSON response into a Python dict

            assert "answer" in data and data["answer"].strip(), f"{path}: empty answer: {data}"
            # check: response has an "answer" field AND it's not empty/whitespace-only
            # data["answer"].strip() returns empty string (falsy) if answer is only whitespace

            assert data.get("model") == path.strip("/"), data
            # check: the "model" field in response matches the endpoint name
            # path.strip("/") removes the leading slash: "/python" → "python"

            preview = data["answer"][:200].replace("\n", " ")
            # take first 200 characters of answer, replace newlines with spaces for clean printing
            print(f"OK {path} -> ({len(data['answer'])} chars) {preview!r}...")
            # {preview!r} = use repr() — shows the string with quotes and escape characters
            # useful for spotting weird characters in the output

        # ── Test auto-routing endpoint ────────────────────────────────────────
        r = client.post("/ask", json={"query": "Explain benzene toxicity and solubility."})
        # send a query with chemistry keywords to the auto-routing endpoint
        assert r.status_code == 200, r.text

        routed = r.json()
        assert routed.get("model") == "organic", routed
        # verify that the /ask endpoint correctly routed to "organic" (chemistry keywords detected)

        print(f"OK /ask (routed organic) -> model={routed['model']}")


if __name__ == "__main__":
    # only runs this block when the script is executed directly (not when imported)
    main()
    print("All smoke tests passed.")
