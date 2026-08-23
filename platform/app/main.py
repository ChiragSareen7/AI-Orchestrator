from __future__ import annotations  # allows modern type hint syntax on older Python

from pathlib import Path  # pathlib lets us work with file/folder paths in an OS-independent way

from fastapi import FastAPI, HTTPException  # FastAPI is the web framework — it handles incoming HTTP requests
from fastapi.responses import FileResponse  # FileResponse sends a file (like HTML) back to the browser
from fastapi.staticfiles import StaticFiles  # StaticFiles serves a whole folder of files (JS, CSS, HTML)
from fastapi.middleware.cors import CORSMiddleware  # CORS middleware allows browser frontends on different ports to call our API
from pydantic import BaseModel, Field  # BaseModel defines the shape of request/response data; Field adds validation rules

from app.services.followup import answer_followup
from app.services.orchestrator import run_query_pipeline  # the main pipeline function that runs all models
from app.services.logger import ensure_store_files  # makes sure the logs.json and model_performance.json files exist
from app.services.learning import generate_report  # generates a summary report of model performance over time
from app.services.adaptive_routing import ensure_routing_files, force_reexploration
from app.services.adaptive_routing.router import get_routing_status
from app.services.adaptive_routing.seed import bootstrap_all_clusters, bootstrap_cluster, get_seed_status, verify_direct_routing
from app.services.self_correction import (
    ensure_profiles_file,
    ensure_self_correction_logs,
    get_requirements_profile,
    list_requirements_profiles,
    run_self_correction_loop,
    save_requirements_profile,
)


# ── Request Model ────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    # This class defines exactly what JSON the client must send in POST /query
    query: str = Field(..., min_length=1)  # 'query' is required (... means required), must be at least 1 character
    context: str | None = Field(None, description="Optional RAG/context text for semantic grounding")
    # 'context' is optional (None by default) — provide it if you have retrieved document chunks to ground the answer against
    ground_truth: str | None = Field(None, description="Optional reference answer for semantic accuracy")
    force_explore: str | None = Field(
        None,
        description="Optional cluster id (e.g. python, chemistry/sub_1) to force full re-exploration",
    )
    enable_routing: bool = Field(
        True,
        description="When false, bypass adaptive routing and call all models every time",
    )


class ReexploreRequest(BaseModel):
    cluster_id: str | None = Field(None, description="Specific cluster to force re-exploration for")
    all_clusters: bool = Field(False, description="Force re-exploration for all clusters")


class SeedRoutingRequest(BaseModel):
    cluster_id: str | None = Field(None, description="Seed one cluster only (python, chemistry, gita, general)")
    mode: str = Field("bootstrap", description="bootstrap (fast) or verify (test direct routing)")


class FollowupRequest(BaseModel):
    followup_question: str = Field(..., min_length=1)
    target_reference: str | None = Field(None, description="Natural-language description of the response being referenced")
    parent_log_id: str | None = Field(None, description="Optional exact parent query log id")
    parent_response_id: str | None = Field(None, description="Optional exact response id within the parent query log")


class RequirementsProfileRequest(BaseModel):
    tone_style: str = ""
    detail_level: str = ""
    must_include: str = ""
    must_avoid: str = ""
    format_expectations: str = ""
    domain_constraints: str = ""
    extra: dict = Field(default_factory=dict)


class SelfCorrectRequest(BaseModel):
    query: str = Field(..., min_length=1)
    client_id: str = Field("default", min_length=1)
    context: str | None = None
    enable_routing: bool = True


# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Multi Model Orchestration Platform", version="1.0.0")
# creates the FastAPI application object with a human-readable title shown in /docs

STATIC_DIR = Path(__file__).resolve().parent / "static"
# __file__ is the path to THIS file (main.py)
# .resolve() converts it to an absolute path
# .parent goes up one folder level (to the app/ folder)
# / "static" joins with the static subfolder — this is where index.html, app.js, styles.css live

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# tells FastAPI: any request to /static/... should serve files from the static/ folder
# e.g. /static/styles.css → serves platform/app/static/styles.css


# ── CORS Configuration ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,  # CORS = Cross-Origin Resource Sharing — controls which websites can call our API
    allow_origins=[  # list of website origins (protocol + domain + port) allowed to make requests
        "http://127.0.0.1:5500",   # common VS Code Live Server port
        "http://localhost:5500",
        "http://[::]:5500",        # IPv6 version of localhost
        "http://127.0.0.1:5173",   # Vite dev server port
        "http://localhost:5173",
        "http://[::]:5173",
        "http://127.0.0.1:3000",   # React / Next.js default dev port
        "http://localhost:3000",
        "http://[::]:3000",
    ],
    allow_credentials=True,  # allows cookies and auth headers to be sent with cross-origin requests
    allow_methods=["*"],     # allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],     # allows all HTTP headers in requests
)


# ── Startup Event ─────────────────────────────────────────────────────────────
@app.on_event("startup")  # this function runs automatically when the server starts up
def startup() -> None:
    ensure_store_files()
    ensure_routing_files()
    ensure_profiles_file()
    ensure_self_correction_logs()
    # makes sure store/logs.json and store/model_performance.json exist on disk
    # if they don't exist, it creates them with empty content ([] and {})


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")  # handles GET requests to the root URL "http://127.0.0.1:8020/"
def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
    # sends the browser the index.html file (the visual dashboard UI)


@app.post("/query")  # handles POST requests to "http://127.0.0.1:8020/query"
def query_endpoint(request: QueryRequest) -> dict:
    # FastAPI automatically reads the JSON body and creates a QueryRequest object from it
    # then returns the pipeline result as JSON
    return run_query_pipeline(
        request.query,           # the user's question string
        context=request.context,       # optional context chunks for grounding
        ground_truth=request.ground_truth,  # optional correct answer for accuracy
        force_explore=request.force_explore,
        enable_routing=request.enable_routing,
    )


@app.get("/routing/status")
def routing_status_endpoint() -> dict:
    return get_routing_status()


@app.get("/routing/seed/status")
def routing_seed_status_endpoint() -> dict:
    return get_seed_status()


@app.post("/routing/seed")
def routing_seed_endpoint(request: SeedRoutingRequest) -> dict:
    if request.mode == "verify":
        return {"results": verify_direct_routing()}
    if request.cluster_id:
        from app.services.adaptive_routing.seed import load_seed_queries

        queries = load_seed_queries().get(request.cluster_id)
        if not queries:
            raise HTTPException(status_code=400, detail=f"Unknown cluster: {request.cluster_id}")
        row = bootstrap_cluster(request.cluster_id, queries)
        return {"cluster_id": request.cluster_id, "result": row, "status": get_seed_status()}
    results = bootstrap_all_clusters()
    return {"bootstrapped": list(results.keys()), "status": get_seed_status()}


@app.post("/routing/re-explore")
def routing_reexplore_endpoint(request: ReexploreRequest) -> dict:
    try:
        return force_reexploration(cluster_id=request.cluster_id, all_clusters=request.all_clusters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query/self-correct")
def self_correct_endpoint(request: SelfCorrectRequest) -> dict:
    try:
        return run_self_correction_loop(
            query=request.query,
            client_id=request.client_id,
            context=request.context,
            enable_routing=request.enable_routing,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/requirements/{client_id}")
def get_requirements_endpoint(client_id: str) -> dict:
    profile = get_requirements_profile(client_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile for client_id={client_id}")
    return profile


@app.put("/requirements/{client_id}")
def put_requirements_endpoint(client_id: str, request: RequirementsProfileRequest) -> dict:
    return save_requirements_profile(client_id, request.model_dump())


@app.get("/requirements")
def list_requirements_endpoint() -> dict:
    return list_requirements_profiles()


@app.post("/followup")
def followup_endpoint(request: FollowupRequest) -> dict:
    try:
        return answer_followup(
            request.followup_question,
            target_reference=request.target_reference,
            parent_log_id=request.parent_log_id,
            parent_response_id=request.parent_response_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/report")  # handles GET requests to "http://127.0.0.1:8020/report"
def report_endpoint() -> dict:
    return generate_report()
    # reads store/logs.json and store/model_performance.json and returns aggregate stats:
    # best model per domain, average latency, accuracy comparison, total queries count
