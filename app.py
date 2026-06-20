from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agent.agentic_workflow import GraphBuilder
import traceback

app = FastAPI(
    title="TripMind API",
    version="1.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production me specific domains use karna
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Model
class QueryRequest(BaseModel):
    query: str


# Root Endpoint
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "TripMind API"
    }


# ── Dedicated Health Check Endpoint (used by Docker HEALTHCHECK) ──────────────
# Docker hits this endpoint every 30s to verify the app is alive.
# Returns 200 OK when healthy — any non-2xx causes the container to go unhealthy.
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "TripMind API",
        "version": "1.0.0"
    }


# Main Query Endpoint
@app.post("/query")
async def query_travel_agent(request: QueryRequest):
    try:
        # Build Graph
        graph = GraphBuilder()
        react_app = graph()

        # Input Message
        messages = {
            "messages": [request.query]
        }

        # Invoke Agent
        output = react_app.invoke(messages)

        # Extract Final Response
        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content
        else:
            final_output = str(output)

        return {
            "answer": final_output
        }

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )
