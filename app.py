from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agent.agentic_workflow import GraphBuilder
import os

app = FastAPI(title="TripMind API", version="1.0")

# ── CORS (allows frontend to call backend) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str


@app.get("/")
async def health():
    return {"status": "ok", "service": "TripMind API"}


@app.post("/query")
async def query_travel_agent(request: QueryRequest):
    try:
        graph = GraphBuilder()
        react_app = graph()

        messages = {
            "messages": [request.query]
        }

        output = react_app.invoke(messages)

        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content
        else:
            final_output = str(output)

        return {"answer": final_output}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )