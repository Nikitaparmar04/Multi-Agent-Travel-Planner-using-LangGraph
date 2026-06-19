from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agent.agentic_workflow import GraphBuilder
import traceback
import os

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


# Health Check Endpoint
@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "TripMind API"
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