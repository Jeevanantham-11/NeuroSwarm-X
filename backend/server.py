import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from digital_twin_core import DigitalTwin

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

twin = DigitalTwin(dt=0.2, seed=7)
running = False


@app.get("/")
def root():
    return {"status": "NeuroSwarm-X Twin Server running"}


@app.post("/start")
def start():
    global running
    running = True
    twin.add_log("▶ Simulation START")
    return {"ok": True}


@app.post("/stop")
def stop():
    global running
    running = False
    twin.add_log("⏸ Simulation STOP")
    return {"ok": True}


@app.post("/toggle_disturb")
def toggle_disturb():
    twin.disturbances = not twin.disturbances
    twin.add_log(f"Disturbances -> {twin.disturbances}")
    return {"ok": True, "disturbances": twin.disturbances}


@app.post("/add_task")
def add_task():
    twin.add_random_task()
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    twin.add_log("Dashboard connected")

    try:
        while True:
            if running:
                twin.step()

            await ws.send_json(twin.export_state())
            await asyncio.sleep(twin.dt)

    except Exception:
        twin.add_log("Dashboard disconnected")
