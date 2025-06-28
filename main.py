import os
import math
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.mount("/", StaticFiles(directory="dist", html=True), name="static")


# ✅ CORS fix: allow specific origin (React Vite frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # react dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input models
class WCSInput(BaseModel):
    bandwidth: float = Field(..., gt=0, description="Bandwidth in Hz")
    quantizer_bits: int = Field(..., gt=0)
    source_code_rate: float = Field(..., gt=0, le=1)
    channel_code_rate: float = Field(..., gt=0, le=1)
    burst_overhead: float = Field(..., ge=0)

class OFDMInput(BaseModel):
    m_order: int = Field(..., gt=1)
    n_subcarriers: int = Field(..., gt=0)
    t_sym: float = Field(..., gt=0)
    n_symbols_prb: int = Field(..., gt=0)
    n_prb_parallel: int = Field(..., gt=0)
    bandwidth: float = Field(..., gt=0)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_openai(scenario: str, inputs: dict, results: dict) -> str:
    prompt = (
        f"You are a helpful assistant. The following are results from a {scenario} system:\n"
        f"{results}\n\n"
        "For each value, write a short sentence that says what it means in simple words, and include the number itself in the sentence.\n"
        "Do not explain how it was calculated, just say what the number represents in an easy and friendly way.\n"
        "One sentence per value."
    )
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    return response.choices[0].message.content
class LinkBudgetInput(BaseModel):
    txPower: float = Field(..., description="Transmit power (dBm)")
    txGain: float = Field(..., description="Transmit antenna gain (dBi)")
    txCableLoss: float = Field(..., ge=0, description="Transmit cable loss (dB)")
    rxGain: float = Field(..., description="Receive antenna gain (dBi)")
    rxCableLoss: float = Field(..., ge=0, description="Receive cable loss (dB)")
    distanceKm: float = Field(..., gt=0, description="Link distance (km)")
    frequencyMHz: float = Field(..., gt=0, description="Carrier frequency (MHz)")
    extraLoss: float = Field(0, ge=0, description="Additional environment losses (dB)")
    arGain: float = Field(0, description="Receiver amplifier gain (dB) — default 0")


@app.post("/api/link-budget")
async def calc_link_budget(inp: LinkBudgetInput):
    # 1) Effective isotropic radiated power
    eirp = inp.txPower + inp.txGain - inp.txCableLoss

    # 2) Free-space path loss
    fspl = 32.44 + 20 * math.log10(inp.distanceKm) + 20 * math.log10(inp.frequencyMHz)

    # 3) Received power
    total_loss = fspl + inp.extraLoss
    rx_power = eirp + inp.rxGain + inp.arGain - total_loss - inp.rxCableLoss

    numbers = {
        "EIRP (dBm)":           round(eirp, 2),
        "Received Power (dBm)": round(rx_power, 2),
    }

    # ---------- richer prompt ----------
    prompt = (
        "Explain step-by-step how each input was used to compute the link budget.\n"
        "• Show the formula not in latex , in simple form for EIRP and substitute the user’s numbers.\n"
        "• Combine gains and losses to get Received Power and show that substitution as well.\n"
        "Keep it concise (≤ 140 words) but include the numeric substitutions and final answers."
    )
    # ------------------------------------

    explanation = call_openai(prompt, inp.dict(), numbers)
    return {"numbers": numbers, "explanation": explanation}


class CellularInput(BaseModel):
    totalArea:       float = Field(..., gt=0, description="Total area to cover (km²)")
    cellRadius:      float = Field(..., gt=0, description="Radius of one cell (km)")
    totalChannels:   int   = Field(..., gt=0, description="Total available channels")
    clusterSize:     int   = Field(..., gt=1, description="Cells per cluster (reuse factor)")
    subsPerCell:     int   = Field(0, ge=0,  description="Subscribers per cell (optional)")


def ij_for_cluster(N: int) -> str:
    """Return the (i,j) hex move that realises N (smallest i ≥ j ≥ 0)."""
    for i in range(int(math.sqrt(N)) + 2):
        for j in range(i + 1):
            if i * i + i * j + j * j == N:
                return f"{i},{j}"
    return "n/a"


@app.post("/api/cellular")
async def calc_cellular(inp: CellularInput):
    # 1) Cell area (hexagon ≈ 2.6·R²)
    cell_area = 2.6 * inp.cellRadius ** 2

    # 2) Number of cells to cover the region
    total_cells = math.ceil(inp.totalArea / cell_area)

    # 3) Channels per cell
    channels_per_cell = inp.totalChannels // inp.clusterSize

    # 4) Cluster count
    total_clusters = math.ceil(total_cells / inp.clusterSize)

    # 5) Traffic numbers
    total_subs = total_cells * inp.subsPerCell
    subs_per_channel = round(inp.subsPerCell / channels_per_cell, 2) if inp.subsPerCell else None

    # 6) Reuse distance
    reuse_distance = inp.cellRadius * math.sqrt(3 * inp.clusterSize)

    # 7) Advanced metrics
    freq_reuse_factor = 1 / inp.clusterSize
    cochannel_ratio   = math.sqrt(3 * inp.clusterSize)
    channels_cluster  = channels_per_cell * inp.clusterSize          # == totalChannels
    system_capacity   = total_clusters * inp.totalChannels
    i_j_move          = ij_for_cluster(inp.clusterSize)

    numbers = {
        # basic
        "Cell Area (km²)":          round(cell_area, 2),
        "Total Cells":              total_cells,
        "Channels per Cell":        channels_per_cell,
        "Total Clusters":           total_clusters,
        "Reuse Distance (km)":      round(reuse_distance, 2),
        "Total Subscribers":        total_subs,
        "Subscribers per Channel":  subs_per_channel,
        # advanced
        "Frequency Reuse Factor":   round(freq_reuse_factor, 3),
        "Co-channel Reuse Ratio Q": round(cochannel_ratio, 2),
        "Channels per Cluster":     channels_cluster,
        "System Capacity (channels)": system_capacity,
        "(i,j) Move":               i_j_move,
    }

    # plain-text, no LaTeX
    prompt = (
        "Write each output on its own line with the formula and numbers, plain text only:\n"
        "Cell area = 2.6 * R^2 = {Cell Area (km²)} km²\n"
        "Total cells = ceil(Area / CellArea) = {Total Cells}\n"
        "Channels/cell = S / N = {Channels per Cell}\n"
        "Clusters = ceil(Cells / N) = {Total Clusters}\n"
        "Reuse distance = R*sqrt(3N) = {Reuse Distance (km)} km\n"
        "Freq-reuse factor = 1/N = {Frequency Reuse Factor}\n"
        "Co-channel ratio Q = sqrt(3N) = {Co-channel Reuse Ratio Q}\n"
        "Channels/cluster = k*N = {Channels per Cluster}\n"
        "System capacity = clusters*S = {System Capacity (channels)}\n"
        "Include one short comment (<15 words) about feasibility at the end."
    )
    explanation = call_openai(prompt, inp.dict(), numbers)
    return {"numbers": numbers, "explanation": explanation}

@app.post("/api/wcs")
async def calc_wcs(inp: WCSInput):
    fs = 2 * inp.bandwidth
    Rq = fs * inp.quantizer_bits
    Rs = Rq * inp.source_code_rate
    Rc = Rs * inp.channel_code_rate
    Ri = Rc
    Rb = Ri * (1 + inp.burst_overhead / 100)

    numbers = {
        "fs_Hz": fs,
        "Rq_bps": Rq,
        "Rs_bps": Rs,
        "Rc_bps": Rc,
        "Ri_bps": Ri,
        "Rb_bps": Rb
    }

    explanation = call_openai("Wireless Communication", inp.dict(), numbers)
    return {"numbers": numbers, "explanation": explanation}

@app.post("/api/ofdm")
async def calc_ofdm(inp: OFDMInput):
    bits_per_re = math.log2(inp.m_order)
    re_rate = bits_per_re / inp.t_sym
    symbol_rate = re_rate * inp.n_subcarriers
    rb_rate = symbol_rate * inp.n_symbols_prb
    throughput = rb_rate * inp.n_prb_parallel
    spectral_eff = throughput / inp.bandwidth

    numbers = {
        "RE_rate_bps": re_rate,
        "Symbol_rate_bps": symbol_rate,
        "RB_rate_bps": rb_rate,
        "Throughput_bps": throughput,
        "Spectral_eff_bpsHz": spectral_eff
    }

    explanation = call_openai("OFDM", inp.dict(), numbers)
    return {"numbers": numbers, "explanation": explanation}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
