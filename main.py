import os
import math
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

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

class CellularInput(BaseModel):
    totalArea:       float = Field(..., gt=0, description="Total area to cover (km²)")
    cellRadius:      float = Field(..., gt=0, description="Radius of one cell (km)")
    totalChannels:   int   = Field(..., gt=0, description="Total available channels")
    clusterSize:     int   = Field(..., gt=1, description="Cells per cluster (reuse factor)")
    subsPerCell:     int   = Field(0, ge=0,  description="Subscribers per cell (optional)")

def ij_for_cluster(N: int) -> str:
    for i in range(int(math.sqrt(N)) + 2):
        for j in range(i + 1):
            if i * i + i * j + j * j == N:
                return f"{i},{j}"
    return "n/a"

def call_openai(prompt: str, inputs: dict, results: dict) -> str:
    # Inject results into the prompt for more context
    context = f"\nUser Inputs: {inputs}\nCalculated Results: {results}\n"
    final_prompt = prompt + context
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=600
    )
    return response.choices[0].message.content

@app.post("/api/link-budget")
async def calc_link_budget(inp: LinkBudgetInput):
    eirp = inp.txPower + inp.txGain - inp.txCableLoss
    fspl = 32.44 + 20 * math.log10(inp.distanceKm) + 20 * math.log10(inp.frequencyMHz)
    total_loss = fspl + inp.extraLoss
    rx_power = eirp + inp.rxGain + inp.arGain - total_loss - inp.rxCableLoss

    numbers = {
        "EIRP (dBm)":           round(eirp, 2),
        "Received Power (dBm)": round(rx_power, 2),
    }
    prompt = (
        "You are a wireless communication assistant. Explain step-by-step how all outputs in this link budget were calculated from the user’s inputs.\n"
        "For each output, write the formula in **plain text only** (not LaTeX, not math markup, not code blocks, just like: EIRP = Tx Power + Tx Gain - Tx Cable Loss)."
        " Substitute the numbers into the formula, show the calculation and the result."
        " After each calculation, briefly explain what the output means and what each value represents."
        " Do not use LaTeX, symbols like $, \\(, \\[, or code formatting. Make the answer simple and beginner-friendly, not brief."
    )
    explanation = call_openai(prompt, inp.dict(), numbers)
    return {"numbers": numbers, "explanation": explanation}

@app.post("/api/cellular")
async def calc_cellular(inp: CellularInput):
    cell_area = 2.6 * inp.cellRadius ** 2
    total_cells = math.ceil(inp.totalArea / cell_area)
    total_clusters = math.ceil(total_cells / inp.clusterSize)

    channels_per_cell = inp.totalChannels // inp.clusterSize if inp.clusterSize > 0 else 0
    if channels_per_cell == 0:
        return {"error": "Channels per cell is zero! Total channels must be at least equal to cluster size."}

    subs_per_channel = (
        round(inp.subsPerCell / channels_per_cell, 2)
        if inp.subsPerCell and inp.subsPerCell > 0 and channels_per_cell > 0
        else None
    )
    total_subs = total_cells * inp.subsPerCell if inp.subsPerCell else 0
    reuse_distance = inp.cellRadius * math.sqrt(3 * inp.clusterSize)
    freq_reuse_factor = 1 / inp.clusterSize
    cochannel_ratio = math.sqrt(3 * inp.clusterSize)
    channels_cluster = channels_per_cell * inp.clusterSize
    system_capacity = total_clusters * inp.totalChannels
    i_j_move = ij_for_cluster(inp.clusterSize)

    numbers = {
        "Cell Area (km²)": round(cell_area, 2),
        "Total Cells": total_cells,
        "Channels per Cell": channels_per_cell,
        "Total Clusters": total_clusters,
        "Reuse Distance (km)": round(reuse_distance, 2),
        "Total Subscribers": total_subs,
        "Subscribers per Channel": subs_per_channel,
        "Frequency Reuse Factor": round(freq_reuse_factor, 3),
        "Co-channel Reuse Ratio Q": round(cochannel_ratio, 2),
        "Channels per Cluster": channels_cluster,
        "System Capacity (channels)": system_capacity,
        "(i,j) Move": i_j_move,
    }
    prompt = (
        "You are a cellular networks assistant. Give a full, clear step-by-step explanation of how every output was computed from the user's inputs in this cellular system design.\n"
        "For each output, write the formula in **plain text only** (not LaTeX, not code, not math markup). Use formulas like: Cell Area = 2.6 * (Cell Radius)^2, and substitute the user's numbers."
        " After each calculation, explain what it means for a real cellular network."
        " Avoid using any LaTeX, code formatting, or symbols like $ or \\( or \\[. Keep everything readable and beginner-friendly."
    )
    explanation = call_openai(prompt, inp.dict(), numbers)
    return {"numbers": numbers, "explanation": explanation}

@app.post("/api/wcs")
async def calc_wcs(inp: WCSInput):
    fs = 2 * inp.bandwidth
    Rq = fs * inp.quantizer_bits
    Rs = Rq * inp.source_code_rate
    Rc = Rs / inp.channel_code_rate
    Ri = Rc
    Rb = Ri * (1 + inp.burst_overhead / 100)
    numbers = {
        "Sampling Rate fs (Hz)": fs,
        "Quantized Data Rate Rq (bps)": Rq,
        "Source Encoded Rate Rs (bps)": Rs,
        "Channel Encoded Rate Rc (bps)": Rc,
        "Interleaved Rate Ri (bps)": Ri,
        "Burst Formatted Rate Rb (bps)": Rb,
    }
    prompt = (
        "You are a digital communications assistant. For each block in the wireless communication chain, do the following:\n"
        "- State the purpose of the block in one short sentence.\n"
        "- Write the calculation formula in **plain text only** (not LaTeX, not math markup, not code blocks, just normal text like 'fs = 2 * bandwidth').\n"
        "- Substitute the numbers directly into the formula, and show the calculation result.\n"
        "- After each result, explain what it means in simple words.\n"
        "Make sure NOT to use any special formatting, LaTeX, code blocks, or math markup. Only plain, readable text for beginners."
    )
    explanation = call_openai(prompt, inp.dict(), numbers)
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
        "Bits per RE": bits_per_re,
        "RE Rate (bps)": re_rate,
        "Symbol Rate (bps)": symbol_rate,
        "RB Rate (bps)": rb_rate,
        "Throughput (bps)": throughput,
        "Spectral Efficiency (bps/Hz)": spectral_eff,
    }
    prompt = (
        "You are an OFDM systems assistant. For each output, do the following:\n"
        "- Write the formula in **plain text only** (not LaTeX, not code, not math markup), e.g., 'Symbol Rate = RE Rate * Number of Subcarriers'."
        "- Substitute the user's numbers into the formula, and show the calculation result."
        "- After each calculation, explain what it means for an OFDM system in simple, readable sentences."
        "- Avoid any LaTeX, code blocks, or special formatting. The answer must be beginner-friendly."
    )
    explanation = call_openai(prompt, inp.dict(), numbers)
    return {"numbers": numbers, "explanation": explanation}

app.mount("/", StaticFiles(directory="dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
