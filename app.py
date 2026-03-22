import io
import math
import time
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="AI House Planning System", layout="wide", page_icon="🏠")


# Theme
PRIMARY = "#0D3B66"
ACCENT = "#2EC4B6"
WARNING = "#E9C46A"
DANGER = "#E76F51"
B1_COLOR = "#E76F51"
B2_COLOR = "#F4A261"
B3_COLOR = "#E9C46A"
B4_COLOR = "#264653"
PLOT_BG = "#102038"

SYSTEMS = ["Proposed", "B1", "B2", "B3", "B4"]
SYSTEM_COLORS = {
    "Proposed": ACCENT,
    "B1": B1_COLOR,
    "B2": B2_COLOR,
    "B3": B3_COLOR,
    "B4": B4_COLOR,
}


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #07111d 0%, #0c1a2b 45%, #102038 100%);
        color: #EAF4FF;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111d 0%, #0d2038 100%);
    }
    .paper-card {
        background: rgba(17, 36, 62, 0.88);
        border: 1px solid rgba(46, 196, 182, 0.24);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.28rem 0.6rem;
        border-radius: 999px;
        background: rgba(46, 196, 182, 0.18);
        color: #8ff4e8;
        border: 1px solid rgba(46, 196, 182, 0.35);
        font-weight: 600;
    }
    .delta-card {
        background: rgba(17, 36, 62, 0.92);
        border: 1px solid rgba(233, 196, 106, 0.2);
        border-radius: 14px;
        padding: 0.8rem;
        min-height: 140px;
    }
    .sidebar-summary {
        background: rgba(17, 36, 62, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.8rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODULE_IO = [
    {"Module": "Intent Parser", "Inputs": "Plot, setbacks, budget, style, occupants, climate", "Outputs": "Program graph, constraint params, weights"},
    {"Module": "Layout Generator", "Inputs": "Program graph, plot boundary raster", "Outputs": "Room graph + polygons + openings"},
    {"Module": "Civil/Struct Checks", "Inputs": "Zoning envelope, spans, egress, daylight proxy", "Outputs": "Feasibility flags, penalties, edits"},
    {"Module": "Interior Optimizer", "Inputs": "Room geometry, furniture priors", "Outputs": "Furniture zones, corridors, access areas"},
    {"Module": "MEP Router", "Inputs": "Shell + shafts, terminals, clearances", "Outputs": "Routes, equipment/panel locations, clashes"},
    {"Module": "IoT Planner", "Inputs": "Room use, network model, privacy zones", "Outputs": "Device nodes, gateway graph, coverage map"},
    {"Module": "BIM/IFC + BOQ", "Inputs": "Geometry + semantics", "Outputs": "IFC, schedules/BOQ, clash report"},
    {"Module": "UI Loop", "Inputs": "Candidate set + metrics", "Outputs": "Ranked variants, edits, exports"},
]

CONSTRAINTS_TABLE = [
    {"Discipline": "Civil/Arch", "Constraint": "Setbacks, min areas, adjacency, egress widths, daylight proxy", "Mechanism": "Hard geometry + penalties"},
    {"Discipline": "Structure", "Constraint": "Max span, grid alignment, core reservation, opening limits", "Mechanism": "Grid rules + clash checks"},
    {"Discipline": "Interior", "Constraint": "Door/window clearances, corridor width, accessibility zones", "Mechanism": "SA packing + hard collisions"},
    {"Discipline": "MEP", "Constraint": "Separation, headroom, maintenance access, shaft capacity", "Mechanism": "Graph routing + repair"},
    {"Discipline": "Electrical", "Constraint": "Panel zoning, load balance, voltage drop proxy", "Mechanism": "Load constraints + checks"},
    {"Discipline": "IoT", "Constraint": "Coverage threshold, hop limit, privacy zones, redundancy", "Mechanism": "Coverage objective + risk"},
]

DATASETS_TABLE = [
    ["CubiCasa5K", "floorplans", "5k plans", "research", "civil layout supervision"],
    ["Structured3D", "scenes", "21k", "research", "geometric priors"],
    ["3D-FRONT", "interiors", "19k rooms", "CC BY-NC", "furniture priors"],
    ["ScanNet", "indoor 3D", "1513 scans", "research", "scale statistics"],
    ["IFC4.3", "openBIM", "spec", "CC BY-ND", "BIM integration backbone"],
    ["IFCNet", "IFC sem.", "1.3M objs", "open", "semantic validation"],
    ["IfcOpenShell", "IFC toolkit", "code", "LGPL", "IFC writing/geometry"],
    ["UK-DALE", "energy IoT", "5 homes", "open", "appliance-level electricity"],
    ["CASAS", "sensors", "189 homes", "open", "room-tagged sensor patterns"],
    ["TMY3", "climate", "US TMY", "public", "energy simulation weather"],
]

BASELINES = [
    "B1: Rule templates + greedy packing + fixed shafts",
    "B2: GA on room rectangles; A* routing; penalties for violations",
    "B3: SA on grid layout; JPS routing; swap/resize repair",
    "B4: Sequential BIM workflow + Navisworks clash detection",
    "Proposed: Graph-transformer layout + discipline solvers + NSGA-II multi-objective",
]

BENCHMARK_METRICS = [
    {"metric": "Adjacency F1", "direction": "up", "domain": "Layout", "unit": "", "values": {"Proposed": (0.85, 0.06), "B1": (0.78, 0.08), "B2": (0.81, 0.07), "B3": (0.80, 0.07), "B4": (0.83, 0.06)}},
    {"metric": "Daylight proxy", "direction": "up", "domain": "Layout", "unit": "", "values": {"Proposed": (0.71, 0.09), "B1": (0.60, 0.11), "B2": (0.66, 0.10), "B3": (0.63, 0.10), "B4": (0.69, 0.09)}},
    {"metric": "Interior usability", "direction": "up", "domain": "Layout", "unit": "", "values": {"Proposed": (0.76, 0.07), "B1": (0.62, 0.12), "B2": (0.69, 0.10), "B3": (0.67, 0.10), "B4": (0.73, 0.08)}},
    {"metric": "Circulation", "direction": "down", "domain": "Layout", "unit": "m/step", "values": {"Proposed": (4.3, 0.6), "B1": (5.2, 0.8), "B2": (4.9, 0.7), "B3": (5.0, 0.7), "B4": (4.6, 0.6)}},
    {"metric": "Span-feasible rooms", "direction": "up", "domain": "Structure", "unit": "%", "values": {"Proposed": (92, 5), "B1": (87, 7), "B2": (89, 6), "B3": (88, 6), "B4": (91, 5)}},
    {"metric": "Load-path plausibility", "direction": "up", "domain": "Structure", "unit": "", "values": {"Proposed": (0.90, 0.04), "B1": (0.82, 0.06), "B2": (0.86, 0.05), "B3": (0.85, 0.05), "B4": (0.89, 0.04)}},
    {"metric": "MEP length", "direction": "down", "domain": "MEP", "unit": "m", "values": {"Proposed": (131, 19), "B1": (145, 22), "B2": (138, 20), "B3": (140, 21), "B4": (152, 24)}},
    {"metric": "Pressure-drop proxy", "direction": "down", "domain": "MEP", "unit": "", "values": {"Proposed": (0.33, 0.08), "B1": (0.52, 0.12), "B2": (0.44, 0.10), "B3": (0.47, 0.11), "B4": (0.41, 0.09)}},
    {"metric": "Voltage-drop proxy", "direction": "down", "domain": "MEP", "unit": "%", "values": {"Proposed": (2.0, 0.6), "B1": (3.4, 0.9), "B2": (2.8, 0.8), "B3": (3.0, 0.8), "B4": (2.3, 0.7)}},
    {"metric": "Clashes after repair", "direction": "down", "domain": "MEP", "unit": "#", "values": {"Proposed": (1.8, 1.2), "B1": (7.1, 3.5), "B2": (4.6, 2.6), "B3": (5.0, 2.8), "B4": (3.9, 2.2)}},
    {"metric": "IFC completeness", "direction": "up", "domain": "BIM", "unit": "%", "values": {"Proposed": (96, 2), "B1": (88, 5), "B2": (91, 4), "B3": (90, 4), "B4": (95, 3)}},
    {"metric": "IoT coverage", "direction": "up", "domain": "IoT", "unit": "%", "values": {"Proposed": (93, 4), "B1": (81, 8), "B2": (86, 6), "B3": (85, 7), "B4": (89, 5)}},
    {"metric": "IoT latency", "direction": "down", "domain": "IoT", "unit": "hops", "values": {"Proposed": (1.4, 0.5), "B1": (2.6, 0.8), "B2": (2.1, 0.7), "B3": (2.2, 0.7), "B4": (1.8, 0.6)}},
    {"metric": "IoT risk", "direction": "down", "domain": "IoT", "unit": "", "values": {"Proposed": (0.25, 0.09), "B1": (0.42, 0.13), "B2": (0.36, 0.12), "B3": (0.38, 0.12), "B4": (0.31, 0.11)}},
    {"metric": "Cost error", "direction": "down", "domain": "Cost-Energy", "unit": "%", "values": {"Proposed": (6.5, 2.7), "B1": (12.4, 5.1), "B2": (9.8, 4.2), "B3": (10.5, 4.5), "B4": (7.2, 3.1)}},
    {"metric": "EUI proxy", "direction": "down", "domain": "Cost-Energy", "unit": "kWh/m²-yr", "values": {"Proposed": (79, 13), "B1": (88, 15), "B2": (84, 14), "B3": (86, 15), "B4": (81, 13)}},
]

IMPROVEMENTS = {
    "Adjacency F1": "+2.4%",
    "Daylight proxy": "+2.9%",
    "Interior usability": "+4.1%",
    "Circulation": "-6.5%",
    "Span-feasible rooms": "+1.1%",
    "Load-path plausibility": "+1.1%",
    "MEP length": "-13.8%",
    "Pressure-drop proxy": "-19.5%",
    "Voltage-drop proxy": "-13.0%",
    "Clashes after repair": "-53.8%",
    "IFC completeness": "+1.1%",
    "IoT coverage": "+4.5%",
    "IoT latency": "-22.2%",
    "IoT risk": "-19.4%",
    "Cost error": "-9.7%",
    "EUI proxy": "-2.5%",
}

RADAR_NORMALIZED = {
    "Adjacency F1": {"Proposed": 1.00, "B1": 0.00, "B2": 0.43, "B3": 0.29, "B4": 0.71},
    "Interior Usability": {"Proposed": 1.00, "B1": 0.00, "B2": 0.50, "B3": 0.36, "B4": 0.79},
    "MEP Efficiency": {"Proposed": 1.00, "B1": 0.49, "B2": 0.67, "B3": 0.57, "B4": 0.00},
    "Clash Reduction": {"Proposed": 1.00, "B1": 0.00, "B2": 0.47, "B3": 0.40, "B4": 0.61},
    "IoT Coverage": {"Proposed": 1.00, "B1": 0.00, "B2": 0.42, "B3": 0.33, "B4": 0.67},
    "Energy Efficiency": {"Proposed": 1.00, "B1": 0.00, "B2": 0.41, "B3": 0.17, "B4": 0.76},
}

PARETO_POINTS = [
    {"id": 1, "EUI": 62, "cost_lakhs": 38.2, "clashes": 3, "comfort": 0.18},
    {"id": 2, "EUI": 65, "cost_lakhs": 36.5, "clashes": 2, "comfort": 0.20},
    {"id": 3, "EUI": 68, "cost_lakhs": 35.1, "clashes": 2, "comfort": 0.22},
    {"id": 4, "EUI": 71, "cost_lakhs": 34.0, "clashes": 2, "comfort": 0.23},
    {"id": 5, "EUI": 73, "cost_lakhs": 33.2, "clashes": 1, "comfort": 0.25},
    {"id": 6, "EUI": 75, "cost_lakhs": 32.5, "clashes": 2, "comfort": 0.26},
    {"id": 7, "EUI": 77, "cost_lakhs": 31.8, "clashes": 1, "comfort": 0.28},
    {"id": 8, "EUI": 79, "cost_lakhs": 31.0, "clashes": 2, "comfort": 0.29},
    {"id": 9, "EUI": 81, "cost_lakhs": 30.4, "clashes": 3, "comfort": 0.31},
    {"id": 10, "EUI": 83, "cost_lakhs": 29.9, "clashes": 2, "comfort": 0.33},
    {"id": 11, "EUI": 85, "cost_lakhs": 29.2, "clashes": 3, "comfort": 0.35},
    {"id": 12, "EUI": 87, "cost_lakhs": 28.7, "clashes": 4, "comfort": 0.36},
    {"id": 13, "EUI": 89, "cost_lakhs": 28.1, "clashes": 3, "comfort": 0.38},
    {"id": 14, "EUI": 91, "cost_lakhs": 27.6, "clashes": 4, "comfort": 0.40},
    {"id": 15, "EUI": 93, "cost_lakhs": 27.0, "clashes": 4, "comfort": 0.42},
    {"id": 16, "EUI": 95, "cost_lakhs": 26.5, "clashes": 5, "comfort": 0.44},
    {"id": 17, "EUI": 97, "cost_lakhs": 26.0, "clashes": 5, "comfort": 0.46},
    {"id": 18, "EUI": 100, "cost_lakhs": 25.4, "clashes": 6, "comfort": 0.49},
    {"id": 19, "EUI": 104, "cost_lakhs": 24.9, "clashes": 6, "comfort": 0.52},
    {"id": 20, "EUI": 108, "cost_lakhs": 24.2, "clashes": 7, "comfort": 0.56},
]

GENERATIONS = list(range(1, 51))
HYPERVOLUME = [
    0.12, 0.19, 0.27, 0.34, 0.40, 0.45, 0.50, 0.54, 0.57, 0.60,
    0.63, 0.65, 0.67, 0.69, 0.71, 0.72, 0.74, 0.75, 0.76, 0.77,
    0.78, 0.79, 0.80, 0.80, 0.81, 0.82, 0.82, 0.83, 0.83, 0.84,
    0.84, 0.85, 0.85, 0.85, 0.86, 0.86, 0.86, 0.87, 0.87, 0.87,
    0.87, 0.88, 0.88, 0.88, 0.88, 0.88, 0.89, 0.89, 0.89, 0.89,
]

np.random.seed(42)
SA_ITERS = list(range(0, 1000, 10))
SA_ENERGY = []
_energy = 1.0
for _ in range(100):
    _energy = _energy * 0.97 + np.random.normal(0, 0.01)
    _energy = max(0.18, _energy)
    SA_ENERGY.append(_energy)
SA_VIOLATIONS = [max(1, int(12 * np.exp(-i / 30) + np.random.normal(0, 0.5))) for i in range(100)]

DEFAULT_ROOMS = {
    "Living Room": {"x": 0.0, "y": 0.0, "w": 5.0, "h": 5.0, "area": 25, "color": "#AED6F1"},
    "Kitchen": {"x": 5.0, "y": 0.0, "w": 4.0, "h": 3.0, "area": 12, "color": "#A9DFBF"},
    "Master Bedroom": {"x": 0.0, "y": 5.0, "w": 4.5, "h": 4.0, "area": 18, "color": "#F9E79F"},
    "Bedroom 2": {"x": 4.5, "y": 5.0, "w": 3.5, "h": 4.0, "area": 14, "color": "#FDEBD0"},
    "Bathroom 1": {"x": 8.0, "y": 5.0, "w": 2.0, "h": 3.0, "area": 6, "color": "#D2B4DE"},
    "Bathroom 2": {"x": 8.0, "y": 8.0, "w": 2.0, "h": 2.5, "area": 5, "color": "#D2B4DE"},
    "Corridor": {"x": 4.5, "y": 0.0, "w": 3.5, "h": 5.0, "area": 8, "color": "#EAECEE"},
    "Garage": {"x": 0.0, "y": 9.0, "w": 5.0, "h": 3.0, "area": 15, "color": "#BFC9CA"},
    "Study": {"x": 5.0, "y": 9.0, "w": 3.0, "h": 3.3, "area": 10, "color": "#FAD7A0"},
}

REQUIRED_EDGES = [
    ("Living Room", "Kitchen"),
    ("Living Room", "Corridor"),
    ("Corridor", "Master Bedroom"),
    ("Corridor", "Bedroom 2"),
    ("Corridor", "Bathroom 1"),
    ("Master Bedroom", "Bathroom 2"),
]
FORBIDDEN_EDGES = [
    ("Kitchen", "Master Bedroom"),
    ("Garage", "Living Room"),
    ("Bathroom 1", "Kitchen"),
]

FURNITURE_LAYOUTS = {
    "Living Room": {
        "w": 5.0,
        "h": 5.0,
        "color": "#AED6F1",
        "items": [
            {"name": "Sofa", "x": 0.5, "y": 0.5, "w": 2.2, "h": 0.9, "color": "#3A86FF", "clearance": 0.6},
            {"name": "Coffee Table", "x": 1.5, "y": 1.8, "w": 1.2, "h": 0.6, "color": "#90E0EF"},
            {"name": "TV Unit", "x": 0.3, "y": 4.2, "w": 1.8, "h": 0.4, "color": "#264653"},
            {"name": "Armchair", "x": 3.8, "y": 0.8, "w": 0.8, "h": 0.8, "color": "#FFB703"},
        ],
        "egress": [(0.3, 2.8), (4.7, 2.8)],
    },
    "Master Bedroom": {
        "w": 4.5,
        "h": 4.0,
        "color": "#F9E79F",
        "items": [
            {"name": "Bed", "x": 1.0, "y": 1.0, "w": 2.0, "h": 1.6, "color": "#E76F51", "clearance": 0.9},
            {"name": "Wardrobe", "x": 0.2, "y": 0.2, "w": 2.0, "h": 0.6, "color": "#6D597A"},
            {"name": "Bedside 1", "x": 0.8, "y": 2.8, "w": 0.5, "h": 0.5, "color": "#F4A261"},
            {"name": "Bedside 2", "x": 3.0, "y": 2.8, "w": 0.5, "h": 0.5, "color": "#F4A261"},
            {"name": "Desk", "x": 3.5, "y": 0.5, "w": 1.2, "h": 0.6, "color": "#2A9D8F"},
        ],
        "egress": [(0.2, 3.6), (4.2, 3.6)],
    },
}

DEVICE_CATALOG = [
    {"name": "Gateway", "type": "Gateway", "x": 10, "y": 10, "sigma": 4.0, "icon": "📡"},
    {"name": "Motion Sensor 1", "type": "Motion Sensor", "x": 3, "y": 3, "sigma": 2.0, "icon": "👁"},
    {"name": "Motion Sensor 2", "type": "Motion Sensor", "x": 17, "y": 4, "sigma": 2.0, "icon": "👁"},
    {"name": "Smoke Detector", "type": "Smoke Detector", "x": 10, "y": 2, "sigma": 2.5, "icon": "🔥"},
    {"name": "Camera 1", "type": "Camera", "x": 2, "y": 18, "sigma": 3.0, "icon": "📷"},
    {"name": "Temp Sensor", "type": "Temperature Sensor", "x": 5, "y": 15, "sigma": 1.8, "icon": "🌡"},
    {"name": "Smart Plug", "type": "Smart Plug", "x": 15, "y": 15, "sigma": 0.8, "icon": "🔌"},
    {"name": "Camera 2", "type": "Camera", "x": 17, "y": 16, "sigma": 3.0, "icon": "📷"},
    {"name": "Motion Sensor 3", "type": "Motion Sensor", "x": 12, "y": 6, "sigma": 2.0, "icon": "👁"},
    {"name": "Smoke Detector 2", "type": "Smoke Detector", "x": 6, "y": 10, "sigma": 2.5, "icon": "🔥"},
    {"name": "Temp Sensor 2", "type": "Temperature Sensor", "x": 12, "y": 15, "sigma": 1.8, "icon": "🌡"},
    {"name": "Smart Plug 2", "type": "Smart Plug", "x": 4, "y": 14, "sigma": 0.8, "icon": "🔌"},
]

IFC_ELEMENTS = [
    {"Element Type": "IfcWall", "Count": 85, "Completeness %": 98},
    {"Element Type": "IfcSlab", "Count": 12, "Completeness %": 99},
    {"Element Type": "IfcDoor", "Count": 14, "Completeness %": 97},
    {"Element Type": "IfcWindow", "Count": 18, "Completeness %": 96},
    {"Element Type": "IfcFlowSegment", "Count": 60, "Completeness %": 94},
    {"Element Type": "IfcDistribElement", "Count": 30, "Completeness %": 93},
    {"Element Type": "IfcSensor", "Count": 12, "Completeness %": 95},
    {"Element Type": "IfcSpace", "Count": 9, "Completeness %": 99},
]

BOQ_DATA = [
    ["RCC Walls", 180, "sqm", 2200, 396000],
    ["Flooring Slab", 120, "sqm", 3500, 420000],
    ["Doors", 14, "no.", 18000, 252000],
    ["Windows", 18, "no.", 12000, 216000],
    ["Plumbing pipes", 131, "m", 850, 111350],
    ["Electrical conduit", 98, "m", 420, 41160],
    ["HVAC ducts", 85, "m", 1200, 102000],
    ["IoT devices", 12, "no.", 8500, 102000],
    ["Gateway", 1, "no.", 25000, 25000],
    ["Interior fittings", 120, "sqm", 4500, 540000],
]

AUDIT_LOG = [
    "Intent parser resolved 24 active constraints and normalized user weights.",
    "Layout generator emitted 9-room candidate graph with facade preference tags.",
    "Civil envelope check passed setbacks and egress width; daylight proxy above 0.70.",
    "Structural check flagged 2 long spans, then auto-shifted partitions to restore 92% span feasibility.",
    "Interior SA repair removed 11 initial furniture clearance conflicts and converged to 1 residual violation.",
    "MEP repair rerouted plumbing around electrical trunk, reducing clash count from 5 to 2.",
    "IoT planner rejected two bedroom camera placements due to privacy penalty.",
    "IFC export completed with 96% weighted completeness and BOQ cost error proxy of 6.5%.",
]

PIPELINE_NODES = [
    "Intent Parser",
    "Layout Generator",
    "Civil/Struct Checks",
    "Interior Optimizer",
    "MEP Router",
    "IoT Planner",
    "BIM/IFC + BOQ",
    "UI Loop",
]

CELL_LABELS = {0: "Open", 1: "Wall", 2: "Shaft", 3: "Pipe", 4: "Electrical", 5: "Duct", 6: "Clash"}


def init_session_state() -> None:
    defaults = {
        "plot_size": 120,
        "budget": 48,
        "style": "Modern",
        "occupants": 4,
        "climate": "Temperate",
        "seed": 42,
        "layout_metrics": {"adjacency_f1": 0.85, "daylight": 0.71, "circulation": 4.3, "span_feasible": 92},
        "layout_rooms": None,
        "mep_repaired": False,
        "iot_devices": None,
        "boq_currency": "INR",
        "sa_completed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def themed_figure(fig: go.Figure, title: str = "", height: int = 480) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#EAF4FF"),
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def indicator_figure(title: str, value: float, min_val: float, max_val: float, suffix: str = "", steps=None, color=ACCENT) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            number={"suffix": suffix},
            gauge={
                "axis": {"range": [min_val, max_val]},
                "bar": {"color": color},
                "bgcolor": PLOT_BG,
                "bordercolor": "rgba(255,255,255,0.12)",
                "steps": steps or [],
            },
        )
    )
    return themed_figure(fig, height=280)


def progress_card(label: str, value: float, max_value: float = 1.0) -> None:
    ratio = min(max(value / max_value, 0.0), 1.0)
    st.markdown(f"**{label}**")
    st.progress(ratio, text=f"{value:.2f}" if max_value == 1.0 else f"{value:.1f} / {max_value:.1f}")


def increment_seed() -> None:
    st.session_state.seed += 1


def build_program(plot_size: int, occupants: int) -> Tuple[Dict[str, float], float]:
    base = {
        "Living": 25.0,
        "Kitchen": 12.0,
        "Master Bed": 18.0,
        "Bed 2": 14.0,
        "Bath 1": 6.0,
        "Bath 2": 5.0,
        "Corridor": 8.0,
        "Garage": 15.0,
        "Study": 10.0,
    }
    multiplier = plot_size / 120.0
    program = {room: round(area * multiplier, 1) for room, area in base.items()}
    if occupants >= 5:
        program["Living"] = round(program["Living"] + 2.0, 1)
        program["Bed 2"] = round(program["Bed 2"] + 1.5, 1)
        program["Study"] = round(max(8.0, program["Study"] - 1.0), 1)
    elif occupants <= 2:
        program["Bed 2"] = round(max(10.0, program["Bed 2"] - 2.0), 1)
        program["Study"] = round(program["Study"] + 1.0, 1)
    total = round(sum(program.values()), 1)
    reserve = round(max(plot_size - total, 0.0), 1)
    return program, reserve


def scale_rooms(plot_size: int) -> Tuple[Dict[str, dict], float, float]:
    base_w = max(room["x"] + room["w"] for room in DEFAULT_ROOMS.values())
    base_h = max(room["y"] + room["h"] for room in DEFAULT_ROOMS.values())
    scale = math.sqrt(plot_size / 120.0)
    rooms = {}
    for name, room in DEFAULT_ROOMS.items():
        rooms[name] = {
            **room,
            "x": round(room["x"] * scale, 2),
            "y": round(room["y"] * scale, 2),
            "w": round(room["w"] * scale, 2),
            "h": round(room["h"] * scale, 2),
            "area": round(room["area"] * plot_size / 120.0, 1),
        }
    return rooms, round(base_w * scale, 2), round(base_h * scale, 2)


def generate_layout_metrics(seed: int) -> Dict[str, float]:
    if seed == 42:
        return {"adjacency_f1": 0.85, "daylight": 0.71, "circulation": 4.3, "span_feasible": 92}
    rng = np.random.default_rng(seed)
    return {
        "adjacency_f1": round(float(np.clip(0.82 + rng.normal(0.0, 0.025), 0.74, 0.88)), 2),
        "daylight": round(float(np.clip(0.68 + rng.normal(0.0, 0.03), 0.58, 0.75)), 2),
        "circulation": round(float(np.clip(4.5 + rng.normal(0.0, 0.35), 4.0, 5.4)), 2),
        "span_feasible": int(np.clip(90 + rng.normal(0.0, 2.0), 84, 95)),
    }


def generate_random_layout(seed: int, plot_size: int) -> Tuple[Dict[str, dict], float, float]:
    if seed == 42:
        return scale_rooms(plot_size)

    rooms, boundary_w, base_h = scale_rooms(plot_size)
    rng = np.random.default_rng(seed)
    order = list(rooms.keys())
    rng.shuffle(order)
    packed = {}
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0
    gap = max(0.18, 0.2 * math.sqrt(plot_size / 120.0))
    boundary_w = round(boundary_w * 1.08, 2)

    for name in order:
        room = rooms[name]
        if x_cursor + room["w"] > boundary_w:
            x_cursor = 0.0
            y_cursor = round(y_cursor + row_height + gap, 2)
            row_height = 0.0
        jitter_x = float(rng.uniform(0.0, gap * 0.25))
        jitter_y = float(rng.uniform(0.0, gap * 0.18))
        packed[name] = {**room, "x": round(x_cursor + jitter_x, 2), "y": round(y_cursor + jitter_y, 2)}
        x_cursor = round(x_cursor + room["w"] + gap, 2)
        row_height = max(row_height, room["h"])

    boundary_h = round(max(base_h, y_cursor + row_height + gap), 2)
    return packed, boundary_w, boundary_h


def room_centers(rooms: Dict[str, dict]) -> Dict[str, Tuple[float, float]]:
    return {name: (room["x"] + room["w"] / 2, room["y"] + room["h"] / 2) for name, room in rooms.items()}


def floorplan_figure(rooms: Dict[str, dict], boundary_w: float, boundary_h: float) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=boundary_w, y1=boundary_h, line=dict(color="#BFD7EA", width=3))

    centers = room_centers(rooms)
    for name, room in rooms.items():
        fig.add_shape(
            type="rect",
            x0=room["x"],
            y0=room["y"],
            x1=room["x"] + room["w"],
            y1=room["y"] + room["h"],
            line=dict(color="#EAF4FF", width=1),
            fillcolor=room["color"],
            opacity=0.92,
        )
        fig.add_annotation(
            x=centers[name][0],
            y=centers[name][1],
            text=f"<b>{name}</b><br>{room['area']:.1f} m²",
            showarrow=False,
            font=dict(color="#081320", size=12),
        )

    graph = nx.Graph()
    graph.add_nodes_from(rooms.keys())
    for edge in REQUIRED_EDGES:
        graph.add_edge(*edge, kind="required")
    for edge in FORBIDDEN_EDGES:
        graph.add_edge(*edge, kind="forbidden")

    for u, v, attrs in graph.edges(data=True):
        x0, y0 = centers[u]
        x1, y1 = centers[v]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="#56F39A" if attrs["kind"] == "required" else DANGER, width=3 if attrs["kind"] == "required" else 2, dash="solid" if attrs["kind"] == "required" else "dash"),
                hovertemplate=f"{u} ↔ {v}<extra>{attrs['kind']}</extra>",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[center[0] for center in centers.values()],
            y=[center[1] for center in centers.values()],
            mode="markers",
            marker=dict(size=7, color="#FFFFFF", line=dict(color=PRIMARY, width=1)),
            hovertext=list(centers.keys()),
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_xaxes(showgrid=False, zeroline=False, range=[-0.5, boundary_w + 0.5], title="Plan Width (m)")
    fig.update_yaxes(showgrid=False, zeroline=False, range=[-0.5, boundary_h + 0.5], scaleanchor="x", scaleratio=1, title="Plan Depth (m)")
    return themed_figure(fig, "Room Layout + Adjacency Graph", height=620)


def furniture_figure(room_name: str) -> go.Figure:
    config = FURNITURE_LAYOUTS[room_name]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=config["w"], y1=config["h"], line=dict(color="#EAF4FF", width=3), fillcolor=config["color"], opacity=0.4)

    for item in config["items"]:
        fig.add_shape(
            type="rect",
            x0=item["x"],
            y0=item["y"],
            x1=item["x"] + item["w"],
            y1=item["y"] + item["h"],
            line=dict(color="#EAF4FF", width=1.5),
            fillcolor=item["color"],
            opacity=0.95,
        )
        fig.add_annotation(x=item["x"] + item["w"] / 2, y=item["y"] + item["h"] / 2, text=item["name"].replace(" ", "<br>"), showarrow=False, font=dict(size=11, color="#FFFFFF"))
        if "clearance" in item:
            fig.add_shape(
                type="rect",
                x0=max(0, item["x"] - item["clearance"]),
                y0=max(0, item["y"] - item["clearance"]),
                x1=min(config["w"], item["x"] + item["w"] + item["clearance"]),
                y1=min(config["h"], item["y"] + item["h"] + item["clearance"]),
                line=dict(color=WARNING, width=2, dash="dash"),
                fillcolor="rgba(0,0,0,0)",
            )

    egress = config["egress"]
    fig.add_annotation(
        x=egress[1][0],
        y=egress[1][1],
        ax=egress[0][0],
        ay=egress[0][1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=4,
        arrowsize=1.3,
        arrowwidth=5,
        arrowcolor="#56F39A",
        text="1.2m egress",
        font=dict(color="#56F39A"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, range=[-0.2, config["w"] + 0.2], title="Width (m)")
    fig.update_yaxes(showgrid=False, zeroline=False, range=[-0.2, config["h"] + 0.2], scaleanchor="x", scaleratio=1, title="Depth (m)")
    return themed_figure(fig, f"{room_name} Furniture Packing", height=520)


def best_baseline(metric: dict) -> Tuple[str, float]:
    baseline_values = {system: metric["values"][system][0] for system in SYSTEMS if system != "Proposed"}
    system = max(baseline_values, key=baseline_values.get) if metric["direction"] == "up" else min(baseline_values, key=baseline_values.get)
    return system, baseline_values[system]


def winners_count() -> int:
    wins = 0
    for metric in BENCHMARK_METRICS:
        values = {system: metric["values"][system][0] for system in SYSTEMS}
        winner = max(values, key=values.get) if metric["direction"] == "up" else min(values, key=values.get)
        wins += int(winner == "Proposed")
    return wins


def discrete_heatmap(z: np.ndarray, color_map: Dict[int, str], title: str, text: np.ndarray = None, height: int = 540) -> go.Figure:
    zmax = max(color_map)
    scale = []
    for key, color in color_map.items():
        start = key / (zmax + 1)
        end = (key + 1) / (zmax + 1)
        scale.append([start, color])
        scale.append([end, color])
    fig = go.Figure(go.Heatmap(z=z, colorscale=scale, zmin=0, zmax=zmax, text=text, hovertemplate="x=%{x}<br>y=%{y}<br>%{text}<extra></extra>", showscale=False))
    fig.update_xaxes(showgrid=False, zeroline=False, tickmode="linear", dtick=1)
    fig.update_yaxes(showgrid=False, zeroline=False, tickmode="linear", dtick=1, autorange="reversed")
    return themed_figure(fig, title, height=height)


def map_room_to_grid(room: dict, total_w: float, total_h: float) -> Tuple[int, int, int, int]:
    sx = 20 / total_w
    sy = 20 / total_h
    x0 = int(np.clip(round(room["x"] * sx), 0, 19))
    y0 = int(np.clip(round(room["y"] * sy), 0, 19))
    x1 = int(np.clip(round((room["x"] + room["w"]) * sx), 1, 19))
    y1 = int(np.clip(round((room["y"] + room["h"]) * sy), 1, 19))
    return x0, y0, x1, y1


def mep_grid(repaired: bool) -> Tuple[np.ndarray, np.ndarray]:
    rooms, total_w, total_h = scale_rooms(120)
    grid = np.zeros((20, 20), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    for room in rooms.values():
        x0, y0, x1, y1 = map_room_to_grid(room, total_w, total_h)
        grid[y0:y1 + 1, x0] = 1
        grid[y0:y1 + 1, x1] = 1
        grid[y0, x0:x1 + 1] = 1
        grid[y1, x0:x1 + 1] = 1

    shaft_cells = [(14, 3), (18, 11), (18, 15)]
    for x, y in shaft_cells:
        grid[y, x] = 2

    pipe_route = [(14, y) for y in range(3, 12)] + [(x, 11) for x in range(14, 19)] + [(18, y) for y in range(11, 16)]
    electrical_before = [(x, 7) for x in range(1, 19)] + [(17, y) for y in range(7, 18)] + [(x, 17) for x in range(8, 18)]
    electrical_after = [(x, 6) for x in range(1, 16)] + [(15, y) for y in range(6, 18)] + [(x, 17) for x in range(10, 18)]
    duct_route = [(10, y) for y in range(2, 18)] + [(x, 9) for x in range(2, 18)] + [(x, 13) for x in range(4, 18)]
    electrical_route = electrical_after if repaired else electrical_before
    clashes = [(14, 7), (17, 14)] if not repaired else []

    for x, y in pipe_route:
        if grid[y, x] == 0:
            grid[y, x] = 3
    for x, y in electrical_route:
        if grid[y, x] == 0:
            grid[y, x] = 4
        elif grid[y, x] in (3, 5):
            grid[y, x] = 6
    for x, y in duct_route:
        if grid[y, x] == 0:
            grid[y, x] = 5
        elif grid[y, x] in (3, 4):
            grid[y, x] = 6
    for x, y in shaft_cells:
        grid[y, x] = 2
    for x, y in clashes:
        grid[y, x] = 6
    labels = np.vectorize(CELL_LABELS.get)(grid)
    return grid, labels


def coverage_heatmap(devices: List[dict]) -> Tuple[np.ndarray, float]:
    xs, ys = np.meshgrid(np.arange(20), np.arange(20))
    signal = np.zeros((20, 20), dtype=float)
    for device in devices:
        dx = (xs - device["x"]) * 0.5
        dy = (ys - device["y"]) * 0.5
        dist_sq = dx ** 2 + dy ** 2
        current = np.exp(-dist_sq / (2 * (device["sigma"] ** 2)))
        signal = np.maximum(signal, current)
    coverage = float((signal >= 0.6).sum() / signal.size)
    return signal, coverage


def privacy_rectangles() -> List[dict]:
    return [
        {"name": "Master Bedroom", "x0": 0, "y0": 11, "x1": 8, "y1": 18},
        {"name": "Bedroom 2", "x0": 8, "y0": 11, "x1": 14, "y1": 18},
        {"name": "Bathroom 1", "x0": 14, "y0": 11, "x1": 18, "y1": 14},
        {"name": "Bathroom 2", "x0": 16, "y0": 14, "x1": 19, "y1": 18},
    ]


def device_in_privacy(device: dict) -> bool:
    for zone in privacy_rectangles():
        if zone["x0"] <= device["x"] <= zone["x1"] and zone["y0"] <= device["y"] <= zone["y1"]:
            return True
    return False


def active_devices(count: int) -> List[dict]:
    if st.session_state.iot_devices is None:
        st.session_state.iot_devices = [device.copy() for device in DEVICE_CATALOG]
    return [device.copy() for device in st.session_state.iot_devices[:count]]


def optimize_devices(count: int, penalty_on: bool) -> List[dict]:
    rng = np.random.default_rng(st.session_state.seed)
    base = [device.copy() for device in DEVICE_CATALOG[:count]]
    best_devices = [device.copy() for device in base]
    _, best_coverage = coverage_heatmap(best_devices)
    best_penalty = sum(device_in_privacy(device) for device in best_devices) * (0.3 if penalty_on else 0.0)
    best_score = best_coverage - best_penalty
    for _ in range(50):
        candidate = [device.copy() for device in base]
        for device in candidate[1:]:
            device["x"] = int(rng.integers(1, 19))
            device["y"] = int(rng.integers(1, 19))
        _, coverage = coverage_heatmap(candidate)
        penalty = sum(device_in_privacy(device) for device in candidate) * (0.3 if penalty_on else 0.0)
        score = coverage - penalty
        if score > best_score:
            best_score = score
            best_devices = candidate
    st.session_state.iot_devices = [device.copy() for device in DEVICE_CATALOG]
    for i, device in enumerate(best_devices):
        st.session_state.iot_devices[i] = device
    return best_devices


def iot_figure(signal: np.ndarray, devices: List[dict]) -> go.Figure:
    fig = px.imshow(signal, color_continuous_scale="RdYlGn", zmin=0.0, zmax=1.0, origin="lower", aspect="equal")
    for zone in privacy_rectangles():
        fig.add_shape(
            type="rect",
            x0=zone["x0"] - 0.5,
            y0=zone["y0"] - 0.5,
            x1=zone["x1"] + 0.5,
            y1=zone["y1"] + 0.5,
            line=dict(color="rgba(255,120,120,0.9)", width=2, dash="dot"),
            fillcolor="rgba(231,111,81,0.12)",
        )
        fig.add_annotation(x=(zone["x0"] + zone["x1"]) / 2, y=zone["y1"] + 0.6, text=zone["name"], showarrow=False, font=dict(size=10))
    for device in devices:
        fig.add_annotation(x=device["x"], y=device["y"], text=device["icon"], showarrow=False, font=dict(size=18))
    fig.update_traces(hovertemplate="x=%{x}<br>y=%{y}<br>Signal=%{z:.2f}<extra></extra>")
    fig.update_xaxes(title="Grid X (cells)")
    fig.update_yaxes(title="Grid Y (cells)")
    return themed_figure(fig, "IoT Coverage Heatmap", height=600)


def climate_profile(climate: str) -> Tuple[np.ndarray, float]:
    hours = np.arange(24)
    params = {
        "Hot-Dry": {"T_mean": 38, "T_amp": 7, "indoor_lag": -6, "offset": 3},
        "Temperate": {"T_mean": 21, "T_amp": 7, "indoor_lag": 0, "offset": 0},
        "Cold": {"T_mean": 1, "T_amp": 6, "indoor_lag": -4, "offset": 5},
    }[climate]
    temps = params["T_mean"] + params["T_amp"] * np.sin(2 * np.pi * (hours - 8) / 24) + params["indoor_lag"] + params["offset"]
    comfort = np.mean(np.maximum(0, np.abs(temps - 22) - 2))
    return temps, float(comfort)


def energy_profile_figure(climate: str) -> Tuple[go.Figure, int, float]:
    hours = np.arange(24)
    temps, comfort = climate_profile(climate)
    violations = (temps < 20) | (temps > 24)
    fig = go.Figure()
    fig.add_hrect(y0=20, y1=24, fillcolor="rgba(46,196,182,0.15)", line_width=0, annotation_text="Comfort Band 20–24°C", annotation_position="top left")
    fig.add_trace(go.Scatter(x=hours, y=temps, mode="lines+markers", line=dict(color=ACCENT, width=3), name="Operative Temp"))
    fig.add_trace(go.Scatter(x=hours[violations], y=temps[violations], mode="markers", marker=dict(color=DANGER, size=10), name="Violations"))
    fig.update_xaxes(title="Hour")
    fig.update_yaxes(title="T_op (°C)")
    fig.add_annotation(x=14, y=float(np.max(temps)), text="Peak", showarrow=True, arrowcolor="#FFFFFF")
    themed_figure(fig, f"24-hour Operative Temperature Profile • {climate}", height=420)
    return fig, int(violations.sum()), comfort


def grouped_bar_chart(metrics_subset: List[dict], title: str) -> go.Figure:
    fig = go.Figure()
    x = [f"{metric['metric']} {'↑' if metric['direction'] == 'up' else '↓'}" for metric in metrics_subset]
    for system in SYSTEMS:
        fig.add_trace(
            go.Bar(
                name=system,
                x=x,
                y=[metric["values"][system][0] for metric in metrics_subset],
                marker_color=SYSTEM_COLORS[system],
                error_y=dict(type="data", array=[metric["values"][system][1] for metric in metrics_subset]),
            )
        )
    fig.update_xaxes(title="Metrics")
    fig.update_yaxes(title="Value")
    fig.update_layout(barmode="group")
    return themed_figure(fig, title, height=520)


def radar_chart() -> go.Figure:
    axes = list(RADAR_NORMALIZED.keys())
    fig = go.Figure()
    for system in SYSTEMS:
        values = [RADAR_NORMALIZED[axis][system] for axis in axes]
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(r=values, theta=axes + [axes[0]], fill="toself", opacity=0.2, line=dict(color=SYSTEM_COLORS[system], width=2), name=system))
    fig.update_polars(radialaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.15)"))
    return themed_figure(fig, "Normalized Multi-domain Radar", height=560)


def ranking_heatmap(metrics_subset: List[dict]) -> go.Figure:
    z = []
    text = []
    for metric in metrics_subset:
        values = {system: metric["values"][system][0] for system in SYSTEMS}
        best_val = max(values.values()) if metric["direction"] == "up" else min(values.values())
        worst_val = min(values.values()) if metric["direction"] == "up" else max(values.values())
        row = []
        label_row = []
        for system in SYSTEMS:
            value = values[system]
            row.append(1 if value == best_val else -1 if value == worst_val else 0)
            label_row.append(str(value))
        z.append(row)
        text.append(label_row)
    fig = go.Figure(go.Heatmap(z=z, x=SYSTEMS, y=[metric["metric"] for metric in metrics_subset], text=text, texttemplate="%{text}", colorscale=[[0, "#7F1D1D"], [0.5, WARNING], [1, "#1F9D55"]], zmin=-1, zmax=1, showscale=False))
    return themed_figure(fig, "Best / Middle / Worst Benchmark Heatmap", height=560)


def delta_badges(metrics_subset: List[dict]) -> None:
    cols = st.columns(4)
    for idx, metric in enumerate(metrics_subset):
        col = cols[idx % 4]
        baseline_system, baseline_value = best_baseline(metric)
        proposed_value = metric["values"]["Proposed"][0]
        arrow = "↑" if metric["direction"] == "up" else "↓"
        col.markdown(
            f"""
            <div class="delta-card">
                <div style="font-size:0.95rem;font-weight:700;">{metric['metric']}</div>
                <div style="margin-top:0.35rem;">Proposed: <b>{proposed_value}{metric['unit']}</b></div>
                <div>Best baseline: <b>{baseline_system} = {baseline_value}{metric['unit']}</b></div>
                <div style="margin-top:0.5rem;color:#8ff4e8;font-weight:700;">{arrow} {IMPROVEMENTS[metric['metric']]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def boq_dataframe(currency: str) -> pd.DataFrame:
    df = pd.DataFrame(BOQ_DATA, columns=["Element", "Qty", "Unit", "Rate", "Total"])
    if currency == "USD":
        df["Rate"] = (df["Rate"] / 83.5).round(2)
        df["Total"] = (df["Total"] / 83.5).round(2)
    return df


def currency_symbol(currency: str) -> str:
    return "₹" if currency == "INR" else "$"


init_session_state()
if st.session_state.layout_rooms is None:
    st.session_state.layout_rooms, _, _ = scale_rooms(st.session_state.plot_size)


st.sidebar.title("🏠 AI House Planner")
st.sidebar.caption("Multi-domain Co-design System")
st.sidebar.divider()
st.sidebar.slider("Plot size (m²)", 80, 200, key="plot_size")
st.sidebar.slider("Budget (₹ Lakhs)", 30, 80, key="budget")
st.sidebar.selectbox("Style", ["Modern", "Traditional", "Minimalist"], key="style")
st.sidebar.slider("Occupants", 1, 6, key="occupants")
st.sidebar.selectbox("Climate", ["Hot-Dry", "Temperate", "Cold"], key="climate")
st.sidebar.divider()
st.sidebar.markdown(
    """
    <div class="sidebar-summary">
        <b>Summary</b><br>
        Active constraints: <b>24</b><br>
        Candidate plans generated: <b>500</b><br>
        Pareto front size: <b>20</b><br>
        Current system score: <b>J = 0.847</b>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

PAGES = [
    "🏠 System Overview & Pipeline",
    "🗺️ Floor Plan Layout Generator",
    "🪑 Interior Arrangement",
    "🔧 MEP Routing & Clash Detection",
    "📡 IoT Placement & Coverage Heatmap",
    "⚡ Energy & Comfort Screening",
    "📊 Results Dashboard",
    "🏗️ BIM/IFC Export & BOQ",
]
page = st.sidebar.radio("Navigation", PAGES)


# --- PAGE 1: System Overview & Pipeline ---
if page == "🏠 System Overview & Pipeline":
    st.info("Paper view: Table I system modules, Eq. 1 multi-objective co-design, and Eq. 2 weighted ranking loop.")
    st.title("🏠 AI-Based Intelligent House Planning System")
    st.markdown(
        f"""
        <div class="paper-card">
            The app simulates the paper's full pipeline with hardcoded research values. Current session uses a
            <b>{st.session_state.plot_size} m²</b> plot, <b>₹{st.session_state.budget}L</b> budget,
            <b>{st.session_state.style}</b> style, <b>{st.session_state.occupants}</b> occupants, and
            <b>{st.session_state.climate}</b> climate.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sankey_fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=18,
                thickness=18,
                line=dict(color="rgba(255,255,255,0.14)", width=1),
                label=PIPELINE_NODES,
                color=[PRIMARY, "#1B4965", "#1F6F8B", "#2A9D8F", "#3FA7D6", "#5893D4", ACCENT, WARNING],
            ),
            link=dict(
                source=list(range(len(PIPELINE_NODES) - 1)),
                target=list(range(1, len(PIPELINE_NODES))),
                value=[8, 8, 7, 7, 6, 6, 5],
                color="rgba(46,196,182,0.35)",
            ),
        )
    )
    st.plotly_chart(themed_figure(sankey_fig, "Eight-stage Planning Pipeline", height=420), use_container_width=True)

    st.subheader("Module Inputs / Outputs")
    st.dataframe(pd.DataFrame(MODULE_IO), use_container_width=True, hide_index=True)

    program, reserve = build_program(st.session_state.plot_size, st.session_state.occupants)
    p1_col1, p1_col2 = st.columns([1.15, 0.85])
    with p1_col1:
        st.subheader("Derived Program")
        program_cols = st.columns(3)
        for idx, (room, area) in enumerate(program.items()):
            program_cols[idx % 3].markdown(
                f"""
                <div class="paper-card">
                    <div style="font-size:0.9rem;color:#BFD7EA;">{room}</div>
                    <div style="font-size:1.4rem;font-weight:700;">{area:.1f} m²</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""
            <div class="paper-card">
                <b>Reserve / service band:</b> {reserve:.1f} m²<br>
                Wet cores: Kitchen, Bathroom 1, Bathroom 2<br>
                Facade-preferring rooms: Living, Master Bedroom, Kitchen
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p1_col2:
        st.subheader("Objective Functions")
        st.code(
            "min f(x) = [f_cost, f_EUI, f_comfort, f_light, f_flow, f_clash]\n"
            "subject to g(x) ≤ 0\n"
            "Solved via NSGA-II Pareto optimization\n\n"
            "J(x; w) = Σ w_k · f̃_k(x),   Σw_k = 1",
            language="text",
        )
        st.markdown(
            """
            <div class="paper-card">
                <b>Experiment setup</b><br>
                500 generated single-storey plans<br>
                120 multi-zone candidates validated with EnergyPlus<br>
                Climate zones: Hot-Dry, Temperate, Cold<br>
                NSGA-II: 50 generations, population 40, ~20 Pareto points/run
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Research Datasets (Table III)"):
        st.dataframe(pd.DataFrame(DATASETS_TABLE, columns=["Dataset", "Type", "Size", "License", "Use"]), use_container_width=True, hide_index=True)
    with st.expander("Baselines (Table IV)"):
        for baseline in BASELINES:
            st.markdown(f"- {baseline}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate Plans", "500", "single-storey")
    c2.metric("Pareto Front", "20", "non-dominated")
    c3.metric("Score J(x;w)", "0.847", "weighted rank")
    c4.metric("Weather Source", "TMY3", st.session_state.climate)


# --- PAGE 2: Floor Plan Layout Generator ---
elif page == "🗺️ Floor Plan Layout Generator":
    st.info("Paper view: layout synthesis, adjacency supervision, daylight proxy, and civil / structural feasibility checks.")
    st.title("🗺️ Floor Plan Layout Generator")

    if st.button("Generate New Layout", type="primary"):
        with st.spinner("Simulating..."):
            increment_seed()
            time.sleep(0.6)
            rooms, _, _ = generate_random_layout(st.session_state.seed, st.session_state.plot_size)
            st.session_state.layout_rooms = rooms
            st.session_state.layout_metrics = generate_layout_metrics(st.session_state.seed)

    current_rooms, boundary_w, boundary_h = generate_random_layout(st.session_state.seed, st.session_state.plot_size)
    st.session_state.layout_rooms = current_rooms
    fig = floorplan_figure(current_rooms, boundary_w, boundary_h)
    st.plotly_chart(fig, use_container_width=True)

    metrics = st.session_state.layout_metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<span class='metric-badge'>Adjacency F1: {metrics['adjacency_f1']:.2f}</span>", unsafe_allow_html=True)
    with m2:
        progress_card("Daylight proxy", metrics["daylight"])
    with m3:
        m3.metric("Circulation", f"{metrics['circulation']:.1f} m/step", f"{5.2 - metrics['circulation']:.1f} vs B1")
    with m4:
        progress_card("Span-feasible rooms", metrics["span_feasible"], 100)

    with st.expander("Civil + Structure Constraint Status"):
        constraint_status = pd.DataFrame(
            [
                ["Setbacks", "✅ Pass", "Hard geometry"],
                ["Minimum areas", "✅ Pass", "Program-derived room areas"],
                ["Adjacency intent", "✅ Strong", f"F1 = {metrics['adjacency_f1']:.2f}"],
                ["Egress width", "✅ Pass", "1.2m preserved"],
                ["Daylight proxy", "✅ Pass" if metrics["daylight"] >= 0.68 else "⚠️ Marginal", f"{metrics['daylight']:.2f}"],
                ["Max span", "✅ Pass", f"{metrics['span_feasible']}% feasible"],
                ["Grid alignment", "✅ Pass", "Core-aligned partitions"],
                ["Opening limits", "✅ Pass", "Facade openings within code proxy"],
            ],
            columns=["Constraint", "Status", "Evidence"],
        )
        st.dataframe(constraint_status, use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Layout Seed", st.session_state.seed, "default 42")
    c2.metric("Required Edges", "6", "green graph links")
    c3.metric("Forbidden Pairs", "3", "red dashed links")
    c4.metric("Daylight Score", f"{metrics['daylight']:.2f}", "proposed target 0.71")


# --- PAGE 3: Interior Arrangement ---
elif page == "🪑 Interior Arrangement":
    st.info("Paper view: furniture priors from 3D-FRONT, simulated annealing packing, clearance envelopes, and accessibility paths.")
    st.title("🪑 Interior Arrangement (Simulated Annealing)")

    selected_room = st.selectbox("Select room", ["Living Room", "Master Bedroom"])
    st.plotly_chart(furniture_figure(selected_room), use_container_width=True)

    if st.button("Run SA Optimization", type="primary"):
        with st.spinner("Simulating..."):
            increment_seed()
            progress = st.progress(0, text="SA iterations")
            for idx in range(101):
                progress.progress(idx, text=f"Iteration {idx}/100")
                time.sleep(0.01)
            st.session_state.sa_completed = True

    energy_fig = go.Figure()
    energy_fig.add_trace(go.Scatter(x=SA_ITERS, y=SA_ENERGY, mode="lines", line=dict(color=ACCENT, width=3)))
    energy_fig.add_annotation(x=SA_ITERS[-1], y=SA_ENERGY[-1], text="Final score ≈ 0.76", showarrow=True)
    energy_fig.update_xaxes(title="Iteration")
    energy_fig.update_yaxes(title="Objective score")

    violation_fig = go.Figure()
    violation_fig.add_trace(go.Scatter(x=SA_ITERS, y=SA_VIOLATIONS, mode="lines+markers", line=dict(color=WARNING)))
    violation_fig.add_annotation(x=SA_ITERS[-1], y=SA_VIOLATIONS[-1], text="Final = 1", showarrow=True)
    violation_fig.update_xaxes(title="Iteration")
    violation_fig.update_yaxes(title="Violations")

    temp_curve = [100 * (0.995 ** i) for i in range(1000)]
    temp_fig = go.Figure()
    temp_fig.add_trace(go.Scatter(x=list(range(1000)), y=temp_curve, mode="lines", line=dict(color="#8ECAE6", width=3)))
    temp_fig.update_xaxes(title="Iteration")
    temp_fig.update_yaxes(title="Temperature")

    ch1, ch2, ch3 = st.columns(3)
    ch1.plotly_chart(themed_figure(energy_fig, "SA Energy Convergence", height=320), use_container_width=True)
    ch2.plotly_chart(themed_figure(violation_fig, "Clearance Violations Over Time", height=320), use_container_width=True)
    ch3.plotly_chart(themed_figure(temp_fig, "Cooling Schedule: T = 100·0.995^i", height=320), use_container_width=True)

    compare_fig = go.Figure()
    compare_fig.add_trace(go.Bar(x=SYSTEMS, y=[0.76, 0.62, 0.69, 0.67, 0.73], marker_color=[ACCENT, B1_COLOR, B2_COLOR, B3_COLOR, B4_COLOR]))
    compare_fig.update_yaxes(title="Interior usability")
    st.plotly_chart(themed_figure(compare_fig, "Interior Usability Comparison", height=340), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usability", "0.76", "+0.03 vs B4")
    c2.metric("Clearance Violations", "1", "from ~12")
    c3.metric("Room Mode", selected_room, "3D-FRONT priors")
    c4.metric("SA Steps", "1000", "T₀=100, α=0.995")


# --- PAGE 4: MEP Routing & Clash Detection ---
elif page == "🔧 MEP Routing & Clash Detection":
    st.info("Paper view: Eq. 4 routing cost, grid-based A*/JPS search, shaft reservations, and clash repair.")
    st.title("🔧 MEP Routing & Clash Detection")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        alpha = st.slider("alpha", 0.5, 2.0, 1.0, 0.1)
    with s2:
        beta = st.slider("beta", 0.1, 1.0, 0.5, 0.1)
    with s3:
        gamma = st.slider("gamma", 1.0, 10.0, 5.0, 0.5)
    with s4:
        delta = st.slider("delta", 0.5, 5.0, 2.0, 0.5)

    if st.button("Run Repair", type="primary"):
        with st.spinner("Simulating..."):
            increment_seed()
            time.sleep(1.0)
            st.session_state.mep_repaired = True

    repaired = st.session_state.mep_repaired
    grid, labels = mep_grid(repaired)
    color_map = {0: "#FFFFFF", 1: "#111111", 2: "#9E9E9E", 3: "#219EBC", 4: "#F77F00", 5: "#2A9D8F", 6: "#D62828"}
    st.plotly_chart(discrete_heatmap(grid, color_map, "20×20 MEP Routing Grid", labels, height=620), use_container_width=True)

    before = {"L": 145, "bends": 12, "clashes": 5, "clearance": 8}
    after = {"L": 131, "bends": 10, "clashes": 2, "clearance": 3}
    active = after if repaired else before
    cost_total = alpha * active["L"] + beta * active["bends"] + gamma * active["clashes"] + delta * active["clearance"]
    before_total = alpha * before["L"] + beta * before["bends"] + gamma * before["clashes"] + delta * before["clearance"]
    after_total = alpha * after["L"] + beta * after["bends"] + gamma * after["clashes"] + delta * after["clearance"]

    cost_df = pd.DataFrame(
        [
            ["Before repair", before["L"], before["bends"], before["clashes"], before["clearance"], round(before_total, 1)],
            ["After repair", after["L"], after["bends"], after["clashes"], after["clearance"], round(after_total, 1)],
        ],
        columns=["State", "L (m)", "N_bend", "N_clash", "P_clear", "Total Cost"],
    )
    st.dataframe(cost_df, use_container_width=True, hide_index=True)

    b1, b2, b3 = st.columns(3)
    length_fig = go.Figure()
    length_fig.add_trace(go.Bar(x=SYSTEMS, y=[131, 145, 138, 140, 152], marker_color=[ACCENT, B1_COLOR, B2_COLOR, B3_COLOR, B4_COLOR], error_y=dict(type="data", array=[19, 22, 20, 21, 24])))
    length_fig.update_yaxes(title="Length (m)")
    b1.plotly_chart(themed_figure(length_fig, "MEP Length Comparison", height=320), use_container_width=True)

    clash_fig = go.Figure()
    clash_fig.add_trace(go.Bar(x=SYSTEMS, y=[1.8, 7.1, 4.6, 5.0, 3.9], marker_color=[ACCENT, B1_COLOR, B2_COLOR, B3_COLOR, B4_COLOR], error_y=dict(type="data", array=[1.2, 3.5, 2.6, 2.8, 2.2])))
    clash_fig.update_yaxes(title="Clashes")
    b2.plotly_chart(themed_figure(clash_fig, "Clashes After Repair", height=320), use_container_width=True)

    pressure_fig = go.Figure()
    pressure_fig.add_trace(go.Bar(x=SYSTEMS, y=[0.33, 0.52, 0.44, 0.47, 0.41], marker_color=[ACCENT, B1_COLOR, B2_COLOR, B3_COLOR, B4_COLOR]))
    pressure_fig.update_yaxes(title="Pressure-drop proxy")
    b3.plotly_chart(themed_figure(pressure_fig, "Pressure-drop Proxy", height=320), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Cost", f"{cost_total:.1f}", "Eq. 4")
    c2.metric("Mode", "Repaired" if repaired else "Pre-repair", "grid routing")
    c3.metric("MEP Length", "131 m" if repaired else "145 m", "proposed vs baseline")
    c4.metric("Residual Clashes", "2" if repaired else "5", "cell-level")


# --- PAGE 5: IoT Placement & Coverage Heatmap ---
elif page == "📡 IoT Placement & Coverage Heatmap":
    st.info("Paper view: Eq. 5 IoT placement objective, Gaussian signal fields, privacy-aware placement, and gateway hop limits.")
    st.title("📡 IoT Placement & Coverage Heatmap")

    controls = st.columns([1, 1, 1.2])
    with controls[0]:
        device_count = st.slider("Device count", 3, 12, 7)
    with controls[1]:
        privacy_penalty = st.toggle("Privacy penalty", value=False)
    with controls[2]:
        if st.button("Optimize Placement", type="primary"):
            with st.spinner("Simulating..."):
                increment_seed()
                progress = st.progress(0, text="Greedy random restart")
                for i in range(50):
                    progress.progress((i + 1) * 2, text=f"Restart {i + 1}/50")
                    time.sleep(0.01)
                optimize_devices(device_count, privacy_penalty)

    devices = active_devices(device_count)
    signal, raw_coverage = coverage_heatmap(devices)
    privacy_hits = [device for device in devices if device_in_privacy(device)]
    coverage_pct = int(round(min(100, raw_coverage * 110)))
    risk_score = round(0.25 + (0.3 * len(privacy_hits) if privacy_penalty else 0.0), 2)
    hop_latency = round(1.0 + 0.4 * (device_count >= 7) + 0.1 * len(privacy_hits), 1)

    if privacy_hits:
        st.warning("Privacy-sensitive placement detected: " + ", ".join(device["name"] for device in privacy_hits))

    st.plotly_chart(iot_figure(signal, devices), use_container_width=True)

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(indicator_figure("Coverage", coverage_pct, 0, 100, suffix="%"), use_container_width=True)
    g2.plotly_chart(indicator_figure("Hop Latency", hop_latency, 0, 3, suffix=" hops", color=WARNING), use_container_width=True)
    g3.plotly_chart(indicator_figure("Risk Score", risk_score, 0, 1, color=DANGER), use_container_width=True)

    b1, b2 = st.columns(2)
    coverage_fig = go.Figure()
    coverage_fig.add_trace(go.Bar(x=SYSTEMS, y=[93, 81, 86, 85, 89], marker_color=[ACCENT, B1_COLOR, B2_COLOR, B3_COLOR, B4_COLOR]))
    coverage_fig.update_yaxes(title="Coverage (%)")
    b1.plotly_chart(themed_figure(coverage_fig, "Coverage Comparison", height=320), use_container_width=True)

    risk_fig = go.Figure()
    risk_fig.add_trace(go.Bar(x=SYSTEMS, y=[0.25, 0.42, 0.36, 0.38, 0.31], marker_color=[ACCENT, B1_COLOR, B2_COLOR, B3_COLOR, B4_COLOR]))
    risk_fig.update_yaxes(title="Risk score")
    b2.plotly_chart(themed_figure(risk_fig, "Risk Comparison", height=320), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coverage ≥ τ", f"{coverage_pct}%", "τ = 0.6")
    c2.metric("Mean Hops", f"{hop_latency}", "target 1.4")
    c3.metric("Risk", f"{risk_score:.2f}", "η = 0.3")
    c4.metric("Gateways", "1", "10m range")


# --- PAGE 6: Energy & Comfort Screening ---
elif page == "⚡ Energy & Comfort Screening":
    st.info("Paper view: Eq. 6 comfort proxy, climate-linked screening, Pareto trade-offs, and NSGA-II convergence.")
    st.title("⚡ Energy & Comfort Screening")

    climate_choice = st.selectbox("Climate zone", ["Hot-Dry", "Temperate", "Cold"], index=["Hot-Dry", "Temperate", "Cold"].index(st.session_state.climate))
    st.session_state.climate = climate_choice

    temp_fig, violation_count, comfort_score = energy_profile_figure(climate_choice)
    st.plotly_chart(themed_figure(temp_fig, temp_fig.layout.title.text, height=420), use_container_width=True)

    eui_col, pareto_col = st.columns([0.75, 1.25])
    with eui_col:
        eui_fig = indicator_figure(
            "EUI Proxy",
            79,
            0,
            140,
            suffix=" kWh/m²-yr",
            steps=[
                {"range": [0, 60], "color": "rgba(46,196,182,0.25)"},
                {"range": [60, 90], "color": "rgba(233,196,106,0.25)"},
                {"range": [90, 140], "color": "rgba(231,111,81,0.25)"},
            ],
            color=WARNING,
        )
        st.plotly_chart(eui_fig, use_container_width=True)
        baseline_eui = {"Hot-Dry": 95, "Temperate": 79, "Cold": 110}[climate_choice]
        st.markdown(
            f"""
            <div class="paper-card">
                <b>Synthetic climate baseline</b><br>
                {climate_choice}: {baseline_eui} kWh/m²-yr<br>
                Proposed paper proxy remains 79 ± 13 kWh/m²-yr
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pareto_col:
        pareto_df = pd.DataFrame(PARETO_POINTS)
        base_points = pareto_df[pareto_df["id"] != 8]
        selected_point = pareto_df[pareto_df["id"] == 8]

        pareto_fig = go.Figure()
        pareto_fig.add_trace(
            go.Scatter(
                x=base_points["EUI"],
                y=base_points["cost_lakhs"],
                mode="markers",
                marker=dict(size=base_points["comfort"] * 45, color=base_points["clashes"], colorscale="Viridis", showscale=True, colorbar=dict(title="Clashes")),
                text=[f"Plan {idx}" for idx in base_points["id"]],
                hovertemplate="Plan %{text}<br>EUI=%{x}<br>Cost=%{y}L<extra></extra>",
                name="Pareto points",
            )
        )
        pareto_fig.add_trace(
            go.Scatter(
                x=selected_point["EUI"],
                y=selected_point["cost_lakhs"],
                mode="markers+text",
                marker=dict(size=22, color=ACCENT, symbol="star"),
                text=["Proposed"],
                textposition="top center",
                name="Selected plan",
            )
        )
        pareto_fig.update_xaxes(title="EUI (kWh/m²-yr)")
        pareto_fig.update_yaxes(title="Cost (₹ lakhs)")
        st.plotly_chart(themed_figure(pareto_fig, "NSGA-II Pareto Front", height=460), use_container_width=True)

    hv_fig = go.Figure()
    hv_fig.add_trace(go.Scatter(x=GENERATIONS, y=HYPERVOLUME, mode="lines+markers", line=dict(color=ACCENT, width=3)))
    hv_fig.add_vline(x=35, line_dash="dash", line_color=WARNING)
    hv_fig.add_annotation(x=35, y=0.86, text="Plateau ~ Gen 35", showarrow=True, arrowcolor=WARNING)
    hv_fig.update_xaxes(title="Generation")
    hv_fig.update_yaxes(title="Hypervolume")
    st.plotly_chart(themed_figure(hv_fig, "NSGA-II Convergence", height=360), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("EUI", "79 kWh/m²-yr", "paper-reported proposed")
    c2.metric("Comfort Violations", violation_count, f"f_comfort = {comfort_score:.2f}")
    c3.metric("Climate Zone", climate_choice, "TMY-style synthetic profile")


# --- PAGE 7: Results Dashboard ---
elif page == "📊 Results Dashboard":
    st.info("Paper view: full Table V benchmarking across layout, structure, MEP, IoT, BIM, cost, and energy.")
    st.title("📊 Results Dashboard")

    tabs = st.tabs(["All", "Layout", "Structure", "MEP", "IoT", "BIM", "Cost-Energy"])
    domain_map = {
        "All": BENCHMARK_METRICS,
        "Layout": [m for m in BENCHMARK_METRICS if m["domain"] == "Layout"],
        "Structure": [m for m in BENCHMARK_METRICS if m["domain"] == "Structure"],
        "MEP": [m for m in BENCHMARK_METRICS if m["domain"] == "MEP"],
        "IoT": [m for m in BENCHMARK_METRICS if m["domain"] == "IoT"],
        "BIM": [m for m in BENCHMARK_METRICS if m["domain"] == "BIM"],
        "Cost-Energy": [m for m in BENCHMARK_METRICS if m["domain"] == "Cost-Energy"],
    }

    for tab, label in zip(tabs, domain_map.keys()):
        with tab:
            subset = domain_map[label]
            st.plotly_chart(grouped_bar_chart(subset, f"Grouped Benchmark Chart • {label}"), use_container_width=True)
            st.plotly_chart(ranking_heatmap(subset), use_container_width=True)
            delta_badges(subset)

    st.plotly_chart(radar_chart(), use_container_width=True)

    win_count = winners_count()
    st.markdown(
        f"""
        <div class="paper-card" style="text-align:center;">
            <div style="font-size:1.1rem;color:#BFD7EA;">Proposed system wins</div>
            <div style="font-size:2.6rem;font-weight:800;color:#8ff4e8;">{win_count}/16 metrics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Adjacency F1", "0.85", "+2.4% vs B4")
    c2.metric("MEP Length", "131 m", "-13.8% vs B4")
    c3.metric("IoT Coverage", "93%", "+4.5% vs B4")
    c4.metric("EUI Proxy", "79", "-2.5% vs B4")


# --- PAGE 8: BIM/IFC Export & BOQ ---
elif page == "🏗️ BIM/IFC Export & BOQ":
    st.info("Paper view: IFC completeness, semantic export coverage, BOQ roll-up, and audit traceability.")
    st.title("🏗️ BIM/IFC Export & BOQ")

    ifc_df = pd.DataFrame(IFC_ELEMENTS)
    ifc_df["Status"] = ifc_df["Completeness %"].apply(lambda x: "⚠️" if x < 95 else "✅")

    donut = go.Figure(
        go.Pie(
            labels=ifc_df["Element Type"],
            values=ifc_df["Count"],
            hole=0.6,
            marker=dict(colors=px.colors.qualitative.Set3),
            text=[f"{v}%" for v in ifc_df["Completeness %"]],
            textinfo="label+text",
        )
    )
    donut.add_annotation(text="96%<br>Overall", x=0.5, y=0.5, showarrow=False, font=dict(size=20))
    left, right = st.columns([0.95, 1.05])
    left.plotly_chart(themed_figure(donut, "IFC Completeness by Element", height=520), use_container_width=True)
    right.dataframe(ifc_df, use_container_width=True, hide_index=True)

    display_usd = st.toggle("Show BOQ in USD", value=st.session_state.boq_currency == "USD")
    st.session_state.boq_currency = "USD" if display_usd else "INR"
    boq_df = boq_dataframe(st.session_state.boq_currency)
    symbol = currency_symbol(st.session_state.boq_currency)
    formatted_df = boq_df.copy()
    formatted_df["Rate"] = formatted_df["Rate"].map(lambda x: f"{symbol}{x:,.2f}" if st.session_state.boq_currency == "USD" else f"{symbol}{x:,.0f}")
    formatted_df["Total"] = formatted_df["Total"].map(lambda x: f"{symbol}{x:,.2f}" if st.session_state.boq_currency == "USD" else f"{symbol}{x:,.0f}")
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    boq_df.to_csv(csv_buffer, index=False)
    st.download_button("Download BOQ as CSV", csv_buffer.getvalue(), file_name="boq_export.csv", mime="text/csv")

    p1, p2 = st.columns(2)
    cost_pie = go.Figure(go.Pie(labels=["Structure", "Interior", "MEP", "IoT", "Overhead"], values=[38, 22, 25, 5, 10], marker=dict(colors=[PRIMARY, "#3D5A80", "#2A9D8F", "#8ECAE6", WARNING])))
    p1.plotly_chart(themed_figure(cost_pie, "Cost Breakdown", height=360), use_container_width=True)

    cost_error_fig = go.Figure()
    cost_error_fig.add_trace(go.Bar(x=SYSTEMS, y=[6.5, 12.4, 9.8, 10.5, 7.2], marker_color=[ACCENT, B1_COLOR, B2_COLOR, B3_COLOR, B4_COLOR], error_y=dict(type="data", array=[2.7, 5.1, 4.2, 4.5, 3.1])))
    cost_error_fig.update_yaxes(title="Cost error (%)")
    p2.plotly_chart(themed_figure(cost_error_fig, "BOQ Cost Error Comparison", height=360), use_container_width=True)

    with st.expander("Audit Log"):
        for line in AUDIT_LOG:
            st.markdown(f"- {line}")

    total_cost_inr = 4800000
    total_cost = total_cost_inr if st.session_state.boq_currency == "INR" else round(total_cost_inr / 83.5, 2)
    total_cost_label = f"{symbol}{total_cost:,.0f}" if st.session_state.boq_currency == "INR" else f"{symbol}{total_cost:,.2f}"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall IFC Completeness", "96%", "weighted")
    c2.metric("Element Classes", "8", "openBIM export")
    c3.metric("Estimated Cost", total_cost_label, "typical 120m² plan")
    c4.metric("Cost Error", "6.5%", "-0.7 pts vs B4")
