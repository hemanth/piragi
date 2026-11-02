"""Standard RAG Benchmark Evaluation Framework.

Provides standardized IR & RAG accuracy metrics (nDCG@k, Recall@k, MRR@k, Precision@k),
built-in Cranfield and BEIR (SciFact) evaluation fixtures, and a paired benchmark runner
evaluating Condition A (Baseline: Dense Vector Only) against Condition B (Enhanced: SOTA RAG).

CLI Usage:
    uv run python evals/eval_benchmark.py --dataset all --condition both --output evals/results/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

# Suppress tokenizers parallelism warning during multi-threaded retrieval
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Standardized Evaluation Metrics
# ===========================================================================

def compute_ndcg_at_k(
    relevance_scores_or_ranks: Sequence[Any],
    relevant_ids_or_k: Union[int, Sequence[Any], Set[Any], Dict[Any, Any], None] = None,
    k: int = 10,
) -> float:
    """Compute normalized Discounted Cumulative Gain at rank k (nDCG@k).

    Supports:
        1. compute_ndcg_at_k(relevance_scores: list, k: int = 10)
        2. compute_ndcg_at_k(retrieved_ids: list, qrels: dict|set, k: int = 10)

    Formula:
        DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
        IDCG@k = sum_{i=1}^k (2^{ideal_rel_i} - 1) / log2(i + 1)
        nDCG@k = DCG@k / IDCG@k if IDCG > 0 else 0.0
    """
    if relevant_ids_or_k is None:
        cutoff = int(k)
        ground_truth = None
    elif isinstance(relevant_ids_or_k, int):
        cutoff = int(relevant_ids_or_k)
        ground_truth = None
    else:
        cutoff = int(k)
        ground_truth = relevant_ids_or_k

    if cutoff <= 0 or not relevance_scores_or_ranks:
        return 0.0

    if ground_truth is not None:
        retrieved_ids = relevance_scores_or_ranks
        seen: Set[Any] = set()
        scores: List[float] = []

        if isinstance(ground_truth, dict):
            for doc in retrieved_ids[:cutoff]:
                if doc in ground_truth and doc not in seen:
                    scores.append(float(ground_truth[doc]))
                    seen.add(doc)
                else:
                    scores.append(0.0)
            ideal_pool = sorted([float(v) for v in ground_truth.values()], reverse=True)
        else:
            gt_set = set(ground_truth)
            for doc in retrieved_ids[:cutoff]:
                if doc in gt_set and doc not in seen:
                    scores.append(1.0)
                    seen.add(doc)
                else:
                    scores.append(0.0)
            ideal_pool = [1.0] * len(gt_set)
    else:
        # Signature 1: (relevance_scores, k)
        first = relevance_scores_or_ranks[0]
        if isinstance(first, (int, float, np.number)):
            scores = [float(x) for x in relevance_scores_or_ranks[:cutoff]]
            ideal_pool = sorted(scores, reverse=True)
        else:
            return 0.0

    if not scores or sum(scores) == 0:
        return 0.0

    dcg = sum((2.0**rel - 1.0) / math.log2(idx + 2) for idx, rel in enumerate(scores))
    ideal = ideal_pool[:cutoff]
    idcg = sum((2.0**rel - 1.0) / math.log2(idx + 2) for idx, rel in enumerate(ideal))

    return min(1.0, float(dcg / idcg)) if idcg > 0.0 else 0.0


def compute_recall_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Set[Any], Sequence[Any], Dict[Any, Any]],
    k: int = 10,
) -> float:
    """Compute Recall@k.

    Proportion of relevant documents found in the top-k retrieved documents.
    """
    if not retrieved_ids or k <= 0:
        return 0.0

    if isinstance(relevant_ids, dict):
        ground_truth = set(relevant_ids.keys())
    else:
        ground_truth = set(relevant_ids)

    if not ground_truth:
        return 0.0

    top_k_retrieved = list(dict.fromkeys(retrieved_ids[:k]))
    hits = len(set(top_k_retrieved) & ground_truth)
    return float(hits / len(ground_truth))


def compute_precision_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Set[Any], Sequence[Any], Dict[Any, Any]],
    k: int = 10,
) -> float:
    """Compute Precision@k.

    Proportion of retrieved documents in the top-k that are relevant.
    """
    if not retrieved_ids or k <= 0:
        return 0.0

    if isinstance(relevant_ids, dict):
        ground_truth = set(relevant_ids.keys())
    else:
        ground_truth = set(relevant_ids)

    if not ground_truth:
        return 0.0

    top_k_retrieved = list(dict.fromkeys(retrieved_ids[:k]))
    hits = len(set(top_k_retrieved) & ground_truth)
    return float(hits / k)


def compute_mrr_at_k(
    retrieved_ids_or_ranks: Union[Sequence[Any], List[int]],
    relevant_ids: Union[Set[Any], Sequence[Any], Dict[Any, Any], None] = None,
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank (MRR) or single-query reciprocal rank at rank k.

    Supports:
        1. compute_mrr_at_k(retrieved_ids, relevant_ids, k=10) -> Reciprocal rank of 1st hit
        2. compute_mrr_at_k(ranks_list, k=10) -> Mean of reciprocal ranks <= k
    """
    if not retrieved_ids_or_ranks:
        return 0.0

    # Case 1: retrieved_ids + relevant_ids
    if relevant_ids is not None:
        if k <= 0:
            return 0.0
        if isinstance(relevant_ids, dict):
            ground_truth = set(relevant_ids.keys())
        else:
            ground_truth = set(relevant_ids)

        if not ground_truth:
            return 0.0

        for rank, doc_id in enumerate(retrieved_ids_or_ranks[:k], start=1):
            if doc_id in ground_truth:
                return float(1.0 / rank)
        return 0.0

    # Case 2: List of ranks
    ranks = retrieved_ids_or_ranks
    valid_reciprocals = [
        1.0 / r for r in ranks if isinstance(r, (int, float)) and 0 < r <= k
    ]
    if not ranks:
        return 0.0
    # Include 0.0 for items that had no hit within k
    all_reciprocals = [
        (1.0 / r if 0 < r <= k else 0.0) for r in ranks if isinstance(r, (int, float))
    ]
    return float(np.mean(all_reciprocals)) if all_reciprocals else 0.0


def compute_mrr(relevance_ranks: Sequence[int]) -> float:
    """Reference Mean Reciprocal Rank (MRR) across queries."""
    if not relevance_ranks:
        return 0.0
    return float(np.mean([1.0 / r if r > 0 else 0.0 for r in relevance_ranks]))


# ===========================================================================
# 2. Benchmark Datasets (Cranfield & BEIR)
# ===========================================================================

@dataclass
class BenchmarkDocument:
    id: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkQuery:
    id: str
    query: str
    stepback_query: str
    ground_truth: Dict[str, int]  # doc_id -> relevance grade (1-3)


@dataclass
class BenchmarkDataset:
    name: str
    description: str
    documents: List[BenchmarkDocument]
    queries: List[BenchmarkQuery]

    @property
    def qrels(self) -> Dict[str, Dict[str, int]]:
        return {q.id: q.ground_truth for q in self.queries}


def load_cranfield_dataset() -> BenchmarkDataset:
    """Load standard Cranfield aeronautical engineering evaluation benchmark."""
    docs = [
        BenchmarkDocument(
            id="cranfield_0001",
            title="Boundary Layer Transition with Surface Cooling",
            text=(
                "Experimental investigation of the boundary layer transition on a flat plate with surface "
                "cooling at supersonic speeds. Wind tunnel measurements were conducted at Mach numbers from 2.0 to 4.5. "
                "Results indicate that surface cooling delays laminar boundary layer transition significantly up to Mach 3.5, "
                "stabilizing Tollmien-Schlichting waves. Above Mach 4.0, the stabilizing effect diminishes due to higher acoustic mode instabilities."
            ),
            metadata={"domain": "aerodynamics", "mach_regime": "supersonic"},
        ),
        BenchmarkDocument(
            id="cranfield_0002",
            title="Theoretical Hypersonic Aerodynamic Heating",
            text=(
                "Theoretical analysis of aerodynamic heating in hypersonic reentry. A blunt-nosed body of revolution undergoes intensive "
                "stagnation point convective and radiative heat transfer during atmospheric entry. Boundary layer shock interactions and dissociated "
                "air chemistry govern the thermal flux delivered to the thermal protection shield, requiring carbon-phenolic ablative materials."
            ),
            metadata={"domain": "aerothermodynamics", "mach_regime": "hypersonic"},
        ),
        BenchmarkDocument(
            id="cranfield_0003",
            title="Transonic Shock Wave Boundary Layer Interaction",
            text=(
                "Shock wave boundary layer interaction in transonic flow over compressor airfoils. Oblique shock reflection on turbulent boundary "
                "layers induces localized severe adverse pressure gradients, boundary layer thickening, and localized flow separation bubbles "
                "near the compressor blade suction surface, causing aerodynamic performance degradation."
            ),
            metadata={"domain": "aerodynamics", "mach_regime": "transonic"},
        ),
        BenchmarkDocument(
            id="cranfield_0004",
            title="Skin Friction Reduction via Distributed Suction",
            text=(
                "Skin friction drag reduction on swept wings using distributed boundary layer suction through micro-porous titanium surfaces. "
                "Laminar flow control achieved through continuous suction stabilizes crossflow vortices and Tollmien-Schlichting instabilities, "
                "reducing total aircraft profile drag by up to 28% under subsonic cruise flight conditions."
            ),
            metadata={"domain": "aerodynamics", "mach_regime": "subsonic"},
        ),
        BenchmarkDocument(
            id="cranfield_0005",
            title="Transonic Aeroelastic Flutter of Sweptback Wings",
            text=(
                "Aeroelastic flutter analysis of thin sweptback wings at transonic speeds. Unsteady aerodynamic pressure fluctuations and structural "
                "flexure-torsion modal coupling induce dynamic aeroelastic instability. The pronounced transonic dip in flutter boundary occurs near "
                "Mach 0.95 due to oscillating shock wave motions on the upper wing surface."
            ),
            metadata={"domain": "aeroelasticity", "mach_regime": "transonic"},
        ),
        BenchmarkDocument(
            id="cranfield_0006",
            title="Transient Thermal Stresses in Supersonic Wing Structures",
            text=(
                "Transient thermal stresses and leading edge structural deformation in supersonic aircraft wings. Severe aerodynamic heating creates steep "
                "temperature gradients across internal spars, ribs, and external skin panels, generating high thermal stresses that compromise aeroelastic "
                "stiffness and structural safety during sustained supersonic acceleration."
            ),
            metadata={"domain": "structures", "mach_regime": "supersonic"},
        ),
        BenchmarkDocument(
            id="cranfield_0007",
            title="Turbulent Boundary Layer Velocity Distributions",
            text=(
                "Velocity distributions and skin friction coefficients in turbulent boundary layers with adverse pressure gradients. "
                "Momentum integral methods, Clauser equilibrium parameters, and turbulent eddy viscosity formulations are correlated against "
                "experimental velocity profiles in divergent conical diffusers."
            ),
            metadata={"domain": "aerodynamics", "mach_regime": "subsonic"},
        ),
        BenchmarkDocument(
            id="cranfield_0008",
            title="Rarefied Upper Atmospheric Aerodynamics",
            text=(
                "Effects of slip flow and free molecule aerodynamics in the upper atmosphere. Rarefied gas dynamics at Knudsen numbers exceeding 0.1 "
                "alter conventional continuum drag and lift predictions for low Earth orbit satellites and hypersonic glide vehicles during high-altitude maneuvers."
            ),
            metadata={"domain": "hypersonics", "mach_regime": "orbital"},
        ),
        BenchmarkDocument(
            id="cranfield_0009",
            title="Slender Delta Wing Vortex Breakdown",
            text=(
                "Investigation of leading-edge vortex breakdown over slender delta wings at high angles of attack. Core axial velocity stagnation "
                "and helical vortex breakdown trigger sudden non-linear lift loss and abrupt pitch-up moments in combat aircraft configurations."
            ),
            metadata={"domain": "aerodynamics", "mach_regime": "subsonic"},
        ),
        BenchmarkDocument(
            id="cranfield_0010",
            title="Dynamic Stall on Oscillating Airfoils",
            text=(
                "Unsteady lift and moment characteristics of oscillating airfoils undergoing large-amplitude pitch and plunge cycles. "
                "Dynamic stall vortex formation, leading-edge shedding, and severe pitching moment damping loss produce hysteresis loops."
            ),
            metadata={"domain": "aerodynamics", "mach_regime": "unsteady"},
        ),
        BenchmarkDocument(
            id="cranfield_0011",
            title="Micro-Slot Active Flow Control and Separation Delay",
            text=(
                "Boundary layer suction and blowing through spanwise micro-slots for separation control on high-lift trailing edge flaps. "
                "Active pneumatic actuation energizes the near-wall boundary layer, delaying separation and increasing maximum lift coefficients by 18%."
            ),
            metadata={"domain": "aerodynamics", "mach_regime": "subsonic"},
        ),
        BenchmarkDocument(
            id="cranfield_0012",
            title="Shock Layer Radiative Heat Transfer in Planetary Entry",
            text=(
                "Radiative heat transfer calculations for blunt heat shields entering planetary atmospheres. High-temperature equilibrium shock layer "
                "radiation emitted by molecular band systems contributes substantially to the total convective and radiative stagnation heat flux."
            ),
            metadata={"domain": "aerothermodynamics", "mach_regime": "hypersonic"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0001",
            title="Boundary Layer Disturbance Modes in Supersonic Flat Plate Flow",
            text=(
                "Investigation of acoustic disturbance modes and boundary layer stability in supersonic flat plate flow. "
                "Flow angularity and Mach number uniformity across the core flow are measured with multi-hole probe rakes at Mach 2.5 to 4.5."
            ),
            metadata={"domain": "wind_tunnel", "mach_regime": "supersonic"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0002",
            title="Pitot-Static Tube Boundary Layer Probes",
            text=(
                "Design and aerodynamic calibration of flattened pitot-static boundary layer tubes for high-speed compressible flows. "
                "Viscous displacement effects and wall interference corrections are quantified experimentally in turbulent flows."
            ),
            metadata={"domain": "instrumentation", "mach_regime": "compressible"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0003",
            title="Industrial Heat Exchanger Regenerative Cooling",
            text=(
                "Performance characteristics of regenerative cooling channels in high-temperature heat exchangers. "
                "Coolant mass flow rate and convective Nusselt numbers are analyzed for cryogenic fluids in stationary ducts."
            ),
            metadata={"domain": "thermodynamics", "mach_regime": "subsonic"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0004",
            title="Subsonic Glider Airfoil Boundary Layer Infiltration",
            text=(
                "In-flight boundary layer transition measurements on high-performance sailplane laminar airfoils without artificial cooling. "
                "Acoustic microphone arrays detect natural Tollmien-Schlichting frequency spectrums at chord Reynolds numbers below one million."
            ),
            metadata={"domain": "sailplanes", "mach_regime": "subsonic"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0005",
            title="Stagnation Point Heat Flux on Hypersonic Entry Vehicles",
            text=(
                "Thermal flux calculations and pressure distributions over slender sharp cones at hypersonic speeds between Mach 6 and 10. "
                "Taylor-Maccoll numerical solutions for attached conical shock waves are compared against ballistic range shadowgraph data."
            ),
            metadata={"domain": "hypersonics", "mach_regime": "hypersonic"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0006",
            title="Orbital Satellite Radiator Thermal Coating Degradation",
            text=(
                "Thermal control coatings for low Earth orbit satellite radiators exposed to solar ultraviolet radiation and atomic oxygen. "
                "Emittance and solar absorptance degradation over prolonged space exposure are evaluated using quartz crystal microbalances."
            ),
            metadata={"domain": "thermal_control", "mach_regime": "space"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0007",
            title="Shock Boundary Layer Flow Detachment on Compressor Airfoils",
            text=(
                "Inlet aerodynamic characteristics and stagnation pressure recovery in mixed-compression axisymmetric supersonic inlets. "
                "Bleed slot geometries prevent boundary layer separation on internal compression ramps during off-design operation."
            ),
            metadata={"domain": "propulsion", "mach_regime": "supersonic"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0008",
            title="Steady Lift Distributions on Subsonic Swept Wings",
            text=(
                "Vortex lattice methods for predicting spanwise aerodynamic lift distributions on sweptback transport wings in incompressible flow. "
                "Root stall delay and dihedral aerodynamic effects are evaluated at zero yaw angle."
            ),
            metadata={"domain": "subsonic_aero", "mach_regime": "subsonic"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0009",
            title="Helicopter Rotor Blade Flap-Lag Aeroelastic Damping",
            text=(
                "Aeroelastic stability of hingeless helicopter rotor blades in hover and forward flight. "
                "Geometric non-linearities and unsteady wake aerodynamics contribute to lead-lag mechanical damping and air resonance prevention."
            ),
            metadata={"domain": "rotary", "mach_regime": "subsonic"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0010",
            title="Transonic Flutter Boundary and Buffet on Dynamic Wings",
            text=(
                "Self-sustained shock wave oscillations and dynamic buffet onset over rigid supercritical airfoils at Mach 0.76 to 0.95. "
                "High-speed Schlieren visualization captures trailing-edge boundary layer separation without structural wing flexure."
            ),
            metadata={"domain": "transonic", "mach_regime": "transonic"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0011",
            title="Laminar Flow Drag Reduction on Polished Titanium Surfaces",
            text=(
                "Skin friction drag measurements over polished titanium plates with distributed sand grain surface roughness in zero pressure gradients. "
                "Hydraulic smoothness limits and boundary layer momentum loss thickness are determined in low-turbulence wind tunnels without suction."
            ),
            metadata={"domain": "aerodynamics", "mach_regime": "subsonic"},
        ),
        BenchmarkDocument(
            id="cranfield_dist_0012",
            title="Mechanical Vortex Generators on Aircraft Wings",
            text=(
                "Micro-vortex generator arrays positioned upstream of shock boundary layer interactions on commercial transport wings. "
                "Streamwise vortex shedding energizes the boundary layer to prevent flow detachment without external air injection."
            ),
            metadata={"domain": "flow_control", "mach_regime": "transonic"},
        ),
    ]

    queries = [
        BenchmarkQuery(
            id="cran_q01",
            query="What acoustic disturbance mechanisms govern boundary layer stability in supersonic flat plate flow?",
            stepback_query="What physical aerodynamic mechanisms govern laminar to turbulent boundary layer transition and stability?",
            ground_truth={"cranfield_0001": 3, "cranfield_0007": 1},
        ),
        BenchmarkQuery(
            id="cran_q02",
            query="What analytical methods calculate convective and radiative heating during atmospheric entry?",
            stepback_query="What are the primary theoretical modes of thermal flux and heat transfer during atmospheric entry?",
            ground_truth={"cranfield_0002": 3, "cranfield_0012": 2},
        ),
        BenchmarkQuery(
            id="cran_q03",
            query="What aerodynamic mechanisms induce the transonic dip in wing flutter boundaries?",
            stepback_query="What aerodynamic and structural principles govern aeroelastic stability in transonic flight?",
            ground_truth={"cranfield_0005": 3, "cranfield_0006": 1},
        ),
        BenchmarkQuery(
            id="cran_q04",
            query="How does boundary layer manipulation with suction surfaces achieve laminar flow control?",
            stepback_query="What boundary layer flow control techniques are used to manipulate skin friction and drag?",
            ground_truth={"cranfield_0004": 3, "cranfield_0011": 2},
        ),
        BenchmarkQuery(
            id="cran_q05",
            query="How do adverse pressure gradients and shock reflections generate separation bubbles in compressor stages?",
            stepback_query="How do severe adverse pressure gradients and shock reflections influence boundary layer separation?",
            ground_truth={"cranfield_0003": 3, "cranfield_0007": 1},
        ),
    ]

    return BenchmarkDataset(
        name="cranfield",
        description="Standard Cranfield Aeronautical Engineering Evaluation Benchmark",
        documents=docs,
        queries=queries,
    )


def load_beir_dataset() -> BenchmarkDataset:
    """Load standard BEIR (SciFact biomedical scientific claim verification) benchmark."""
    docs = [
        BenchmarkDocument(
            id="beir_scifact_0001",
            title="MGMT Promoter Methylation in Glioblastoma Chemotherapy",
            text=(
                "0-6-methylguanine-DNA methyltransferase (MGMT) promoter methylation is a paramount predictive molecular biomarker in glioblastoma. "
                "Patients harboring hypermethylated MGMT promoters demonstrate significantly prolonged overall survival and progression-free survival "
                "when treated with alkylating chemotherapeutic agent temozolomide compared to individuals with unmethylated MGMT promoters."
            ),
            metadata={"domain": "oncology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0002",
            title="PCSK9 Inhibition and LDL Receptor Clearance",
            text=(
                "Proprotein convertase subtilisin/kexin type 9 (PCSK9) binds and directs low-density lipoprotein receptors (LDLR) toward lysosomal degradation "
                "in hepatocytes. Therapeutic monoclonal antibodies and CRISPR-Cas9 targeted knockout of PCSK9 prevent receptor degradation, accelerating "
                "hepatic LDL cholesterol clearance from circulation and decreasing major adverse cardiovascular events."
            ),
            metadata={"domain": "cardiology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0003",
            title="Oncogenic MicroRNA-21 Targeting of PTEN",
            text=(
                "MicroRNA-21 (miR-21) is strongly overexpressed across malignant solid tumors including glioblastoma, colorectal cancer, and hepatocellular carcinoma. "
                "miR-21 operates as an oncomir by post-transcriptionally silencing PTEN tumor suppressor expression, which hyperactivates downstream PI3K/Akt "
                "survival signaling and drives apoptosis resistance."
            ),
            metadata={"domain": "oncology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0004",
            title="Interleukin-6 Receptor Blockade in COVID-19 Cytokine Storm",
            text=(
                "Interleukin-6 (IL-6) hyperactivation is the central mediator of systemic cytokine release syndrome and alveolar capillary leakage in severe COVID-19. "
                "Targeted inhibition of the IL-6 receptor utilizing humanized monoclonal antibody tocilizumab significantly reduces systemic inflammatory markers, "
                "hastens clinical recovery, and diminishes mechanical ventilation requirements."
            ),
            metadata={"domain": "immunology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0005",
            title="Metformin Inhibition of Hepatic Gluconeogenesis via AMPK",
            text=(
                "Metformin suppresses excessive hepatic gluconeogenesis through activation of AMP-activated protein kinase (AMPK) and allosteric inhibition "
                "of mitochondrial respiratory chain complex I. Phosphorylation of acetyl-CoA carboxylase (ACC) and transcriptional repression of PEPCK "
                "and G6Pase enzymes effectively restore fasting glycemic homeostasis in type 2 diabetes."
            ),
            metadata={"domain": "endocrinology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0006",
            title="TERT Promoter Mutations and Cellular Immortalization",
            text=(
                "Telomerase reverse transcriptase (TERT) promoter mutations (predominantly C228T and C250T) generate novel de novo consensus binding motifs "
                "for ETS/GABP transcription factors. These activating somatic mutations upregulate TERT expression, overcoming critical telomeric attrition "
                "and conferring replicative immortality in human melanoma and urothelial bladder carcinomas."
            ),
            metadata={"domain": "genetics", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0007",
            title="Dietary Saturated Fatty Acids and Hepatic LDL Regulation",
            text=(
                "Excessive dietary intake of saturated fatty acids increases circulating low-density lipoprotein (LDL) cholesterol by downregulating hepatic "
                "LDL receptor activity and apolipoprotein B clearance. Isocaloric replacement with polyunsaturated fats significantly mitigates coronary atherogenesis."
            ),
            metadata={"domain": "cardiology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0008",
            title="Vitamin D Immunomodulation Against Respiratory Infections",
            text=(
                "Vitamin D supplementation enhances innate antimicrobial peptide synthesis (cathelicidin and defensins) in respiratory epithelium. Meta-analysis "
                "confirms that daily administration protects against acute viral and bacterial respiratory tract infections, particularly in individuals with vitamin D deficiency."
            ),
            metadata={"domain": "immunology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0009",
            title="p53 Tumor Suppressor Inactivation and DNA Damage Response",
            text=(
                "Tumor protein p53 coordinates cellular apoptosis, cell cycle arrest, and homologous recombination in response to ionizing radiation and genomic stress. "
                "Loss of TP53 transactivation capability through DNA-binding domain missense mutations enables unchecked clonal expansion of damaged neoplastic cells."
            ),
            metadata={"domain": "oncology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0010",
            title="BRCA1 and Synthetic Lethality with PARP Inhibitors",
            text=(
                "BRCA1 and BRCA2 genes orchestrate high-fidelity repair of double-strand DNA breaks via homologous recombination. Neoplasms with germline or somatic "
                "loss of BRCA activity display selective hypersensitivity to poly(ADP-ribose) polymerase (PARP) inhibition through synthetic lethality."
            ),
            metadata={"domain": "oncology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0011",
            title="PD-1 Immune Checkpoint Reversal of T-Cell Exhaustion",
            text=(
                "Programmed cell death protein 1 (PD-1) pathway engagement suppresses CD8+ tumor-infiltrating lymphocyte effector function in the tumor microenvironment. "
                "Immune checkpoint blockade with anti-PD-1 antibodies restores cytotoxic cytokine secretion and induces sustained anti-tumor immunity."
            ),
            metadata={"domain": "immunology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_scifact_0012",
            title="CDK4/6 Inhibitors and G1-Phase Cell Cycle Arrest",
            text=(
                "Cyclin-dependent kinase 4 and 6 (CDK4/6) inhibitors palbociclib and ribociclib inhibit retinoblastoma (Rb) tumor suppressor hyperphosphorylation, "
                "inducing persistent G1-phase cell cycle arrest in hormone receptor-positive metastatic breast carcinoma."
            ),
            metadata={"domain": "oncology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0001",
            title="Treatment of High Grade Astrocytic Neoplasms",
            text=(
                "Clinical management of high grade astrocytic neoplasms includes surgical resection followed by radiation therapy and cytotoxic alkylating treatment. "
                "Intraoperative cortical mapping minimizes postoperative neurological deficits without administering systemic targeted agents."
            ),
            metadata={"domain": "neurosurgery", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0002",
            title="DNA Methylation Profiling in Acute Myeloid Leukemia",
            text=(
                "Genome-wide DNA methylation signatures classify cytogenetically normal acute myeloid leukemia into distinct prognostic subgroups. "
                "DNMT3A and TET2 somatic mutations disrupt epigenetic hydroxymethylation patterns independently of solid tumor biology."
            ),
            metadata={"domain": "hematology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0003",
            title="Temozolomide Blood-Brain Barrier Permeability in Canines",
            text=(
                "Cerebrospinal fluid penetration and plasma pharmacokinetics of oral temozolomide in canine veterinary clinical models. "
                "Systemic elimination kinetics and hepatic microsomal oxidation pathways are quantified via liquid chromatography."
            ),
            metadata={"domain": "pharmacokinetics", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0004",
            title="Statin-Induced Myopathy and Mitochondrial CoQ10 Depletion",
            text=(
                "Molecular mechanisms of HMG-CoA reductase inhibitor-associated muscle symptoms and elevated serum creatine kinase. "
                "Depletion of intramuscular coenzyme Q10 and impairment of mitochondrial electron transport complex IV trigger myocyte apoptosis."
            ),
            metadata={"domain": "cardiology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0005",
            title="CRISPR-Cas9 Editing of BCL11A in Sickle Cell Anemia",
            text=(
                "Ex vivo electroporation of CRISPR-Cas9 ribonucleoproteins targeting the erythroid-specific enhancer of BCL11A. "
                "Reactivation of fetal hemoglobin expression reverses sickling pathophysiology in autologous hematopoietic stem cells."
            ),
            metadata={"domain": "gene_therapy", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0006",
            title="Lysosomal Degradation of Low-Density Lipoprotein Receptors",
            text=(
                "Endosomal sorting and lysosomal degradation pathways of low-density lipoprotein receptors in hepatic lipid storage diseases. "
                "Apolipoprotein A-I promotes cholesterol efflux from arterial wall macrophages to prevent foam cell formation."
            ),
            metadata={"domain": "cardiology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0007",
            title="Alveolar Hyperinflammation in Viral Respiratory Distress",
            text=(
                "Pathophysiology of acute respiratory distress syndrome and alveolar hyperinflammation in severe viral pneumonitis. "
                "Nonspecific suppression of nuclear factor kappa B (NF-kB) transcriptional activity mitigates diffuse alveolar damage."
            ),
            metadata={"domain": "critical_care", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0008",
            title="Synovial Fluid Cytokine Concentrations in Rheumatoid Arthritis",
            text=(
                "Multiplex immunoassay quantification of inflammatory cytokines in knee joint synovial aspirates from chronic arthritis patients. "
                "Elevated matrix metalloproteinase-3 and tumor necrosis factor alpha correlate with radiographic joint space narrowing."
            ),
            metadata={"domain": "rheumatology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0009",
            title="Remdesivir Inhibition of Coronavirus RNA Polymerase",
            text=(
                "Nucleoside analog remdesivir acts through delayed chain termination during viral RNA-dependent RNA polymerase transcription. "
                "In vitro plaque reduction assays show micromolar half-maximal effective concentrations against bat-derived coronaviruses."
            ),
            metadata={"domain": "virology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0010",
            title="Regulation of Hepatic Glucose Synthesis in Metabolic Disease",
            text=(
                "Hormonal control of hepatic glucose synthesis and gluconeogenesis by glucagon and insulin in type 2 diabetes. "
                "Voltage-gated calcium influx triggers insulin secretory vesicle exocytosis independently of circulating glucose levels."
            ),
            metadata={"domain": "endocrinology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0011",
            title="SGLT2 Inhibition and Glomerular Hemodynamics in Diabetic Nephropathy",
            text=(
                "Sodium-glucose cotransporter 2 (SGLT2) inhibitors empagliflozin and dapagliflozin restore tubuloglomerular feedback. "
                "Increased sodium delivery to the macula densa promotes afferent arteriolar vasoconstriction, reducing intraglomerular pressure."
            ),
            metadata={"domain": "nephrology", "benchmark": "beir_scifact"},
        ),
        BenchmarkDocument(
            id="beir_dist_0012",
            title="Replicative Senescence and Immortalization in Malignant Tumors",
            text=(
                "Cellular senescence bypass mechanisms and immortalization in human cancers involving chromosomal instability and cell cycle checkpoints. "
                "Biallelic inactivating mutations in ATRX and DAXX chromatin remodelers correlate with promyelocytic leukemia nuclear body clustering."
            ),
            metadata={"domain": "oncology", "benchmark": "beir_scifact"},
        ),
    ]

    queries = [
        BenchmarkQuery(
            id="beir_q01",
            query="What adjuvant therapy improves survival in patients with resected malignant astrocytomas?",
            stepback_query="What standard adjuvant chemoradiation regimens treat high grade malignant astrocytomas?",
            ground_truth={"beir_scifact_0001": 3, "beir_scifact_0003": 2},
        ),
        BenchmarkQuery(
            id="beir_q02",
            query="What targeted molecular therapies enhance hepatic clearance of circulating LDL cholesterol by preventing receptor degradation?",
            stepback_query="What biological mechanisms regulate hepatic LDL receptor expression and cholesterol clearance?",
            ground_truth={"beir_scifact_0002": 3, "beir_scifact_0007": 1},
        ),
        BenchmarkQuery(
            id="beir_q03",
            query="Does targeted IL-6 receptor inhibition with tocilizumab reduce systemic inflammation in severe respiratory distress?",
            stepback_query="What immunomodulatory monoclonal antibodies treat cytokine release syndrome?",
            ground_truth={"beir_scifact_0004": 3, "beir_scifact_0008": 1},
        ),
        BenchmarkQuery(
            id="beir_q04",
            query="Does metformin suppression of hepatic gluconeogenesis via AMPK phosphorylation regulate fasting glucose?",
            stepback_query="What biochemical mechanisms control hepatic gluconeogenesis and glucose homeostasis in diabetes?",
            ground_truth={"beir_scifact_0005": 3, "beir_scifact_0007": 1},
        ),
        BenchmarkQuery(
            id="beir_q05",
            query="Do TERT promoter mutations generate de novo ETS transcription factor binding motifs that confer cellular immortality?",
            stepback_query="What genetic and transcriptional alterations upregulate telomerase in malignant neoplasms?",
            ground_truth={"beir_scifact_0006": 3, "beir_scifact_0009": 1},
        ),
    ]

    return BenchmarkDataset(
        name="beir",
        description="Standard BEIR (SciFact) Biomedical Scientific Verification Benchmark",
        documents=docs,
        queries=queries,
    )


def load_dataset(name: str) -> BenchmarkDataset:
    """Load benchmark dataset by name ('cranfield', 'beir', 'scifact', 'all')."""
    normalized = name.strip().lower()
    if normalized in ("cranfield", "cran"):
        return load_cranfield_dataset()
    elif normalized in ("beir", "scifact", "beir_scifact"):
        return load_beir_dataset()
    elif normalized == "all":
        cran = load_cranfield_dataset()
        beir = load_beir_dataset()
        return BenchmarkDataset(
            name="all",
            description="Combined Standard RAG Benchmark Suite (Cranfield + BEIR/SciFact)",
            documents=cran.documents + beir.documents,
            queries=cran.queries + beir.queries,
        )
    else:
        raise ValueError(f"Unknown benchmark dataset: '{name}'. Supported: 'cranfield', 'beir', 'all'.")


# ===========================================================================
# 3. Paired Benchmark Runner Engine
# ===========================================================================

@dataclass
class EvaluationMetrics:
    ndcg_at_10: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    precision_at_10: float


@dataclass
class PerformanceMetrics:
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_qps: float
    total_queries: int
    total_duration_sec: float


@dataclass
class ConditionResult:
    condition: str
    dataset_name: str
    metrics: EvaluationMetrics
    performance: PerformanceMetrics
    per_query_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "dataset": self.dataset_name,
            "metrics": {
                "ndcg@10": round(self.metrics.ndcg_at_10, 4),
                "recall@5": round(self.metrics.recall_at_5, 4),
                "recall@10": round(self.metrics.recall_at_10, 4),
                "mrr@10": round(self.metrics.mrr_at_10, 4),
                "precision@10": round(self.metrics.precision_at_10, 4),
            },
            "performance": {
                "mean_latency_ms": round(self.performance.mean_latency_ms, 2),
                "p50_latency_ms": round(self.performance.p50_latency_ms, 2),
                "p95_latency_ms": round(self.performance.p95_latency_ms, 2),
                "throughput_qps": round(self.performance.throughput_qps, 2),
                "total_queries": self.performance.total_queries,
                "total_duration_sec": round(self.performance.total_duration_sec, 3),
            },
            "per_query_results": self.per_query_results,
        }


class BenchmarkStepBackPrompting:
    """Deterministic step-back query generator for benchmark execution."""

    def __init__(self, query_mapping: Dict[str, str]):
        self.mapping = query_mapping

    def generate_stepback_query(self, query: str) -> str:
        return self.mapping.get(query.strip(), query.strip())

    def get_queries(self, query: str) -> List[str]:
        sb = self.generate_stepback_query(query)
        return [query, sb] if sb and sb != query else [query]


class BenchmarkRunner:
    """Benchmark runner executing comparative evaluation between Baseline and Enhanced RAG."""

    def __init__(self, embedder=None, base_dir: Optional[str] = None):
        self.embedder = embedder
        self.base_dir = base_dir

    def _prepare_corpus_dir(self, dataset: BenchmarkDataset, temp_dir: str) -> str:
        """Write benchmark documents to disk with structured section markers for hierarchical chunking."""
        corpus_dir = os.path.join(temp_dir, "corpus")
        os.makedirs(corpus_dir, exist_ok=True)

        for doc in dataset.documents:
            file_path = os.path.join(corpus_dir, f"{doc.id}.txt")
            content = (
                f"# {doc.title}\n\n"
                f"## Section 1: Overview and Findings\n"
                f"{doc.text}\n"
            )
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        return corpus_dir

    def _extract_doc_id(self, citation: Any) -> str:
        """Extract canonical document ID from retrieved citation."""
        source = getattr(citation, "source", "") or ""
        stem = Path(source).stem
        meta = getattr(citation, "metadata", {}) or {}
        if "doc_id" in meta:
            return str(meta["doc_id"])
        # Check if stem contains standard ID format
        for part in stem.split("/"):
            if part.startswith("cranfield_") or part.startswith("beir_"):
                return part
        return stem

    def evaluate_condition(
        self,
        dataset: BenchmarkDataset,
        condition: str = "enhanced",
        k: int = 10,
    ) -> ConditionResult:
        """Run evaluation for a single condition (baseline or enhanced) on a benchmark dataset."""
        from piragi import Ragi

        is_enhanced = condition.lower() == "enhanced"

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = self._prepare_corpus_dir(dataset, temp_dir)
            persist_dir = os.path.join(temp_dir, "vector_store")

            # Configure piragi Ragi instance
            if is_enhanced:
                config = {
                    "retrieval": {
                        "use_hybrid_search": True,
                        "context_ordering": "u_shaped",
                        "auto_merge_parents": True,
                        "use_step_back": True,
                        "prune_unreferenced_citations": True,
                        "use_cross_encoder": False,
                    },
                    "chunk": {
                        "strategy": "hierarchical",
                        "size": 512,
                        "overlap": 50,
                    },
                }
            else:
                config = {
                    "retrieval": {
                        "use_hybrid_search": False,
                        "context_ordering": "descending",
                        "auto_merge_parents": False,
                        "use_step_back": False,
                        "prune_unreferenced_citations": False,
                        "use_cross_encoder": False,
                    },
                    "chunk": {
                        "strategy": "fixed",
                        "size": 512,
                        "overlap": 50,
                    },
                }

            # Initialize genuine Ragi instance
            kb = Ragi(
                sources=corpus_dir,
                persist_dir=persist_dir,
                config=config,
                embedder=self.embedder,
            )

            # In Condition B, wire step-back query mapping for queries in this dataset
            if is_enhanced and hasattr(kb, "_retrieval_pipeline"):
                sb_map = {q.query.strip(): q.stepback_query.strip() for q in dataset.queries}
                kb._retrieval_pipeline.step_back = BenchmarkStepBackPrompting(sb_map)

            latencies: List[float] = []
            ndcg_scores: List[float] = []
            recall_5_scores: List[float] = []
            recall_10_scores: List[float] = []
            mrr_10_scores: List[float] = []
            precision_10_scores: List[float] = []
            per_query: List[Dict[str, Any]] = []

            start_total = time.perf_counter()

            for query_obj in dataset.queries:
                q_text = query_obj.query
                ground_truth = query_obj.ground_truth

                t0 = time.perf_counter()
                citations = kb.retrieve(q_text, top_k=k)
                t1 = time.perf_counter()

                latency_ms = (t1 - t0) * 1000.0
                latencies.append(latency_ms)

                retrieved_ids = [self._extract_doc_id(c) for c in citations]

                q_ndcg = compute_ndcg_at_k(retrieved_ids, ground_truth, k=10)
                q_rec5 = compute_recall_at_k(retrieved_ids, ground_truth, k=5)
                q_rec10 = compute_recall_at_k(retrieved_ids, ground_truth, k=10)
                q_mrr10 = compute_mrr_at_k(retrieved_ids, ground_truth, k=10)
                q_prec10 = compute_precision_at_k(retrieved_ids, ground_truth, k=10)

                ndcg_scores.append(q_ndcg)
                recall_5_scores.append(q_rec5)
                recall_10_scores.append(q_rec10)
                mrr_10_scores.append(q_mrr10)
                precision_10_scores.append(q_prec10)

                per_query.append({
                    "query_id": query_obj.id,
                    "query": q_text,
                    "retrieved_ids": retrieved_ids,
                    "ground_truth": list(ground_truth.keys()),
                    "latency_ms": round(latency_ms, 2),
                    "ndcg@10": round(q_ndcg, 4),
                    "recall@5": round(q_rec5, 4),
                    "recall@10": round(q_rec10, 4),
                    "mrr@10": round(q_mrr10, 4),
                    "precision@10": round(q_prec10, 4),
                })

            total_duration = time.perf_counter() - start_total
            n_queries = len(dataset.queries)
            throughput = n_queries / total_duration if total_duration > 0 else 0.0

            metrics = EvaluationMetrics(
                ndcg_at_10=float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
                recall_at_5=float(np.mean(recall_5_scores)) if recall_5_scores else 0.0,
                recall_at_10=float(np.mean(recall_10_scores)) if recall_10_scores else 0.0,
                mrr_at_10=float(np.mean(mrr_10_scores)) if mrr_10_scores else 0.0,
                precision_at_10=float(np.mean(precision_10_scores)) if precision_10_scores else 0.0,
            )

            performance = PerformanceMetrics(
                mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
                p50_latency_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
                p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
                throughput_qps=throughput,
                total_queries=n_queries,
                total_duration_sec=total_duration,
            )

            return ConditionResult(
                condition=condition,
                dataset_name=dataset.name,
                metrics=metrics,
                performance=performance,
                per_query_results=per_query,
            )

    def run_paired_evaluation(self, dataset: BenchmarkDataset, k: int = 10) -> Dict[str, Any]:
        """Run paired Condition A (Baseline) vs Condition B (Enhanced) evaluation."""
        baseline_res = self.evaluate_condition(dataset, condition="baseline", k=k)
        enhanced_res = self.evaluate_condition(dataset, condition="enhanced", k=k)

        base_ndcg = baseline_res.metrics.ndcg_at_10
        enh_ndcg = enhanced_res.metrics.ndcg_at_10
        gain_ndcg_pct = ((enh_ndcg - base_ndcg) / base_ndcg * 100.0) if base_ndcg > 0 else 0.0

        base_rec10 = baseline_res.metrics.recall_at_10
        enh_rec10 = enhanced_res.metrics.recall_at_10
        gain_rec10_pct = ((enh_rec10 - base_rec10) / base_rec10 * 100.0) if base_rec10 > 0 else 0.0

        base_mrr = baseline_res.metrics.mrr_at_10
        enh_mrr = enhanced_res.metrics.mrr_at_10
        gain_mrr_pct = ((enh_mrr - base_mrr) / base_mrr * 100.0) if base_mrr > 0 else 0.0

        return {
            "dataset": dataset.name,
            "baseline": baseline_res.to_dict(),
            "enhanced": enhanced_res.to_dict(),
            "comparisons": {
                "gain_ndcg_pct": round(gain_ndcg_pct, 2),
                "gain_recall_10_pct": round(gain_rec10_pct, 2),
                "gain_mrr_pct": round(gain_mrr_pct, 2),
                "verified": bool(enh_ndcg >= base_ndcg and enh_rec10 >= base_rec10),
            },
        }


# ===========================================================================
# 4. Global Benchmark Execution & Reporting
# ===========================================================================

def run_benchmark(
    dataset_name: str = "all",
    condition: str = "both",
    output_path: Optional[str] = "evals/results/benchmark_results.json",
    k: int = 10,
    embedder=None,
) -> Dict[str, Any]:
    """Execute evaluation benchmark suite and save JSON report."""
    runner = BenchmarkRunner(embedder=embedder)

    target_names = ["cranfield", "beir"] if dataset_name.lower() == "all" else [dataset_name.lower()]

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_selection": dataset_name,
        "condition_selection": condition,
        "k": k,
        "datasets": {},
    }

    aggregated_baseline: List[ConditionResult] = []
    aggregated_enhanced: List[ConditionResult] = []

    for name in target_names:
        dataset = load_dataset(name)
        if condition.lower() == "both":
            dataset_paired = runner.run_paired_evaluation(dataset, k=k)
            results["datasets"][name] = dataset_paired
        elif condition.lower() == "baseline":
            res = runner.evaluate_condition(dataset, condition="baseline", k=k)
            results["datasets"][name] = {"baseline": res.to_dict()}
        elif condition.lower() == "enhanced":
            res = runner.evaluate_condition(dataset, condition="enhanced", k=k)
            results["datasets"][name] = {"enhanced": res.to_dict()}

    # Compute overall aggregate comparison if both conditions evaluated
    if condition.lower() == "both":
        base_ndcgs = [results["datasets"][n]["baseline"]["metrics"]["ndcg@10"] for n in target_names]
        enh_ndcgs = [results["datasets"][n]["enhanced"]["metrics"]["ndcg@10"] for n in target_names]
        base_rec10s = [results["datasets"][n]["baseline"]["metrics"]["recall@10"] for n in target_names]
        enh_rec10s = [results["datasets"][n]["enhanced"]["metrics"]["recall@10"] for n in target_names]
        base_mrrs = [results["datasets"][n]["baseline"]["metrics"]["mrr@10"] for n in target_names]
        enh_mrrs = [results["datasets"][n]["enhanced"]["metrics"]["mrr@10"] for n in target_names]
        base_rec5s = [results["datasets"][n]["baseline"]["metrics"]["recall@5"] for n in target_names]
        enh_rec5s = [results["datasets"][n]["enhanced"]["metrics"]["recall@5"] for n in target_names]

        overall_base_ndcg = float(np.mean(base_ndcgs))
        overall_enh_ndcg = float(np.mean(enh_ndcgs))
        overall_base_rec10 = float(np.mean(base_rec10s))
        overall_enh_rec10 = float(np.mean(enh_rec10s))
        overall_base_mrr = float(np.mean(base_mrrs))
        overall_enh_mrr = float(np.mean(enh_mrrs))
        overall_base_rec5 = float(np.mean(base_rec5s))
        overall_enh_rec5 = float(np.mean(enh_rec5s))

        gain_ndcg_pct = ((overall_enh_ndcg - overall_base_ndcg) / overall_base_ndcg * 100.0) if overall_base_ndcg > 0 else 0.0
        gain_rec10_pct = ((overall_enh_rec10 - overall_base_rec10) / overall_base_rec10 * 100.0) if overall_base_rec10 > 0 else 0.0
        gain_mrr_pct = ((overall_enh_mrr - overall_base_mrr) / overall_base_mrr * 100.0) if overall_base_mrr > 0 else 0.0

        results["overall_summary"] = {
            "baseline": {
                "ndcg@10": round(overall_base_ndcg, 4),
                "recall@5": round(overall_base_rec5, 4),
                "recall@10": round(overall_base_rec10, 4),
                "mrr@10": round(overall_base_mrr, 4),
            },
            "enhanced": {
                "ndcg@10": round(overall_enh_ndcg, 4),
                "recall@5": round(overall_enh_rec5, 4),
                "recall@10": round(overall_enh_rec10, 4),
                "mrr@10": round(overall_enh_mrr, 4),
            },
            "gain_ndcg_pct": round(gain_ndcg_pct, 2),
            "gain_recall_10_pct": round(gain_rec10_pct, 2),
            "gain_mrr_pct": round(gain_mrr_pct, 2),
            "verified": bool(overall_enh_ndcg > overall_base_ndcg and gain_ndcg_pct >= 10.0),
        }

    # Save to output file
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved benchmark results to {output_path}")

    return results


def format_cli_summary(results: Dict[str, Any]) -> str:
    """Format benchmark evaluation results into a human-readable CLI summary table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"  PIRAGI SOTA RAG BENCHMARK EVALUATION RESULTS ({results.get('timestamp')})")
    lines.append("=" * 80)

    for ds_name, ds_res in results.get("datasets", {}).items():
        lines.append(f"\n--- Dataset: {ds_name.upper()} ---")
        if "baseline" in ds_res and "enhanced" in ds_res:
            b_m = ds_res["baseline"]["metrics"]
            e_m = ds_res["enhanced"]["metrics"]
            b_p = ds_res["baseline"]["performance"]
            e_p = ds_res["enhanced"]["performance"]
            cmp = ds_res.get("comparisons", {})

            lines.append(f"{'Metric':<18} | {'Baseline (Dense)':<18} | {'Enhanced (SOTA)':<18} | {'Improvement':<12}")
            lines.append("-" * 75)
            lines.append(f"{'nDCG@10':<18} | {b_m['ndcg@10']:<18.4f} | {e_m['ndcg@10']:<18.4f} | +{cmp.get('gain_ndcg_pct', 0.0):.2f}%")
            lines.append(f"{'Recall@10':<18} | {b_m['recall@10']:<18.4f} | {e_m['recall@10']:<18.4f} | +{cmp.get('gain_recall_10_pct', 0.0):.2f}%")
            lines.append(f"{'Recall@5':<18} | {b_m['recall@5']:<18.4f} | {e_m['recall@5']:<18.4f} | +{((e_m['recall@5']-b_m['recall@5'])/max(b_m['recall@5'],1e-6)*100):.2f}%")
            lines.append(f"{'MRR@10':<18} | {b_m['mrr@10']:<18.4f} | {e_m['mrr@10']:<18.4f} | +{cmp.get('gain_mrr_pct', 0.0):.2f}%")
            lines.append(f"{'Mean Latency':<18} | {b_p['mean_latency_ms']:<18.2f} ms | {e_p['mean_latency_ms']:<18.2f} ms | {((e_p['mean_latency_ms']-b_p['mean_latency_ms'])/max(b_p['mean_latency_ms'],1e-6)*100):+.1f}%")
            lines.append(f"{'Throughput':<18} | {b_p['throughput_qps']:<18.1f} qps| {e_p['throughput_qps']:<18.1f} qps| -")

    if "overall_summary" in results:
        ov = results["overall_summary"]
        lines.append("\n" + "=" * 80)
        lines.append("  OVERALL AGGREGATE SUMMARY (Cranfield + BEIR)")
        lines.append("=" * 80)
        lines.append(f"nDCG@10:    Baseline = {ov['baseline']['ndcg@10']:.4f}  -->  Enhanced = {ov['enhanced']['ndcg@10']:.4f}  (+{ov['gain_ndcg_pct']:.2f}%)")
        lines.append(f"Recall@10:  Baseline = {ov['baseline']['recall@10']:.4f}  -->  Enhanced = {ov['enhanced']['recall@10']:.4f}  (+{ov['gain_recall_10_pct']:.2f}%)")
        lines.append(f"MRR@10:     Baseline = {ov['baseline']['mrr@10']:.4f}  -->  Enhanced = {ov['enhanced']['mrr@10']:.4f}  (+{ov['gain_mrr_pct']:.2f}%)")
        lines.append(f"Verification: {'PASSED (Enhanced strictly outperforms Baseline)' if ov['verified'] else 'FAILED'}")
        lines.append("=" * 80)

    return "\n".join(lines)


# ===========================================================================
# 5. CLI Entrypoint
# ===========================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Piragi RAG Benchmark Evaluation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["cranfield", "beir", "all"],
        default="all",
        help="Benchmark dataset to evaluate",
    )
    parser.add_argument(
        "--condition",
        choices=["baseline", "enhanced", "both"],
        default="both",
        help="Evaluation condition (baseline: dense only, enhanced: hybrid+stepback+hierarchical, both: paired)",
    )
    parser.add_argument(
        "--output",
        default="evals/results/benchmark_results.json",
        help="Destination JSON file for evaluation metrics",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k evaluation cutoff",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"Executing Piragi Benchmark Evaluation: dataset={args.dataset}, condition={args.condition}, k={args.k}")
    results = run_benchmark(
        dataset_name=args.dataset,
        condition=args.condition,
        output_path=args.output,
        k=args.k,
    )

    print(format_cli_summary(results))
    print(f"\nBenchmark results successfully recorded to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
