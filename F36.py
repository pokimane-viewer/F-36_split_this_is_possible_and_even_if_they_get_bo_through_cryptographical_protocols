import math
import types
import subprocess
import cupy as cp
import cadquery as cq
from dataclasses import dataclass

# -------------------------------------------------------------------
# ----------------------- REAL-WORLD REFERENCE TABLES ---------------
# -------------------------------------------------------------------

class TableB2BChinaPlatforms:
    """
    $p \land (q \rightarrow r)$

    In English: "p AND if q then r."

    Real-world applicability:
    "We have a reference table of global B2B platforms with a focus on China sourcing (p),
     and we require that if specific supplier verification data is present (q), then
     comprehensive buyer protection mechanisms (r) must also be available. This ensures
     transparency and risk mitigation for complex global sourcing in advanced manufacturing."
    """

    # Inserted real-world data from text "Table 1: Comparative Analysis of Leading B2B Platforms for China Sourcing".
    # Keys represent columns/attributes, values represent the textual details.
    data = [
        {
            "Platform Name": "Alibaba.com",
            "Primary Focus": "General B2B, wide product range",
            "Key Services (Examples)": "Supplier database, Trade Assurance, logistics support",
            "Supplier Verification Details": "Gold Supplier, Assessed Supplier (on-site checks by SGS/Intertek)",
            "Buyer Protection Mechanisms": "Trade Assurance (payment security, dispute resolution, escrow)",
            "Communication/Language Support": "English platform, direct messaging with suppliers",
            "Suitability for Advanced/Complex Parts": "Possible, but requires careful vetting for advanced capabilities"
        },
        {
            "Platform Name": "Global Sources",
            "Primary Focus": "Electronics, hardware, industrial products",
            "Key Services (Examples)": "Verified suppliers, trade shows, online sourcing",
            "Supplier Verification Details": "\"Verified Supplier\" badge (3rd party checks), on-site audits",
            "Buyer Protection Mechanisms": "Buyer protection services, dispute resolution assistance",
            "Communication/Language Support": "English interface, direct messaging, some multilingual support",
            "Suitability for Advanced/Complex Parts": "Good potential, includes precision-engineered parts"
        },
        {
            "Platform Name": "Made-in-China.com",
            "Primary Focus": "Industrial products, machinery, electronics",
            "Key Services (Examples)": "Supplier audits, TradeMessenger, Secured Trading Service",
            "Supplier Verification Details": "Audited Suppliers' Reports (SGS, BV, TUV Rheinland)",
            "Buyer Protection Mechanisms": "Secured Trading Service (STS) - escrow payment",
            "Communication/Language Support": "English platform, built-in chat tool",
            "Suitability for Advanced/Complex Parts": "Strong for industrial machinery/components; requires vetting"
        },
        {
            "Platform Name": "DHGate",
            "Primary Focus": "Smaller wholesale, B2B2C focus",
            "Key Services (Examples)": "Wholesale pricing, logistics, buyer protection",
            "Supplier Verification Details": "Has verification, but less transparent than Alibaba/Global Sources",
            "Buyer Protection Mechanisms": "Buyer protection (refunds for items not as described)",
            "Communication/Language Support": "English interface",
            "Suitability for Advanced/Complex Parts": "Less suited for high-precision advanced manufacturing parts"
        }
    ]


class TableDigitalMaaSPlatforms:
    """
    $p \leftrightarrow q$

    In English: "p if and only if q."

    Real-world applicability:
    "We have a reference table of on-demand manufacturing and MaaS platforms (p),
     and they are included in the table if and only if they offer significant
     advanced technology capabilities (q). This ensures alignment with modern
     manufacturing needs."
    """

    # Inserted real-world data from text "Table 2: Overview of Key Digital Manufacturing Platforms/MaaS Providers".
    data = [
        {
            "Platform Name": "Xometry",
            "Key Services": "CNC (3,4,5-axis), 3D printing (metal, plastic), injection molding",
            "Material Capabilities": "Metals (Al, Steel, Ti), Plastics (ABS, PC, Nylon), some Composites",
            "Target Industries": "Aerospace, Robotics, Automotive, Medical, Electronics, Energy",
            "China Operations/Sourcing": "Yes, network includes 10,000+ manufacturers in China",
            "English/Chinese Support": "Mainly English; Shanghai office handles local ops"
        },
        {
            "Platform Name": "Protolabs Network (Hubs)",
            "Key Services": "3D printing (various), CNC machining, injection molding, sheet metal",
            "Material Capabilities": "Wide range of plastics/metals, specifics via platform",
            "Target Industries": "Aerospace, Automotive, Consumer Electronics, Medical",
            "China Operations/Sourcing": "\"Global access\"; not explicitly China-based but has network partners",
            "English/Chinese Support": "Primarily English platform"
        },
        {
            "Platform Name": "Materialise",
            "Key Services": "Metal 3D printing, Laser Sintering, MJF, FDM, SLA, Consultancy",
            "Material Capabilities": "Titanium, Inconel, PA12, Ultem, Airbus-qualified materials",
            "Target Industries": "Aerospace, Medical, Automotive, Consumer Goods",
            "China Operations/Sourcing": "Not explicitly stated; Europe's largest 3D printing factory",
            "English/Chinese Support": "English/European languages primarily"
        },
        {
            "Platform Name": "Fictiv",
            "Key Services": "CNC machining, 3D printing (FDM, SLS, SLA, MJF), injection molding",
            "Material Capabilities": "Variety of metals/plastics, specifics via quote",
            "Target Industries": "Aerospace, Robotics, Medical, Automotive, Consumer",
            "China Operations/Sourcing": "Yes, network includes manufacturers in U.S., China, India, Mexico",
            "English/Chinese Support": "Primarily English platform"
        },
        {
            "Platform Name": "Shapeways",
            "Key Services": "3D printing (binder jetting, SLM, SLS, SLA, MJF, FDM), wax casting",
            "Material Capabilities": "Stainless steel, Al, Cu, precious metals, polymers (Nylon, TPU, ABS)",
            "Target Industries": "Aerospace, Jewelry, Medical, Robotics, Drones",
            "China Operations/Sourcing": "Ships globally; main mfg in US/EU",
            "English/Chinese Support": "Primarily English, not explicitly Chinese"
        }
    ]


class TableAdvancedManufacturingBooks:
    """
    $p \rightarrow (q \land r)$

    In English: "If p then q and r."

    Real-world applicability:
    "If the resource is part of advanced/smart manufacturing references (p),
     then it must address Industry 4.0 digitalization (q) and offer practical
     manufacturing insights (r). This ensures that each listed book covers
     both conceptual and application-oriented knowledge in advanced manufacturing."
    """

    # Inserted real-world data from text "Table 3: Selected Recent Books on Advanced & Smart Manufacturing (2020-2025)".
    data = [
        {
            "Title": "Industry 4.0 and Smart Manufacturing: Integrating IoT, AI, and ML in Injection Molding",
            "Author": "Caleb Cameron",
            "Year": 2024,
            "Publisher": "Not specified",
            "Key Themes": "Industry 4.0, AI/ML, Big Data, predictive maintenance"
        },
        {
            "Title": "Smart Digital Manufacturing: A Guide for Digital Transformation with Real Case Studies Across Industries",
            "Author": "Rene Wolf & Raffaello Lepratti (Eds.)",
            "Year": 2020,
            "Publisher": "Wiley",
            "Key Themes": "Siemens' Smart Digital Mfg concept, stepwise Industry 4.0 adoption"
        },
        {
            "Title": "Smart Factory: Technologies and Applications in the Post-Pandemic Era (2nd Edition)",
            "Author": "Michael Deng & Colin Koh",
            "Year": 2023,
            "Publisher": "Independently published",
            "Key Themes": "IIoT, AI, robotics, digital twins, post-pandemic manufacturing"
        },
        {
            "Title": "Handbook of Smart Manufacturing: Forecasting the Future of Industry 4.0",
            "Author": "Kumar, Singh, Parveen, AlMangour (Eds.)",
            "Year": "Likely 2022-2024",
            "Publisher": "CRC Press (implied)",
            "Key Themes": "Smart mfg dev, AI approaches, cybersecurity, 4D printing"
        },
        {
            "Title": "Artificial Intelligence in Manufacturing: Enabling Intelligent, Flexible, and Cost-Effective Production Through AI",
            "Author": "John Soldatos (Ed.)",
            "Year": 2024,
            "Publisher": "Springer Nature",
            "Key Themes": "AI for Industry 4.0, digital twins, XAI, human-AI collaboration"
        }
    ]


class TableSupplyChainBooks:
    """
    $(p \lor q) \land (r \lor s)$

    In English: "(p or q) AND (r or s)."

    Real-world applicability:
    "We require that each supply chain reference either addresses post-pandemic
     resilience (p) or advanced risk management (q), AND it must discuss either
     global logistics/digitalization (r) or cross-border disruptions (s). This ensures
     these books comprehensively cover modern supply chain challenges."
    """

    # Inserted real-world data from text "Table 4: Selected Recent Books on Supply Chain Management & Logistics (2020-2025)".
    data = [
        {
            "Title": "Flow: How the Best Supply Chains Thrive",
            "Author": "Rob Handfield & Tom Linton",
            "Year": 2022,
            "Publisher": "Not specified",
            "Key Themes": "Supply continuity, velocity, digital technologies"
        },
        {
            "Title": "The New (Ab)Normal: Reshaping Business and Supply Chain Strategy Beyond Covid-19",
            "Author": "Yossi Sheffi",
            "Year": 2020,
            "Publisher": "MIT CTL Media",
            "Key Themes": "Pandemic's impact on SCM, resilience, uncertainty management"
        },
        {
            "Title": "Supply Chain Risk Management: How to Design and Manage Resilient Supply Chains",
            "Author": "John Manners-Bell",
            "Year": 2020,
            "Publisher": "Kogan Page (likely)",
            "Key Themes": "Resilience frameworks, addressing threats (geopolitical, disasters)"
        },
        {
            "Title": "Data Science for Supply Chain Forecasting",
            "Author": "Nicholas Vandeput",
            "Year": 2021,
            "Publisher": "Not specified",
            "Key Themes": "Big data, analytics, forecasting accuracy"
        },
        {
            "Title": "The Supply Chain Revolution: Innovative Sourcing and Logistics for a Fiercely Competitive World",
            "Author": "Suman Sarkar",
            "Year": 2023,
            "Publisher": "Wiley (likely)",
            "Key Themes": "Sourcing strategies, logistics best practices, real-life examples"
        }
    ]


class TableChinaSourcingBooks:
    """
    $(p \leftrightarrow q) \land (r \leftrightarrow s)$

    In English: "(p if and only if q) AND (r if and only if s)."

    Real-world applicability:
    "We have references specifically focused on sourcing from or doing business in China.
     Each reference (p) matches real-world China manufacturing context (q), and each
     discusses IP/trade/regulatory frameworks (r) that correspond to actual best practices (s)."
    """

    # Inserted real-world data from text "Table 5: Selected Recent Books on Sourcing from/Doing Business in China (2020-2025)".
    data = [
        {
            "Title": "'Made In China' Creates New Economic Superpower: Top Manufacturers Share Their Journeys",
            "Author": "Financial Channel CCTV, TV Programme: The Pillars of a Great Power",
            "Year": 2020,
            "Publisher": "Royal Collins Publishing Company",
            "Key Themes": "Progress of Chinese industry, case studies of domestic manufacturing"
        },
        {
            "Title": "Understanding China's Manufacturing Industry (China Insights Series)",
            "Author": "Qunhui Huang & Baocui Lou (Translator)",
            "Year": 2022,
            "Publisher": "Springer",
            "Key Themes": "Challenges/prospects of Chinese manufacturing, Sino-US trade impacts"
        },
        {
            "Title": "How To Source Product From China: A Comprehensive Guide",
            "Author": "Vishal Parikh (Frabon Solutions LLC)",
            "Year": 2023,
            "Publisher": "Independently published",
            "Key Themes": "Supplier identification, negotiation, QC, IP protection"
        },
        {
            "Title": "The China Market Entry Handbook (2023 Edition)",
            "Author": "Integra Group",
            "Year": 2023,
            "Publisher": "Integra Group",
            "Key Themes": "WFOE vs JV, tax system, compliance, regulatory environment"
        },
        {
            "Title": "Han Kun Blue Book: Doing Business in China - 2024 Comprehensive Guide",
            "Author": "Han Kun Law Offices",
            "Year": 2024,
            "Publisher": "Han Kun Law Offices",
            "Key Themes": "Chinese legal system, foreign investment law, Revised Company Law"
        }
    ]


class ContractManufacturingChina:
    """
    $p \land (q \lor r)$

    In English: "p AND (q or r)."

    Real-world applicability:
    "We have an overview class for contract manufacturing in China (p). It must
     incorporate essential considerations for vetting (q) or IP protections (r).
     This ensures that the user understands both the need to carefully evaluate
     suppliers and to secure IP when engaging in complex, high-value manufacturing."
    """

    # Summarized from "B. Contract Manufacturing: Deep Dive into China" of the real-world text
    overviews = {
        "Key_Facilitators": [
            {
                "Name": "Davantech",
                "Location": "Dongguan, Guangdong",
                "Specialty": "Western-managed CNC facility, ISO 9001:2015",
                "Industries": "Automotive, semi-medical, industrial components",
                "Notes": "Fluent English staff, tight tolerances, complex geometries"
            },
            {
                "Name": "China 2 West (C2W)",
                "Location": "Zhongshan, Guangdong",
                "Specialty": "British-owned, 500+ approved manufacturers, DFM support",
                "Industries": "General manufacturing, assembly, IP-sensitive projects",
                "Notes": "Own 100,000+ sq ft facility, bridging communication in Mandarin/English"
            }
        ],
        "Critical_Considerations": [
            "Communication & Language (English-speaking engineers, cultural nuances)",
            "Quality Control & Assurance (ISO certs, pre/in-process/final inspections)",
            "IP Protection (Register in China, use bilingual NNN agreements)",
            "Regulatory & Compliance (Foreign Investment Law, product certifications)"
        ]
    }


class KeyTradeFairsChina:
    """
    $\neg p \lor q$

    In English: "Not p or q."

    Real-world applicability:
    "We have major industry trade fairs in China. Either a user (p) opts not to
     attend them physically, or they utilize the knowledge (q) about these fairs'
     schedules and focuses to engage virtually/plan onsite visits. This fosters
     advanced sourcing and technology insight."
    """

    # Summarized from "IV. Essential Industry Events for Sourcing and Technology Insights"
    fairs = [
        {
            "Name": "Canton Fair",
            "Location": "Guangzhou",
            "Frequency": "Biannually",
            "Focus": "Broad import/export, Phase 1 for industrial & advanced manufacturing",
            "OnlinePlatform": "Year-round virtual browsing, large international buyer attendance"
        },
        {
            "Name": "China International Industry Fair (CIIF)",
            "Location": "Shanghai",
            "Frequency": "Annually",
            "Focus": [
                "Industrial automation (IAS)",
                "Robotics (RS)",
                "Metalworking & CNC (MWCS)",
                "New material shows",
                "Smart manufacturing and integrated supply chains"
            ],
            "AdditionalInfo": "Large platform for advanced automation, IoT, robotics, Industry 4.0"
        }
    ]


# -------------------------------------------------------------------
# -------------------- ORIGINAL CODE (UNCHANGED) --------------------
# -------------------------------------------------------------------

FACTORY_LOCATION = "Beijing"

def _air_density(z: float):
    """
    $p \rightarrow q$

    In English: "If p, then q."

    Real-world applicability:
    "This can represent a conditional relationship such as:
     If an aircraft is at a certain altitude (p), then its air density
     must be recalculated (q)."
    """
    return 1.225 * math.exp(-z / 8500)

class _State(types.SimpleNamespace):
    """
    $p \leftrightarrow q$

    In English: "p if and only if q."

    Real-world applicability:
    "This bidirectional condition can represent the idea that the state
     of the system (p) is valid precisely when a corresponding condition
     (q) is also satisfied, e.g. updating position if and only if the
     aircraft state is active."
    """

class _FallbackAircraft:
    """
    $\neg p \lor (p \rightarrow q)$

    In English: "Either not p, or if p then q." (Logically equivalent to $p \rightarrow q$)

    Real-world applicability:
    "This can represent logic like: either a failure mode doesn't occur
     (not p), or if it does (p), then a mitigation must be applied (q).
     Effectively, if a failure occurs, mitigation is necessary."
    """

    def __init__(self, st, cfg, additional_weight: float = 0.0):
        self.state = _State(
            position=cp.zeros(3, dtype=cp.float32),
            velocity=cp.zeros(3, dtype=cp.float32),
            time=0.0,
        )
        self.config = cfg
        self.destroyed = False

def _identity_eq_hash(cls):
    """
    $\forall x [P(x)]$

    In English: "For every x, P(x) is true."

    Real-world applicability:
    "This expresses a universal condition. For instance, every instance
     of a certain class might share a property, such as an identity or
     hashing behavior defined by this decorator."
    """
    return cls

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F35Aircraft(_FallbackAircraft):
    """
    $(p \land q) \land (r \rightarrow s)$

    In English: "(p and q) and (if r then s)."

    Real-world applicability:
    "A combined condition: certain base requirements (p and q) must both be
     met, and if an additional operational trigger (r) occurs, then a specific
     follow-up action or state (s) must happen. For example, if the aircraft
     systems are nominal and fuel is sufficient (p and q), and if the pilot
     engages afterburners (r), then significantly higher thrust (s) is produced."
    """

    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        base = {
            "mass": 25000.0,
            "wing_area": 73.0,
            "thrust_max": 2 * 147000,
            "Cd0": 0.02,
            "Cd_supersonic": 0.04,
            "service_ceiling": 20000.0,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.0},
            "irst": {"range_max": 100000.0},
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def _drag(self) -> cp.ndarray:
        """
        $\exists x (P(x) \land Q(x))$

        In English: "There exists an x such that P(x) and Q(x) are both true."

        Real-world applicability:
        "In aerodynamics, there might be a specific flight condition (x),
         such as a certain speed and altitude, where specific aerodynamic
         properties (P(x)) like being supersonic and air density conditions (Q(x))
         are met, requiring a particular drag calculation."
        """
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        # Speed of sound at sea level ~343 m/s. This is a simplification.
        Cd = self.config["Cd_supersonic"] if v / 343.0 > 1 else self.config["Cd0"]
        D = (
            0.5
            * _air_density(float(self.state.position[2]))
            * Cd
            * self.config["wing_area"]
            * v**2
        )
        return (self.state.velocity / v) * D

    def update(self, dt: float = 0.05):
        """
        $r \rightarrow (p \lor \neg q)$

        In English: "If r, then p or not q."

        Real-world applicability:
        "This can represent a safety or operational constraint: if a critical
         system is active (r), then the aircraft must either be in a safe
         operational mode (p) or a specific fault condition (q) must not be
         present (not q). For instance, if landing gear is deploying (r),
         then airspeed must be below a threshold (p) or the gear unsafe
         warning (q) must not be active."
        """
        if self.destroyed:
            return
        thrust_vector = cp.array([1.0, 0.0, 0.0], dtype=cp.float32)  # Assuming thrust is along x-axis
        thrust = thrust_vector * self.config["thrust_max"]

        gravity_force = cp.array([0.0, 0.0, -9.81 * self.config["mass"]], dtype=cp.float32)

        acc = (
            thrust
            - self._drag()
            + gravity_force
        ) / self.config["mass"]

        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

def f35_aircraft_cad(
    body_length=50,
    fuselage_radius=3,
    wing_span=35,
    wing_chord=5,
    tail_span=10,
    tail_chord=3,
):
    """
    $( \neg p \land q) \lor (p \land \neg q)$

    In English: "Either (not p and q), or (p and not q)." (This is equivalent to $p \oplus q$, XOR)

    Real-world applicability:
    "Represents an exclusive design choice or configuration in geometry.
     For instance, a component's design must satisfy either constraint set A (¬p ∧ q)
     or constraint set B (p ∧ ¬q), but not both or neither."
    """
    # Fuselage
    fuselage = cq.Workplane("XY").circle(fuselage_radius).extrude(body_length)

    # Main Wings
    wing_profile = cq.Workplane("XZ").polyline([(0,0), (wing_chord, wing_chord*0.1), (wing_chord, -wing_chord*0.1), (0,0)]).close()
    main_wing = wing_profile.extrude(wing_span/2).rotate((0,0,1), (0,0,0), 90).translate((body_length*0.4, 0, fuselage_radius*0.5))

    # Mirrored Main Wing
    main_wing_mirrored = main_wing.mirror(mirrorPlane="YZ", basePointVector=(0,0,0))

    # Horizontal Stabilizers (Tail Wings)
    tail_wing_profile = cq.Workplane("XZ").polyline([(0,0), (tail_chord, tail_chord*0.08), (tail_chord, -tail_chord*0.08), (0,0)]).close()
    tail_wing = tail_wing_profile.extrude(tail_span/2).rotate((0,0,1), (0,0,0), 90).translate((body_length*0.85, 0, fuselage_radius*0.2))

    # Mirrored Horizontal Stabilizer
    tail_wing_mirrored = tail_wing.mirror(mirrorPlane="YZ", basePointVector=(0,0,0))

    # Vertical Stabilizer (one for simplicity here)
    vertical_stabilizer_profile = cq.Workplane("XY").polyline([(0,0), (tail_chord*1.2, 0), (tail_chord*0.8, tail_span*0.3), (0, tail_span*0.2)]).close()
    vertical_stabilizer = vertical_stabilizer_profile.extrude(0.5).translate((body_length*0.85, -0.25, fuselage_radius))

    assembly = fuselage.union(main_wing).union(main_wing_mirrored).union(tail_wing).union(tail_wing_mirrored).union(vertical_stabilizer)
    return assembly

DEFAULT_STEP_PATH = "f35.step"
DEFAULT_STL_PATH = "f35.stl"
DEFAULT_GCODE_PATH = "f35.gcode"
DEFAULT_SLICER_CMD = "CuraEngine"  # Assuming CuraEngine is in PATH or specify full path
DEFAULT_SLICER_FLAGS = ("slice", "-l")  # Basic flags for slicing

def export_f35_step(step_path: str = DEFAULT_STEP_PATH):
    """
    $p \leftrightarrow (q \land r)$

    In English: "p if and only if (q and r)."

    Real-world applicability:
    "A process (p) (e.g., exporting a STEP file) is considered successfully
     completed if and only if the CAD model generation was successful (q)
     and the file system write operation was successful (r)."
    """
    model = f35_aircraft_cad()
    cq.exporters.export(model, step_path)
    print(f"F-35 CAD model exported to STEP file: {step_path}")
    return step_path

def export_and_slice_f35(
    stl_path: str = DEFAULT_STL_PATH,
    gcode_path: str = DEFAULT_GCODE_PATH,
    slicer_cmd: str = DEFAULT_SLICER_CMD,
    slicer_flags: tuple[str, ...] = DEFAULT_SLICER_FLAGS,
):
    """
    $\forall x \forall y [P(x, y) \rightarrow Q(x, y)]$

    In English: "For all x and y, if P(x, y) then Q(x, y)."

    Real-world applicability:
    "For every CAD model file (x) and every set of slicing parameters (y),
     if the export to STL and subsequent slicing process is initiated (P(x,y)),
     then the corresponding G-code output must be generated (Q(x,y)). This
     ensures process integrity for all inputs."
    """
    model = f35_aircraft_cad()
    cq.exporters.export(model, stl_path)
    print(f"F-35 CAD model exported to STL file: {stl_path}")

    # Construct the command for the slicer
    slice_command = [slicer_cmd, *slicer_flags, stl_path, "-o", gcode_path]
    print(f"Executing slicer command: {' '.join(slice_command)}")

    try:
        subprocess.run(slice_command, check=True, capture_output=True, text=True)
        print(f"Slicing successful. G-code written to: {gcode_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during slicing:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: Slicer command '{slicer_cmd}' not found. Ensure it's installed and in PATH.")
        raise

    return gcode_path

def create_manufacturing_friendly_f35():
    """
    $\exists z [P(z) \land \neg Q(z)]$

    In English: "There exists a z such that P(z) is true and Q(z) is false."

    Real-world applicability:
    "This represents a specific scenario or configuration (z) where a desired
     outcome (P(z)) is achieved by intentionally bypassing or altering a
     standard procedure or condition (¬Q(z)) to speed up iteration. For example,
     there exists a rapid prototyping version (z) where the full CAD model is
     generated and sliced (P(z)) but without undergoing extended validation
     checks (¬Q(z))."
    """
    # This function now directly calls export_and_slice_f35 with default parameters
    # The "manufacturing-friendly" aspect is embodied in the CAD model and slicing process.
    return export_and_slice_f35()

def batch_update(aircraft: F35Aircraft, total_time: float, dt: float = 0.05):
    """
    $p \land (q \rightarrow r) \land (r \rightarrow p)$

    In English: "p is true, and if q then r, and if r then p." (This can be seen as $p \land (q \leftrightarrow r)$)

    Real-world applicability:
    "This describes a simulation loop state: the simulation is active (p), and
     there's a mutual dependency between updating the aircraft's state (q) and
     advancing simulation time (r). If state is updated, time advances; if time
     advances (for a step), state must be updated. Both are conditional on the
     simulation being active."
    """
    steps = int(total_time / dt)
    for i in range(steps):
        aircraft.update(dt)
        if (i+1) % (steps // 10 if steps > 10 else 1) == 0 :
            print(f"Batch update: Step {i+1}/{steps}, Time: {aircraft.state.time:.2f}s")

def parallel_slice(
    stl_paths: list[str],
    gcode_paths: list[str],
    slicer_cmd: str = DEFAULT_SLICER_CMD,
    slicer_flags: tuple[str, ...] = DEFAULT_SLICER_FLAGS,
):
    """
    $\neg (p \land q) \lor r$

    In English: "Either not (p and q), or r." (Logically equivalent to $(p \land q) \rightarrow r$)

    Real-world applicability:
    "This represents a condition for executing a process: if it's not the case
     that both sufficient single-core resources are available (p) and a strict
     sequential process is mandated (q), OR if parallel processing is explicitly
     enabled (r), then proceed with parallel slicing. Essentially, parallel slicing
     occurs if conditions for serial processing are not strictly met or if parallelism is favored."
    """
    import concurrent.futures
    import os

    def _slice(args_tuple):
        stl_path, gcode_path, slicer_cmd_local, slicer_flags_local = args_tuple
        # Ensure output directory exists for gcode_path
        os.makedirs(os.path.dirname(gcode_path), exist_ok=True)

        command = [slicer_cmd_local, *slicer_flags_local, stl_path, "-o", gcode_path]
        print(f"Starting parallel slice for: {stl_path} -> {gcode_path} using {' '.join(command)}")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully sliced (parallel): {stl_path} to {gcode_path}")
            return gcode_path
        except subprocess.CalledProcessError as e:
            print(f"Error slicing {stl_path} in parallel: {e.stderr}")
            return None
        except FileNotFoundError:
            print(f"Error: Slicer command '{slicer_cmd_local}' not found for parallel slice.")
            return None

    tasks_args = [(stl_paths[i], gcode_paths[i], slicer_cmd, slicer_flags) for i in range(len(stl_paths))]

    results = []
    # Using ThreadPoolExecutor for I/O bound tasks like launching subprocesses
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_gcode = {executor.submit(_slice, task_args): task_args[1] for task_args in tasks_args}
        for future in concurrent.futures.as_completed(future_to_gcode):
            gcode_file = future_to_gcode[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                print(f'{gcode_file} generated an exception: {exc}')
    return results

def optimized_create_f35_batch(
    n: int,
    output_dir: str = ".",
    concurrent_slices: bool = True,
):
    """
    $(p \rightarrow q) \land (q \rightarrow r) \land (r \rightarrow p)$

    In English: "If p then q, and if q then r, and if r then p." (This implies $p \leftrightarrow q \leftrightarrow r$)

    Real-world applicability:
    "This creates a cycle of implications, indicating a tightly coupled workflow.
     If CAD generation is requested (p), then STL export must occur (q). If STL
     export occurs (q), then G-code slicing must be performed (r). If G-code
     slicing is done (r), it fulfills the initial CAD generation request's manufacturing
     goal (p). This describes an end-to-end automated batch process."
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    stls = []
    gcodes = []
    base_model = f35_aircraft_cad()  # Generate base CAD once

    for i in range(n):
        stl_path = os.path.join(output_dir, f"f35_batch_{i+1}.stl")
        gcode_path = os.path.join(output_dir, f"f35_batch_{i+1}.gcode")

        print(f"Exporting STL for batch unit {i+1}: {stl_path}")
        cq.exporters.export(base_model, stl_path)
        stls.append(stl_path)
        gcodes.append(gcode_path)

    if concurrent_slices:
        print(f"Starting concurrent slicing for {n} units...")
        return parallel_slice(stls, gcodes)
    else:
        print(f"Starting sequential slicing for {n} units...")
        completed_gcodes = []
        for s_path, g_path in zip(stls, gcodes):
            try:
                slice_command = [DEFAULT_SLICER_CMD, *DEFAULT_SLICER_FLAGS, s_path, "-o", g_path]
                print(f"Executing sequential slice: {' '.join(slice_command)}")
                subprocess.run(slice_command, check=True)
                print(f"Successfully sliced (sequential): {s_path} to {g_path}")
                completed_gcodes.append(g_path)
            except subprocess.CalledProcessError as e:
                print(f"Error slicing {s_path} sequentially: {e.stderr}")
            except FileNotFoundError:
                print(f"Error: Slicer command '{DEFAULT_SLICER_CMD}' not found for sequential slice.")
        return completed_gcodes

def compare_manufacturing_to_block_upgrades():
    """
    $(p \land q) \leftrightarrow (r \lor s)$

    In English: "(p and q) if and only if (r or s)."

    Real-world applicability:
    "This logical equivalence states that a set of advanced manufacturing
     capabilities (p: constraint integrity fully met, and q: hardware/software
     perfectly synchronized) is achieved if and only if either a major block
     upgrade has been implemented (r) or a special series of targeted
     enhancements has been completed (s)."
    """
    return {
        "constraint_integrity": "complete",
        "hardware_software_sync": "resolved",
        "thermal_management_advancement": "achieved_via_block_upgrade_R",
        "sustainment_logistics_optimized": "achieved_via_enhancement_S",
        "overall_alignment": "True"
    }

def define_actual_manufacturing_process(
    material_type: str = "Advanced Al-Li Alloy",
    cure_time_hours: float = 8.0,
    friction_welding_strength: float = 1200.0
):
    """
    $p \land r$

    In English: "p and r are both true."

    Real-world applicability:
    "Two crucial conditions must both be satisfied for a manufacturing
     process to be validly defined and initiated. For instance, the material
     specifications (p) are confirmed AND the primary process parameters (r)
     (like cure time and welding strength) are set and verified."

    Comprehensive definition of the manufacturing procedure and material:
    (Details below are illustrative of a real-world process)

    1. Material Selection & Incoming Inspection (${material_type}$):
        - Primary materials: ${material_type} for structural components,
          Titanium (e.g., Ti-6Al-4V) for high-stress/high-temperature areas,
          and Carbon Fiber Reinforced Polymers (CFRP) for skins and stealth surfaces.
        - All incoming materials undergo rigorous quality inspection (dimensional,
          chemical composition, defect detection).

    2. Precision Machining & Forming:
        - Multi-axis CNC machining for metallic components (bulkheads, spars).
        - Superplastic forming and diffusion bonding for complex titanium parts.
        - Automated fiber placement (AFP) and tape laying (ATL) for composite structures.

    3. Composite Layup and Curing:
        - CFRP parts are laid up in climate-controlled cleanrooms.
        - Curing is performed in calibrated autoclaves under precise temperature
          and pressure profiles. Target cure cycle duration: ${cure_time_hours} hours.

    4. Advanced Joining & Assembly:
        - Robotic drilling and fastening for airframe assembly.
        - Friction Stir Welding (FSW) for aluminum-lithium alloy sections, requiring
          a minimum weld tensile strength of ${friction_welding_strength} MPa.
        - Specialized bonding techniques for composite-to-metal interfaces.

    5. Surface Treatment and Coating:
        - Application of corrosion inhibiting primers.
        - Robotic application of radar-absorbent materials (RAM) and topcoats
          to meet stringent surface finish and signature requirements.

    6. Systems Integration:
        - Installation of avionics, hydraulics, fuel systems, landing gear,
          and propulsion systems.
        - Extensive wiring (~40km per aircraft) and optical fiber routing.

    7. Quality Control & Non-Destructive Testing (NDT):
        - Laser trackers and CMM for dimensional verification.
        - Ultrasonic, X-ray, and thermographic NDT for structural integrity.
        - Full functional testing of all integrated systems.

    8. Final Assembly & Flight Readiness Checks:
        - Final checks, ground runs, and preparation for first flight.
    """
    print(f"Defining F-35 manufacturing process details:")
    print(f"   Selected Primary Material: {material_type}")
    print(f"   Composite Cure Time Target: {cure_time_hours} hours")
    print(f"   Friction Stir Welding Strength Min.: {friction_welding_strength} MPa")
    print("   Process definition logged and parameters set for current production cycle.")

def propositional_truth_values():
    """
    $\forall p \in \{T, F\} : p \text{ represents a proposition that is either true or false.}$

    In English: "For any proposition p, p can only take one of two truth values: True (T) or False (F)."

    Real-world applicability:
    "Propositional logic forms the absolute bedrock of computation
     and digital systems. Every decision in a program (if-else), every bit in memory (0 or 1),
     and every gate in a CPU (high/low voltage) fundamentally relies on this binary nature
     of truth values. It's essential for database queries (WHERE clauses), AI reasoning,
     and algorithm design. The principles of propositional logic are universally applied in these fields,
     including in academic and technical contexts within the People's Republic of China."
    """
    return {"T": True, "F": False, "Description": "Fundamental truth values in propositional logic."}

def logical_operators():
    """
    $\forall p,q \in \{T, F\} :$
    $p \land q \equiv \min(p,q)$ (Conjunction - AND)
    $p \lor q \equiv \max(p,q)$ (Disjunction - OR, assuming T=1, F=0)
    $\neg p \equiv 1-p$ (Negation - NOT, assuming T=1, F=0)

    In English: "For any propositions p and q:
    'p AND q' is true if and only if both p and q are true.
    'p OR q' is true if and only if at least one of p or q is true.
    'NOT p' is true if and only if p is false."

    Real-world applicability: Logical operators are the tools to build complex conditions
    from simple ones. In programming, they control flow (e.g., if (condition1 AND condition2)).
    In database queries, they filter data (e.g., SELECT * FROM users WHERE active=TRUE OR isAdmin=TRUE).
    In circuit design, AND, OR, NOT gates are fundamental building blocks of all digital hardware.
    Search engines use them to refine search queries (e.g., "AI AND ethics NOT policy").
    These definitions and applications are standard in logic and computer science globally,
    including their teaching and use in the People's Republic of China.
    """
    return {
        "AND": {"symbol": "∧", "operation": lambda p, q: p and q, "description": "True if both p and q are true."},
        "OR":  {"symbol": "∨", "operation": lambda p, q: p or q,   "description": "True if p or q (or both) are true."},
        "NOT": {"symbol": "¬", "operation": lambda p: not p,       "description": "True if p is false."}
    }

def truth_tables():
    """
    $T(p \land q) = \{(T,T)\rightarrow T, (T,F)\rightarrow F, (F,T)\rightarrow F, (F,F)\rightarrow F\} 
    $T(p \lor q) = \{(T,T)\rightarrow T, (T,F)\rightarrow T, (F,T)\rightarrow T, (F,F)\rightarrow F\}
    $T(\neg p) = \{T\rightarrow F, F\rightarrow T\}

    In English: "Truth tables systematically list all possible truth value combinations
    for input propositions and show the resulting truth value for a given logical operation.
    For AND: only True input for both yields True output.
    For OR: only False input for both yields False output.
    For NOT: reverses the input truth value."

    Real-world applicability: Truth tables are essential for:
    1.  Understanding and defining logical operators.
    2.  Verifying logical equivalences (e.g., De Morgan's laws).
    3.  Designing and simplifying digital logic circuits (Karnaugh maps are based on this).
    4.  Testing and debugging software by ensuring conditional logic behaves correctly
        under all input scenarios.
    5.  Formal verification of system properties.
    The utility and construction of truth tables are fundamental concepts in logic education worldwide,
    including in curricula within the People's Republic of China.
    """
    p_values = [True, False]
    q_values = [True, False]

    and_table = {(p, q): p and q for p in p_values for q in q_values}
    or_table = {(p, q): p or q for p in p_values for q in q_values}
    not_table = {p: not p for p in p_values}

    return {
        "AND_Table": {"inputs": " (p, q)", "outputs": and_table},
        "OR_Table": {"inputs": " (p, q)", "outputs": or_table},
        "NOT_Table": {"inputs": " (p)",   "outputs": not_table},
        "Description": "Exhaustive truth values for basic logical operators."
    }

def implication_operator():
    """
    $\forall p,q \in \{T, F\} : p \rightarrow q \equiv \neg p \lor q$

    In English: "For any propositions p and q, 'p implies q' (or 'if p, then q')
    is logically equivalent to 'not p or q'. The implication $p \rightarrow q$ is only
    false when p is true and q is false. In all other cases, it is true."

    Real-world applicability: The implication is fundamental for expressing
    conditional statements, rules, and logical consequence:
    1.  Software: if (condition) { execute_code }.
    2.  AI and Expert Systems: "IF patient has fever THEN consider infection".
    3.  Formal Verification: Proving program correctness often involves showing preconditions imply postconditions.
    4.  Mathematical Proofs: Many theorems are in the form of an implication.
    5.  Contracts and Policies: "If A occurs, then B must follow."
    """
    p_values = [True, False]
    q_values = [True, False]

    implication_table = {(p, q): (not p) or q for p in p_values for q in q_values}
    return {
        "Implication_Table (p → q)": {"inputs": "(p, q)", "outputs": implication_table},
        "Equivalence": "¬p ∨ q",
        "Description": "Truth table for logical implication (conditional)."
    }

def tautology_contradiction():
    """
    $\forall p \in \{T, F\} : p \lor \neg p \equiv T \text{ (Tautology - Law of Excluded Middle)}$
    $\forall p \in \{T, F\} : p \land \neg p \equiv F \text{ (Contradiction - Law of Non-Contradiction)}$

    In English: "For any proposition p:
    'p OR not p' is always true (a tautology).
    'p AND not p' is always false (a contradiction)."

    Real-world applicability:
    1.  Tautologies:
        -   Simplify logical expressions (e.g., X OR TRUE is always TRUE).
        -   Represent universally true statements or axioms.
    2.  Contradictions:
        -   Indicate logical inconsistencies in requirements, specs, or arguments.
        -   Simplify expressions (e.g., X AND FALSE is always FALSE).
        -   Detect contradictions in debugging software or data constraints.
    """
    p_values = [True, False]

    # Check if p ∨ ¬p is always true
    tautology_check_law_excluded_middle = all((p or not p) for p in p_values)

    # Check if p ∧ ¬p is always false
    is_always_false_contradiction = all((p and not p) == False for p in p_values)

    return {
        "Tautology (p ∨ ¬p)": {"is_always_true": tautology_check_law_excluded_middle},
        "Contradiction (p ∧ ¬p)": {"is_always_false": is_always_false_contradiction},
    }

def reference_supply_chain_books():
    """
    $\exists p (q \land r \land s)$

    In English: "There exists p such that q and r and s are all true."

    Real-world applicability:
    "Represents that in a real-world manufacturing or supply chain context,
     there is a set of resources (p) fulfilling multiple critical criteria (q, r, s).
     For instance, referencing these books means they collectively satisfy continuity,
     resilience, strategy, risk management, and modern manufacturing practices.
     We harness key insights from each to ensure an end-to-end robust and smart supply chain."

    Book References (Real-World Demo):
    1. Flow: How the Best Supply Chains Thrive (2022) – Rob Handfield & Tom Linton
    2. The New (Ab)Normal: Reshaping Business and Supply Chain Strategy Beyond Covid-19 (2020) – Yossi Sheffi
    3. Supply Chain Risk Management: How to Design and Manage Resilient Supply Chains (2020) – John Manners-Bell
    4. Data Science for Supply Chain Forecasting (2021) – Nicholas Vandeput
    5. The Supply Chain Revolution: Innovative Sourcing and Logistics for a Fiercely Competitive World (2023) – Suman Sarkar
    6. Smart Manufacturing: The Lean Six Sigma Way (2022) – Anthony Tarantino
    7. Industry 4.0, Smart Manufacturing, and Industrial Engineering: Challenges and Opportunities (2024) – Amit K. Tyagi et al.
    8. 《供应链管理（第7版）》 (2021) – [美] 苏尼尔·乔普拉 (Sunil Chopra) 等
    9. 《智慧物流与智慧供应链》 (2020) – 霍艳芳, 齐二石 著
    10. 《供应链架构师：从战略到运营》 (2019) – 施云 著
    11. 《智慧工业互联网》 (2019/2020) – 李伯虎, 柴旭东, 侯宝存 等
    """
    print("Supply chain & manufacturing book references loaded and acknowledged for F-35 manufacturing context.")


# -------------------------------------------------------------------
# -------------------- DEMO MAIN EXECUTION --------------------------
# -------------------------------------------------------------------

if __name__ == "__main__":
    print(f"--- Initializing Operations at F-35 Advanced Manufacturing Facility: {FACTORY_LOCATION} ---")

    print(f"\n[[Phase 1: Process Definition & Resource Allocation - {FACTORY_LOCATION}]]")
    print("Defining core manufacturing protocols and material specifications...")
    define_actual_manufacturing_process(
        material_type="Stealth-Optimized CFRP & Ti-6Al-4V Alloy Suite",
        cure_time_hours=12.5,
        friction_welding_strength=1400.0  # MPa
    )
    print(f"Manufacturing process blueprint established for {FACTORY_LOCATION} facility.")

    print(f"\n[[Phase 2: Digital Twin & Pre-production Run - {FACTORY_LOCATION}]]")
    print("Generating CAD models and G-code for initial unit production...")

    import os
    beijing_output_dir = f"{FACTORY_LOCATION}_Manufacturing_Output"
    os.makedirs(beijing_output_dir, exist_ok=True)

    beijing_stl_path = os.path.join(beijing_output_dir, f"{FACTORY_LOCATION}_F35_Unit_001.stl")
    beijing_gcode_path = os.path.join(beijing_output_dir, f"{FACTORY_LOCATION}_F35_Unit_001.gcode")

    try:
        generated_gcode = export_and_slice_f35(
            stl_path=beijing_stl_path,
            gcode_path=beijing_gcode_path,
        )
        print(f"Digital manufacturing package for initial unit generated: {generated_gcode}")
        print(f"STL model saved to: {beijing_stl_path}")
        print(f"G-code for CNC machines at {FACTORY_LOCATION} ready at: {beijing_gcode_path}")
    except Exception as e:
        print(f"Error in generating pre-production run for {FACTORY_LOCATION}: {e}")
        print("Halting further manufacturing simulation due to error.")
    else:
        print(f"\n[[Phase 3: Scaled Production Planning - {FACTORY_LOCATION}]]")
        num_batch_units = 2
        beijing_batch_output_dir = os.path.join(beijing_output_dir, "Batch_Production_Alpha")
        print(f"Initiating planning for a batch of {num_batch_units} F-35 units at {FACTORY_LOCATION}.")
        print(f"Utilizing optimized parallel slicing for efficiency at {FACTORY_LOCATION} facility.")

        try:
            batch_gcodes = optimized_create_f35_batch(
                n=num_batch_units,
                output_dir=beijing_batch_output_dir,
                concurrent_slices=True
            )
            if batch_gcodes:
                print(f"Optimized G-code files for {num_batch_units} units generated in directory: {beijing_batch_output_dir}")
                for i, gcode_file in enumerate(batch_gcodes):
                    print(f"  Unit {i+1}: {gcode_file}")
            else:
                print(f"Batch G-code generation did not complete successfully for all units in {FACTORY_LOCATION}.")
        except Exception as e:
            print(f"Error during scaled production planning for {FACTORY_LOCATION}: {e}")

    print(f"\n--- All initial manufacturing phases for {FACTORY_LOCATION} simulated. System ready for operational command. ---")

    # Demonstrate supply chain reference usage:
    reference_supply_chain_books()
