"""Task configuration registry for TreasuryAgent environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SMEConfig:
    sme_id: str
    initial_cash: float          # INR
    monthly_revenue_low: float
    monthly_revenue_high: float
    overdraft_limit: float
    overdraft_rate_annual: float  # e.g. 0.18
    udyam_registered: bool = True
    vendor_payment_days: int = 30  # days SME must pay its vendors


@dataclass(frozen=True)
class BuyerConfig:
    buyer_id: str
    payment_days: int            # current contractual days (60-90 typical)
    credit_score: float          # 0-1; affects TReDS rate
    buyer_power: float           # 0-1; affects concession willingness
    annual_order_volume_low: float
    annual_order_volume_high: float
    tax_rate: float = 0.25       # for 43B(h) impact estimate


@dataclass(frozen=True)
class VendorConfig:
    vendor_id: str
    payment_days_required: int   # days vendor expects to be paid
    monthly_supply_low: float
    monthly_supply_high: float
    stress_threshold_days: int = 45  # days overdue before stress triggers


@dataclass(frozen=True)
class TreasuryTaskConfig:
    name: str
    difficulty: Literal["EASY", "MEDIUM", "HARD"]
    description: str
    context_note: str
    max_days: int                # episode horizon in simulation days
    max_steps: int               # max agent tool calls per episode
    smes: tuple[SMEConfig, ...]
    buyers: tuple[BuyerConfig, ...]
    vendors: tuple[VendorConfig, ...]
    # stochastic invoice parameters
    invoices_per_month_low: int
    invoices_per_month_high: int
    invoice_amount_mu: float     # lognormal μ for log(amount)
    invoice_amount_sigma: float
    payment_delay_mu: float      # lognormal μ for log(delay_days)
    payment_delay_sigma: float
    # financing
    treds_available: bool = False
    dd_available: bool = False
    compliance_active: bool = False
    grader_id: str = "treasury_balanced"


# ── EASY ──────────────────────────────────────────────────────────────────────
_EASY = TreasuryTaskConfig(
    name="treasury-easy",
    difficulty="EASY",
    description=(
        "Single SME with one large buyer paying at 90 days. "
        "Keep solvent over 90 days using overdraft and basic ERP queries."
    ),
    context_note=(
        "Overdraft facility is available at 18% pa. "
        "Vendor must be paid within 30 days. No TReDS or compliance risk."
    ),
    max_days=90,
    max_steps=30,
    smes=(
        SMEConfig(
            sme_id="SME_1",
            initial_cash=200_000.0,
            monthly_revenue_low=400_000.0,
            monthly_revenue_high=600_000.0,
            overdraft_limit=300_000.0,
            overdraft_rate_annual=0.18,
            vendor_payment_days=30,
        ),
    ),
    buyers=(
        BuyerConfig(
            buyer_id="BUYER_A",
            payment_days=90,
            credit_score=0.80,
            buyer_power=0.40,
            annual_order_volume_low=4_000_000.0,
            annual_order_volume_high=6_000_000.0,
        ),
    ),
    vendors=(
        VendorConfig(
            vendor_id="VENDOR_X",
            payment_days_required=30,
            monthly_supply_low=200_000.0,
            monthly_supply_high=300_000.0,
        ),
    ),
    invoices_per_month_low=5,
    invoices_per_month_high=10,
    invoice_amount_mu=4.0,
    invoice_amount_sigma=1.0,
    payment_delay_mu=4.4,   # median ~81 days for 90-day buyer
    payment_delay_sigma=0.3,
    treds_available=False,
    dd_available=False,
    compliance_active=False,
    grader_id="treasury_balanced",
)

# ── MEDIUM ─────────────────────────────────────────────────────────────────────
_MEDIUM = TreasuryTaskConfig(
    name="treasury-medium",
    difficulty="MEDIUM",
    description=(
        "Two SMEs each with a distinct large buyer (60-day and 75-day terms). "
        "Use TReDS discounting and dynamic discounting to close the working-capital gap "
        "over a 180-day horizon. Vendors must be paid within 45 days."
    ),
    context_note=(
        "TReDS is available for eligible invoices. "
        "Dynamic discounting can be proposed to buyers. "
        "Minimize combined financing cost across both SMEs."
    ),
    max_days=180,
    max_steps=60,
    smes=(
        SMEConfig(
            sme_id="SME_1",
            initial_cash=300_000.0,
            monthly_revenue_low=500_000.0,
            monthly_revenue_high=800_000.0,
            overdraft_limit=400_000.0,
            overdraft_rate_annual=0.18,
            vendor_payment_days=45,
        ),
        SMEConfig(
            sme_id="SME_2",
            initial_cash=150_000.0,
            monthly_revenue_low=300_000.0,
            monthly_revenue_high=500_000.0,
            overdraft_limit=250_000.0,
            overdraft_rate_annual=0.20,
            vendor_payment_days=30,
        ),
    ),
    buyers=(
        BuyerConfig(
            buyer_id="BUYER_A",
            payment_days=75,
            credit_score=0.85,
            buyer_power=0.50,
            annual_order_volume_low=5_000_000.0,
            annual_order_volume_high=8_000_000.0,
        ),
        BuyerConfig(
            buyer_id="BUYER_B",
            payment_days=60,
            credit_score=0.70,
            buyer_power=0.35,
            annual_order_volume_low=3_000_000.0,
            annual_order_volume_high=5_000_000.0,
        ),
    ),
    vendors=(
        VendorConfig(
            vendor_id="VENDOR_X",
            payment_days_required=45,
            monthly_supply_low=250_000.0,
            monthly_supply_high=400_000.0,
        ),
        VendorConfig(
            vendor_id="VENDOR_Y",
            payment_days_required=30,
            monthly_supply_low=150_000.0,
            monthly_supply_high=250_000.0,
        ),
    ),
    invoices_per_month_low=8,
    invoices_per_month_high=15,
    invoice_amount_mu=4.2,
    invoice_amount_sigma=1.1,
    payment_delay_mu=4.25,  # median ~70 days
    payment_delay_sigma=0.35,
    treds_available=True,
    dd_available=True,
    compliance_active=False,
    grader_id="treasury_balanced",
)

# ── HARD ───────────────────────────────────────────────────────────────────────
_HARD = TreasuryTaskConfig(
    name="treasury-hard",
    difficulty="HARD",
    description=(
        "Three SMEs across a hostile supply chain. Buyers pay at 75-90 days; "
        "vendors demand payment in 30 days. Section 43B(h) compliance breaches are "
        "possible. Use all available tools — TReDS, dynamic discounting, Samadhaan — "
        "to keep all three SMEs solvent over a full 365-day fiscal year. "
        "Minimize financing cost and concentration risk simultaneously."
    ),
    context_note=(
        "Compliance tool triggers buyer penalties but damages relationships. "
        "TReDS rates vary by buyer credit score. "
        "Portfolio concentration risk is measured by buyer HHI index."
    ),
    max_days=365,
    max_steps=120,
    smes=(
        SMEConfig(
            sme_id="SME_1",
            initial_cash=500_000.0,
            monthly_revenue_low=800_000.0,
            monthly_revenue_high=1_200_000.0,
            overdraft_limit=600_000.0,
            overdraft_rate_annual=0.18,
            vendor_payment_days=30,
        ),
        SMEConfig(
            sme_id="SME_2",
            initial_cash=200_000.0,
            monthly_revenue_low=400_000.0,
            monthly_revenue_high=700_000.0,
            overdraft_limit=350_000.0,
            overdraft_rate_annual=0.20,
            vendor_payment_days=30,
        ),
        SMEConfig(
            sme_id="SME_3",
            initial_cash=100_000.0,
            monthly_revenue_low=250_000.0,
            monthly_revenue_high=450_000.0,
            overdraft_limit=200_000.0,
            overdraft_rate_annual=0.22,
            vendor_payment_days=45,
        ),
    ),
    buyers=(
        BuyerConfig(
            buyer_id="BUYER_A",
            payment_days=90,
            credit_score=0.88,
            buyer_power=0.65,
            annual_order_volume_low=8_000_000.0,
            annual_order_volume_high=12_000_000.0,
        ),
        BuyerConfig(
            buyer_id="BUYER_B",
            payment_days=75,
            credit_score=0.72,
            buyer_power=0.50,
            annual_order_volume_low=5_000_000.0,
            annual_order_volume_high=8_000_000.0,
        ),
        BuyerConfig(
            buyer_id="BUYER_C",
            payment_days=60,
            credit_score=0.60,
            buyer_power=0.30,
            annual_order_volume_low=3_000_000.0,
            annual_order_volume_high=5_000_000.0,
        ),
    ),
    vendors=(
        VendorConfig(
            vendor_id="VENDOR_X",
            payment_days_required=30,
            monthly_supply_low=400_000.0,
            monthly_supply_high=600_000.0,
        ),
        VendorConfig(
            vendor_id="VENDOR_Y",
            payment_days_required=30,
            monthly_supply_low=200_000.0,
            monthly_supply_high=350_000.0,
        ),
        VendorConfig(
            vendor_id="VENDOR_Z",
            payment_days_required=45,
            monthly_supply_low=125_000.0,
            monthly_supply_high=225_000.0,
        ),
    ),
    invoices_per_month_low=12,
    invoices_per_month_high=20,
    invoice_amount_mu=4.5,
    invoice_amount_sigma=1.2,
    payment_delay_mu=4.35,  # median ~77 days
    payment_delay_sigma=0.40,
    treds_available=True,
    dd_available=True,
    compliance_active=True,
    grader_id="treasury_balanced",
)

TASK_REGISTRY: dict[str, TreasuryTaskConfig] = {
    "treasury-easy": _EASY,
    "treasury-medium": _MEDIUM,
    "treasury-hard": _HARD,
}

DIFFICULTY_MAP: dict[str, str] = {
    "EASY": "treasury-easy",
    "MEDIUM": "treasury-medium",
    "HARD": "treasury-hard",
}


def resolve_task_id(task_name: str | None, difficulty: str = "MEDIUM") -> str:
    if task_name and task_name in TASK_REGISTRY:
        return task_name
    return DIFFICULTY_MAP.get(difficulty.upper(), "treasury-medium")
