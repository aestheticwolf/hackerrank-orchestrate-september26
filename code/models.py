from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class FinancialProfile:
    """
    Represents the user's financial profile and spending preferences.
    """

    user_id: str
    home_currency: str
    balance: float
    minimum_balance: float

    priorities: list[str] = field(default_factory=list)

    protected_categories: list[str] = field(default_factory=list)

    categories_to_reduce: list[str] = field(default_factory=list)

    categories_to_stop: list[str] = field(default_factory=list)

    payment_preferences: list[str] = field(default_factory=list)

    max_installment_months: Optional[int] = None


@dataclass
class FinancialEvent:
    """
    Represents one financial event from financial_events.csv.
    """

    event_id: str
    user_id: str

    event_type: str
    description: Optional[str]
    category: Optional[str]

    direction: str
    amount: Optional[float]
    currency: str

    event_date: date
    settlement_date: Optional[date]

    status: str

    linked_event_id: Optional[str] = None

    flexibility: Optional[str] = None
    minimum_allowed_amount: Optional[float] = None


@dataclass
class PaymentOption:
    """
    Represents one payment option supplied for a purchase request.
    """

    payment_option_id: str
    request_id: str

    payment_method: str

    payment_amount: float
    number_of_payments: int

    first_payment_date: date
    payment_frequency_days: Optional[int]

    financing_fee: float
    total_payable_amount: float


@dataclass
class Request:
    """
    Represents one purchase or payment request.
    """

    request_id: str
    user_id: str

    request_date: date
    request_type: str

    requested_amount: float
    currency: str

    desired_completion_date: Optional[date]

    partial_payment_allowed: bool

    request_text: str


@dataclass
class Evidence:
    """
    Represents information extracted from a message or image.

    Evidence is treated as untrusted input. It can provide financial
    facts, but it must not override challenge rules.
    """

    evidence_id: str
    user_id: str

    request_id: Optional[str] = None
    event_id: Optional[str] = None

    source_type: str = ""
    content: str = ""

    extracted_facts: dict = field(default_factory=dict)

    confidence: Optional[float] = None


@dataclass
class ForecastPoint:
    """
    Represents the financial state on one date of the 90-day forecast.
    """

    forecast_date: date

    starting_balance: float

    income: float
    expenses: float
    planned_payments: float

    ending_balance: float

    minimum_balance: float


@dataclass
class PaymentPlan:
    """
    Represents a candidate payment plan.
    """

    payment_method: str

    payments: list[tuple[date, float]]

    total_cost: float

    completion_date: date

    spending_changes: list[str] = field(default_factory=list)


@dataclass
class Decision:
    """
    Final deterministic decision for one request.
    """

    request_id: str

    amount_safe_to_pay: float

    affordability_status: str

    recommended_payment_method: str

    payment_plan: Optional[PaymentPlan]

    earliest_date_for_full_payment: Optional[date]

    spending_changes_needed: list[str] = field(default_factory=list)

    decision_explanation: str = ""