from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class FinancialProfile:
    user_id: str
    balance: float
    minimum_balance: float
    home_currency: str
    priorities: list[str] = field(default_factory=list)
    protected_categories: list[str] = field(default_factory=list)
    flexible_categories: list[str] = field(default_factory=list)
    payment_preferences: list[str] = field(default_factory=list)


@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_date: date
    amount: Optional[float]
    currency: str
    event_type: str
    status: str
    category: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    related_event_id: Optional[str] = None


@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    number_of_payments: int
    installment_amount: Optional[float] = None
    total_amount: Optional[float] = None
    frequency: Optional[str] = None


@dataclass
class Request:
    request_id: str
    user_id: str
    request_date: date
    requested_amount: float
    currency: str
    desired_completion_date: Optional[date] = None
    partial_payment_allowed: bool = False


@dataclass
class Evidence:
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
    forecast_date: date
    starting_balance: float
    income: float
    expenses: float
    planned_payments: float
    ending_balance: float
    minimum_balance: float


@dataclass
class PaymentPlan:
    payment_method: str
    payments: list[tuple[date, float]]
    total_cost: float
    completion_date: date
    spending_changes: list[str] = field(default_factory=list)


@dataclass
class Decision:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: Optional[PaymentPlan]
    earliest_date_for_full_payment: Optional[date]
    spending_changes_needed: list[str] = field(default_factory=list)
    decision_explanation: str = ""