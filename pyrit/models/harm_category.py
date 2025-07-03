# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Tuple

import yaml

with open(os.path.join(os.path.dirname(__file__), "harm_category_definitions.yaml")) as f:
    _STATIC_HARM_DEFINITIONS = yaml.safe_load(f).get("definitions", {})


class HarmCategory(StrEnum):
    VERSION = "v1.0.0"

    HATESPEECH = "Hate Speech"
    HARASSMENT = "Harassment"
    VIOLENT_CONTENT = "Graphic Violence and Gore"
    SEXUAL_CONTENT = "Pornography & Sexual Content"
    PROFANITY = "Profanity"
    QUALITY_OF_SERVICE = "Quality of Service (QoS)"
    ALLOCATION = "Allocation of Resources & Opportunities"
    REPRESENTATIONAL = "Representational Harms (Stereotyping, Demeaning & Erasing Outputs)"
    SUICIDE = "Suicide"
    SELF_HARM = "Self-Harm"
    EATING_DISORDERS = "Eating Disorders"
    DRUG_USE = "Drug Use"
    VIOLENT_THREATS = "Violent Threats, Incitement & Glorification"
    VIOLENT_EXTREMISM = "Terrorism & Violent Extremism"
    COORDINATION_HARM = "Coordination of Harm"
    REGULATED_GOODS = "Sale of Regulated Goods"
    SEXUAL_SOLICITATION = "Sexual Solicitation"
    SCAMS = "Scams"
    SPAM = "Spam"
    IMPERSONATION = "Impersonation"
    FAKE_ACCOUNTS = "Fake Accounts"
    INFO_INTEGRITY = "Inaccurate, Incomplete, False & Misleading Information"
    CURRENT_EVENTS_MISINFO = "Misinformation for current events"
    HISTORICAL_EVENTS_BIAS = "Biased or revisionist retelling of controversial historical events"
    ELECTION_INTEGRITY = "Inaccurate, Incomplete, False & Misleading Info in Election Context"
    DECEPTION = "Deceptive Inducement"
    COVERT_TARGETED = "Covert Targeted Persuasion"
    REPUTATIONAL_DAMAGE = "Reputational Damage"
    COPYRIGHT = "Copyright & Piracy"
    TRADEMARK = "Trademark"
    IP_UPLOAD = "Upload IP Images"
    PLAGIARISM = "Plagiarism & Academic Dishonesty"
    PROPRIETARY_INFO = "Proprietary, Confidential & Classified Information"
    PPI = "Private Personal Information"
    PUBLIC_FIGURES = "Images of Public Figures"
    NONCONSENSUAL_UPLOAD = "Upload Images of People without Consent"
    INSECURE_CODE = "Insecure Code"
    MALWARE = "Malware"
    MILITARY = "Weapons Development & Military"
    CBRN = "Chemical, Biological, Radiological, and Nuclear"
    HIGH_RISK_GOVERNMENT = "High-Risk Government Decision-Making"
    INFRASTRUCTURE_RISK = "Management or Operation of Critical Infrastructure in Energy, Transportation & Water"
    FINANCIAL_ADVICE = "Financial Advice"
    MLM = "Multi-Level Marketing"
    GAMBLING = "Gambling"
    LENDING = "Lending"
    FINANCIAL_ELIGIBILITY = "Financial Service Eligibility"
    HEALTH_DIAGNOSIS = "Health Diagnosis"
    PSEUDO_PHARMA = "Pseudo-Pharmaceuticals"
    PUBLIC_HEALTH = "Public & Personal Health"
    CAMPAIGNING = "Political Campaigning & Lobbying"
    LEGAL_ADVICE = "Legal Advice"
    ROMANTIC = "Romantic"
    SELF_VALIDATION = "Self-Validation"
    MENTAL_HEALTH = "Mental Health"
    EMOTIONAL = "Emotional"
    PROTECTED_INFERENCE = "Legally-Protected Attributes"
    EMOTION_INFERENCE = "Emotion"
    ILLEGAL = "Illegal Activity"
    OTHER = "Other"

    _ALIASES = {  # TODO: Add the rest of the aliases
        "violent": VIOLENT_CONTENT,
        "bullying": HARASSMENT,
        "illegal": ILLEGAL,
    }  # type: ignore

    _DEFINITIONS = _STATIC_HARM_DEFINITIONS

    @classmethod
    def parse(cls, value: str) -> "HarmCategory":
        value = value.strip().lower()

        for member in cls:
            if str(member.value).lower() == value:
                return member

        if value in cls._ALIASES:
            return cls._ALIASES[value]  # type: ignore

        return cls.OTHER

    @classmethod
    def get_definition(cls, category: "HarmCategory") -> str:
        return _STATIC_HARM_DEFINITIONS.get(category.name, "No definition available.")


@dataclass(frozen=True)
class SeedPrompt:
    text: str
    harm_categories: Tuple[HarmCategory, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "harm_categories", self._parse_categories(self.harm_categories))

    @staticmethod
    def _parse_categories(raw):
        if isinstance(raw, str):
            raw = [raw]
        return tuple(c if isinstance(c, HarmCategory) else HarmCategory.parse(c) for c in raw)
