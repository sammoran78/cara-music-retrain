from __future__ import annotations

from validation.validator import CARAValidator, ValidationResult


def repair_attr_string(attr_string: str, validator: CARAValidator) -> ValidationResult:
    return validator.validate(attr_string)
