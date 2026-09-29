"""Series compliance couples predicted reaction to the measured displacement."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Observation:
    compliance_mm_per_kN: float = 0.0
    seating_offset_mm: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.compliance_mm_per_kN) or self.compliance_mm_per_kN < 0:
            raise ValueError("Series compliance cannot be negative")

    def collar_displacement(self, measured_displacement, predicted_force_N):
        return measured_displacement-self.seating_offset_mm-self.compliance_mm_per_kN*predicted_force_N/1000

    def measured_displacement(self, collar_displacement, predicted_force_N):
        return collar_displacement+self.compliance_mm_per_kN*predicted_force_N/1000+self.seating_offset_mm

    @classmethod
    def for_embedment(cls, reference_compliance, geometry, reference_embedment_mm=500.0,
                      equal_total_length=True):
        extra = (1000*(reference_embedment_mm-geometry.length_mm)/geometry.axial_rigidity_N
                 if equal_total_length else 0.0)
        return cls(reference_compliance+extra)
