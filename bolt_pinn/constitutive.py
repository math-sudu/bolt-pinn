"""Tangential slip-weakening return maps; units: MPa and mm.

The additive slip/state organization follows Jahangir et al. (2021).
The linear and exponential laws are project reductions, not reproductions
of that paper's pressure/dilatancy-dependent law.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Interface:
    stiffness: float
    peak: float
    residual: float
    critical_slip: float
    weakening: str = "linear"
    contact_length: float | None = None
    mobilization_slip: float | None = None
    yield_fraction: float = 1.0
    softening_onset: float = 0.0

    def __post_init__(self):
        if self.weakening not in {"linear", "exponential"}:
            raise ValueError(f"Unknown weakening law: {self.weakening}")
        if not all(math.isfinite(v) for v in (self.stiffness,self.peak,self.residual,self.critical_slip)):
            raise ValueError("Interface parameters must be finite")
        if self.critical_slip <= 0 or not 0 <= self.residual <= self.peak:
            raise ValueError("Require critical_slip > 0 and peak >= residual >= 0")
        if self.contact_length is not None:
            if not math.isfinite(self.contact_length) or self.contact_length <= 0:
                raise ValueError("Contact length must be finite and positive")
            if self.weakening != "exponential":
                raise ValueError("Contact evolution is defined for exponential weakening")
        if self.mobilization_slip is not None:
            if not math.isfinite(self.mobilization_slip) or self.mobilization_slip <= 0:
                raise ValueError("Mobilization slip must be finite and positive")
            if self.weakening != "exponential" or not 0 < self.yield_fraction <= 1:
                raise ValueError("Mobilization requires exponential weakening and 0 < yield_fraction <= 1")
        if not math.isfinite(self.softening_onset) or self.softening_onset < 0:
            raise ValueError("Softening onset must be finite and nonnegative")
        if self.softening_onset and self.weakening != "exponential":
            raise ValueError("Delayed softening requires exponential weakening")
        if self.stiffness <= self.softening_modulus:
            raise ValueError("Single-valued local return requires stiffness > softening modulus")

    @property
    def softening_modulus(self):
        return ((self.peak-self.residual)/self.critical_slip
                + (0 if self.contact_length is None else self.peak/self.contact_length))

    def strength(self, kappa):
        remaining = (torch.exp(-torch.clamp(kappa-self.softening_onset, min=0)/self.critical_slip)
                     if self.weakening == "exponential"
                     else torch.clamp(1-kappa/self.critical_slip,min=0))
        strength = self.residual+(self.peak-self.residual)*remaining
        if self.mobilization_slip is not None:
            strength = strength*(1-(1-self.yield_fraction)*torch.exp(-kappa/self.mobilization_slip))
        if self.contact_length is not None:
            strength = strength*torch.clamp(1-kappa/self.contact_length, min=0)
        return strength

    def exponential_strength_derivative(self, kappa):
        decay = torch.exp(-torch.clamp(kappa-self.softening_onset, min=0)/self.critical_slip)
        base = self.residual+(self.peak-self.residual)*decay
        base_derivative = -(self.peak-self.residual)*decay/self.critical_slip
        base_derivative = torch.where(kappa >= self.softening_onset, base_derivative,
                                      torch.zeros_like(kappa))
        value, derivative = base, base_derivative
        if self.mobilization_slip is not None:
            missing = (1-self.yield_fraction)*torch.exp(-kappa/self.mobilization_slip)
            value = base*(1-missing)
            derivative = base_derivative*(1-missing)+base*missing/self.mobilization_slip
        if self.contact_length is not None:
            overlap = torch.clamp(1-kappa/self.contact_length, min=0)
            derivative = overlap*derivative-value/self.contact_length
            derivative = torch.where(kappa < self.contact_length, derivative, torch.zeros_like(kappa))
        return derivative

    def delayed_work(self, k0, increment):
        """Exact resistance integral, split at the onset and contact exit.

        An optional mobilization factor is integrated on both sides of onset;
        it changes the resistance, not the accumulated-slip state definition.
        """
        start, end = k0, k0+increment
        if self.contact_length is not None:
            start = torch.clamp(start, max=self.contact_length)
            end = torch.clamp(end, max=self.contact_length)

        def area(a, b):
            span = b-a
            return span if self.contact_length is None else span*(1-(a+b)/(2*self.contact_length))

        def moment(a, b, width, origin=0.0):
            span = b-a
            if self.contact_length is None:
                return width*torch.exp(-(a-origin)/width)*(-torch.expm1(-span/width))
            length = self.contact_length
            return width*torch.exp(-(a-origin)/width)*(
                (1-a/length-width/length)*(-torch.expm1(-span/width))
                +span/length*torch.exp(-span/width))

        before_a = torch.clamp(start, max=self.softening_onset)
        before_b = torch.clamp(end, max=self.softening_onset)
        after_a = torch.clamp(start, min=self.softening_onset)
        after_b = torch.clamp(end, min=self.softening_onset)
        before = area(before_a, before_b)
        after = area(after_a, after_b)
        decay = moment(after_a, after_b, self.critical_slip, self.softening_onset)
        if self.mobilization_slip is not None:
            missing = 1-self.yield_fraction
            before = before-missing*moment(before_a, before_b, self.mobilization_slip)
            after = after-missing*moment(after_a, after_b, self.mobilization_slip)
            joint_width = 1/(1/self.critical_slip+1/self.mobilization_slip)
            decay = decay-missing*math.exp(-self.softening_onset/self.mobilization_slip)*moment(
                after_a, after_b, joint_width, self.softening_onset)
        return self.peak*before+self.residual*after+(self.peak-self.residual)*decay

    def mobilized_work(self, k0, increment):
        """Integrate the mobilized resistance as three exponential moments."""
        start, end = k0, k0+increment
        if self.contact_length is not None:
            start = torch.clamp(start, max=self.contact_length)
            end = torch.clamp(end, max=self.contact_length)
        span = end-start

        def exponential_area(width):
            if self.contact_length is None:
                return width*torch.exp(-start/width)*(-torch.expm1(-span/width))
            length = self.contact_length
            return width*torch.exp(-start/width)*(
                (1-start/length-width/length)*(-torch.expm1(-span/width))
                +span/length*torch.exp(-span/width))

        area = span if self.contact_length is None else span*(1-(start+end)/(2*self.contact_length))
        joint_width = 1/(1/self.critical_slip+1/self.mobilization_slip)
        missing = 1-self.yield_fraction
        return (self.residual*(area-missing*exponential_area(self.mobilization_slip))
                +(self.peak-self.residual)*(exponential_area(self.critical_slip)
                                           -missing*exponential_area(joint_width)))

    def contact_strength_derivative(self, kappa):
        overlap = torch.clamp(1-kappa/self.contact_length, min=0)
        decay = torch.exp(-kappa/self.critical_slip)
        base = self.residual+(self.peak-self.residual)*decay
        derivative = (-base/self.contact_length
                      - overlap*(self.peak-self.residual)*decay/self.critical_slip)
        return torch.where(kappa < self.contact_length, derivative, torch.zeros_like(kappa))

    def contact_work(self, k0, increment):
        """Integral of the nonnegative overlap-weighted resistance, to exit."""
        start = torch.clamp(k0, max=self.contact_length)
        end = torch.clamp(k0+increment, max=self.contact_length)
        width, length = self.critical_slip, self.contact_length
        # Integrate from start to end in a cancellation-resistant difference form.
        span = end-start
        exponential_area = width*torch.exp(-start/width)*(
            (1-start/length-width/length)*(-torch.expm1(-span/width))
            +span/length*torch.exp(-span/width))
        residual_area = span*(1-(start+end)/(2*length))
        return self.residual*residual_area+(self.peak-self.residual)*exponential_area


def contact_increment(q, k0, material):
    """Unique return for the homogenized diminishing-overlap law.

    K exceeds a global bound on -dr/dk. The resistance is decreasing and
    convex before contact exit, so the initial tangent gives a Newton upper
    bound, including increments that cross the zero-contact corner.
    """
    excess = torch.clamp(q-material.strength(k0), min=0)
    dl = excess/(material.stiffness+material.contact_strength_derivative(k0))
    tolerance = 16*torch.finfo(q.dtype).eps*(1+q/material.stiffness)
    for _ in range(40):
        residual = material.stiffness*dl+material.strength(k0+dl)-material.strength(k0)-excess
        update = residual/(material.stiffness+material.contact_strength_derivative(k0+dl))
        dl = torch.clamp(dl-update, min=0)
        if bool(torch.all(torch.abs(update.detach()) <= tolerance.detach())):
            return dl
    raise RuntimeError("Contact local return did not converge")


def bracketed_increment(q, k0, material):
    """Safeguarded Newton return for mobilization or delayed softening.

    The increasing mobilization factor cannot make the negative tangent
    larger than the original softening bound, hence K+dr/dk remains positive.
    """
    old_strength = material.strength(k0)
    excess = torch.clamp(q-old_strength, min=0)
    lower = torch.zeros_like(q)
    upper = excess/(material.stiffness-material.softening_modulus)
    dl = excess/material.stiffness
    tolerance = 16*torch.finfo(q.dtype).eps*(1+q/material.stiffness)
    for _ in range(60):
        residual = material.stiffness*dl+material.strength(k0+dl)-old_strength-excess
        tangent = material.stiffness+material.exponential_strength_derivative(k0+dl)
        update = residual/tangent
        if bool(torch.all(torch.abs(update.detach()) <= tolerance.detach())):
            # One final Newton expression preserves the implicit tangent even
            # when the bracket's most recent step was a bisection.
            return torch.clamp(dl-update, min=0)
        lower = torch.where(residual < 0, dl, lower)
        upper = torch.where(residual > 0, dl, upper)
        candidate = dl-update
        outside = (candidate < lower) | (candidate > upper)
        dl = torch.where(outside, (lower+upper)/2, candidate)
    raise RuntimeError("Bracketed local return did not converge")


def exponential_increment(q, k0, material):
    """Solve K*dl + r(k0+dl) = q on the active branch.

    K > (peak-residual)/critical_slip makes the scalar equation strictly
    increasing. Newton starts at the tangent upper bound and decreases to
    its unique root. Autodiff follows the converged return, including dr/dk.
    """
    excess = torch.clamp(q-material.strength(k0), min=0)
    h0 = material.softening_modulus*torch.exp(-k0/material.critical_slip)
    dl = excess/(material.stiffness-h0)
    tolerance = 16*torch.finfo(q.dtype).eps*(1+q/material.stiffness)
    for _ in range(40):
        decay = torch.exp(-(k0+dl)/material.critical_slip)
        # This difference form also returns exactly zero on an elastic trial.
        residual = material.stiffness*dl + (material.peak-material.residual)*(
            decay-torch.exp(-k0/material.critical_slip))-excess
        update = residual/(material.stiffness-material.softening_modulus*decay)
        dl = torch.clamp(dl-update, min=0)
        if bool(torch.all(torch.abs(update.detach()) <= tolerance.detach())):
            return dl
    raise RuntimeError("Exponential local return did not converge")


@dataclass
class State:
    plastic_slip: torch.Tensor
    accumulated_slip: torch.Tensor

    @classmethod
    def zero(cls, like):
        return cls(torch.zeros_like(like),torch.zeros_like(like))

    def committed(self):
        return State(self.plastic_slip.detach().clone(),self.accumulated_slip.detach().clone())


def return_map(slip, committed: State, material: Interface):
    """Pure trial evaluation; never mutate the supplied committed state."""
    trial=material.stiffness*(slip-committed.plastic_slip)
    q=torch.abs(trial)
    k0=committed.accumulated_slip
    excess=torch.clamp(q-material.strength(k0),min=0)
    if material.mobilization_slip is not None or material.softening_onset > 0:
        dl=bracketed_increment(q,k0,material)
    elif material.contact_length is not None:
        dl=contact_increment(q,k0,material)
    elif material.weakening == "exponential":
        dl=exponential_increment(q,k0,material)
    else:
        dl_soft=excess/(material.stiffness-material.softening_modulus)
        dl_res=torch.clamp((q-material.residual)/material.stiffness,min=0)
        soft=(k0 < material.critical_slip)&(dl_soft <= material.critical_slip-k0)
        dl=torch.where(excess>0,torch.where(soft,dl_soft,dl_res),torch.zeros_like(excess))
    z1=committed.plastic_slip+dl*torch.sign(trial)
    k1=k0+dl
    tau=material.stiffness*(slip-z1)
    # Integrate r(kappa) over the plastic increment, including crossing the residual branch.
    def primitive(k):
        c=torch.clamp(k,max=material.critical_slip)
        return c-c*c/(2*material.critical_slip)
    if material.softening_onset > 0:
        dissipation=material.delayed_work(k0,dl)
        return tau,State(z1,k1),dissipation
    if material.mobilization_slip is not None:
        dissipation=material.mobilized_work(k0,dl)
        return tau,State(z1,k1),dissipation
    if material.contact_length is not None:
        dissipation=material.contact_work(k0,dl)
        return tau,State(z1,k1),dissipation
    if material.weakening == "exponential":
        weakening_work=material.critical_slip*torch.exp(-k0/material.critical_slip)*(
            -torch.expm1(-dl/material.critical_slip))
    else:
        weakening_work=primitive(k1)-primitive(k0)
    dissipation=material.residual*dl+(material.peak-material.residual)*weakening_work
    return tau,State(z1,k1),dissipation
