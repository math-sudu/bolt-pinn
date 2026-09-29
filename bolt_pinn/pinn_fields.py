"""Separate displacement and force fields following RE22 Table 2 Model D."""
import math
from dataclasses import dataclass
import torch
from torch import nn
from .observation_operator import Observation


@dataclass(frozen=True)
class Geometry:
    length_mm: float = 500.0
    diameter_mm: float = 25.0
    youngs_modulus_MPa: float = 200000.0

    @property
    def axial_rigidity_N(self):
        return self.youngs_modulus_MPa*math.pi*self.diameter_mm**2/4

    @property
    def perimeter_mm(self):
        return math.pi*self.diameter_mm


def mlp(width):
    model=nn.Sequential(nn.Linear(1,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
    for layer in model:
        if isinstance(layer,nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model


class MixedFields(nn.Module):
    force_scale_N=100000.0

    def __init__(self,geometry: Geometry,observation: Observation,width=16):
        super().__init__()
        self.geometry=geometry
        self.observation=observation
        self.u_net=mlp(width)
        self.n_net=mlp(width)

    @property
    def axial_displacement_scale_mm(self):
        return self.force_scale_N*self.geometry.length_mm/self.geometry.axial_rigidity_N

    def forward(self,xi,measured_displacement):
        end_force=self.force_scale_N*self.n_net(torch.ones((1,1),dtype=xi.dtype,device=xi.device))
        collar=self.observation.collar_displacement(measured_displacement,end_force)
        u=collar+self.axial_displacement_scale_mm*(xi-1)*self.u_net(xi)
        force=self.force_scale_N*xi*self.n_net(xi)
        return u,force
