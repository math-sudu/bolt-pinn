import unittest
import torch
from bolt_pinn.constitutive import Interface,State,return_map
from bolt_pinn.pinn_fields import Geometry,MixedFields
from bolt_pinn.observation_operator import Observation


class MechanicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.material=Interface(10,2,0.4,1)

    def test_elastic_softening_residual_and_complementarity(self):
        state=State.zero(torch.zeros(1))
        for slip in [0,0.1,0.2,0.4,0.8,1.04,1.3,2.0]:
            tau,trial,diss=return_map(torch.tensor([slip]),state,self.material)
            dl=trial.accumulated_slip-state.accumulated_slip
            f=tau.abs()-self.material.strength(trial.accumulated_slip)
            self.assertLessEqual(float(f.max()),1e-12)
            self.assertLess(float(torch.abs(dl*f).max()),1e-12)
            self.assertGreaterEqual(float(diss.min()),-1e-12)
            self.assertAlmostEqual(float((trial.plastic_slip-state.plastic_slip).abs()),float(dl),12)
            state=trial.committed()
        self.assertAlmostEqual(float(tau),0.4,12)

    def test_trial_does_not_mutate_committed_history(self):
        state=State.zero(torch.zeros(1))
        first=return_map(torch.tensor([0.5]),state,self.material)
        return_map(torch.tensor([0.8]),state,self.material)
        again=return_map(torch.tensor([0.5]),state,self.material)
        self.assertTrue(torch.equal(state.accumulated_slip,torch.zeros(1)))
        self.assertTrue(torch.equal(first[1].plastic_slip,again[1].plastic_slip))

    def test_unloading_and_reverse_sliding(self):
        tau,loaded,_=return_map(torch.tensor([0.5]),State.zero(torch.zeros(1)),self.material)
        unloading=return_map(torch.tensor([0.49]),loaded.committed(),self.material)
        self.assertTrue(torch.equal(unloading[1].plastic_slip,loaded.plastic_slip))
        self.assertAlmostEqual(float(unloading[0]),float(tau)-0.1,12)
        reverse=return_map(torch.tensor([-0.5]),loaded.committed(),self.material)
        self.assertLess(float(reverse[0]),0)
        self.assertGreater(float(reverse[1].accumulated_slip),float(loaded.accumulated_slip))
        self.assertGreater(float(reverse[2]),0)

    def test_zero_residual_keeps_sliding(self):
        material=Interface(10,2,0,1)
        state=State.zero(torch.zeros(1))
        for slip in [1.5,2.5]:
            tau,state,diss=return_map(torch.tensor([slip]),state,material)
            self.assertAlmostEqual(float(tau),0,12)
            self.assertAlmostEqual(float(state.plastic_slip),slip,12)
        self.assertAlmostEqual(float(state.accumulated_slip),2.5,12)

    def test_return_gradient_on_active_softening_branch(self):
        s=torch.tensor([0.5],requires_grad=True)
        tau,_,_=return_map(s,State.zero(s),self.material)
        grad=torch.autograd.grad(tau,s)[0]
        h=self.material.softening_modulus
        self.assertAlmostEqual(float(grad),-10*h/(10-h),12)

    def test_exact_plastic_work_across_softening_corner(self):
        _,_,d=return_map(torch.tensor([2.0]),State.zero(torch.zeros(1)),self.material)
        # kappa = 2 - tau_r/K = 1.96; integral r dk is a trapezoid plus a rectangle.
        self.assertAlmostEqual(float(d),(2+0.4)/2+0.96*0.4,12)

    def test_reject_nonunique_local_return(self):
        with self.assertRaises(ValueError):
            Interface(1,2,0,1)

    def test_implicit_boundary_uses_predicted_reaction(self):
        obs=Observation(0.02)
        model=MixedFields(Geometry(),obs).double()
        x=torch.tensor([[0.0],[1.0]])
        u,n=model(x,8.0)
        self.assertEqual(float(n[0].detach()),0)
        self.assertAlmostEqual(float(obs.measured_displacement(u[-1],n[-1]).detach()),8.0,12)
        self.assertAlmostEqual(float(obs.collar_displacement(8.0,100000.0)),6.0,12)

    def test_exponential_return_and_consistent_tangent(self):
        material = Interface(10, 2, 0.4, 1, "exponential")
        state = State.zero(torch.zeros(1))
        for slip in [0.0, 0.1, 0.3, 0.8, 3.0, 100.0]:
            s = torch.tensor([slip], requires_grad=True)
            tau, trial, work = return_map(s, state, material)
            yield_value = tau.abs()-material.strength(trial.accumulated_slip)
            increment = trial.accumulated_slip-state.accumulated_slip
            self.assertLessEqual(float(yield_value.detach()), 1e-12)
            self.assertLess(abs(float((increment*yield_value).detach())), 1e-10)
            self.assertGreaterEqual(float(work.detach()), 0.0)
            if slip == 0.8:
                gradient = torch.autograd.grad(tau, s)[0]
                high = return_map(torch.tensor([slip+1e-5]), state, material)[0]
                low = return_map(torch.tensor([slip-1e-5]), state, material)[0]
                self.assertAlmostEqual(float(gradient), float((high-low)/2e-5), 8)
            state = trial.committed()

    def test_exponential_path_work_and_trial_purity(self):
        material = Interface(10, 2, 0.4, 1, "exponential")
        virgin = State.zero(torch.zeros(1))
        direct = return_map(torch.tensor([3.0]), virgin, material)
        state, total = virgin, 0.0
        for slip in [0.1, 0.3, 0.8, 1.5, 3.0]:
            _, trial, work = return_map(torch.tensor([slip]), state, material)
            state = trial.committed()
            total += float(work)
        self.assertAlmostEqual(float(state.accumulated_slip), float(direct[1].accumulated_slip), 12)
        self.assertAlmostEqual(total, float(direct[2]), 12)
        self.assertEqual(float(virgin.accumulated_slip), 0.0)
        unloaded = return_map(torch.tensor([2.99]), state, material)
        self.assertEqual(float(unloaded[1].accumulated_slip), float(state.accumulated_slip))
        self.assertAlmostEqual(float(unloaded[0]), float(direct[0])-0.1, 12)

    def test_length_compliance_tracks_added_free_rod_extension(self):
        long = Geometry(length_mm=500)
        short = Geometry(length_mm=100)
        obs_long = Observation.for_embedment(0.01, long)
        obs_short = Observation.for_embedment(0.01, short)
        extension = obs_short.measured_displacement(0, 20000)-obs_long.measured_displacement(0, 20000)
        self.assertAlmostEqual(extension, 20000*400/long.axial_rigidity_N, 12)
        self.assertEqual(Observation.for_embedment(0.01, short, equal_total_length=False).compliance_mm_per_kN, 0.01)

    def test_contact_exit_work_and_consistent_tangent(self):
        material = Interface(10, 2, 0.4, 1, "exponential", contact_length=3)
        virgin = State.zero(torch.zeros(1))
        state, total = virgin, 0.0
        for slip in [0.1, 0.3, 0.8, 2.0, 3.1, 4.0]:
            s = torch.tensor([slip], requires_grad=True)
            tau, trial, work = return_map(s, state, material)
            dl = trial.accumulated_slip-state.accumulated_slip
            yield_value = tau.abs()-material.strength(trial.accumulated_slip)
            self.assertLessEqual(float(yield_value.detach()), 1e-12)
            self.assertLess(abs(float((dl*yield_value).detach())), 1e-11)
            self.assertGreaterEqual(float(work.detach()), -1e-14)
            if slip == 0.8:
                gradient = torch.autograd.grad(tau, s)[0]
                high = return_map(torch.tensor([slip+1e-5]), state, material)[0]
                low = return_map(torch.tensor([slip-1e-5]), state, material)[0]
                self.assertAlmostEqual(float(gradient), float((high-low)/2e-5), 8)
            total += float(work.detach())
            state = trial.committed()
        direct = return_map(torch.tensor([4.0]), virgin, material)
        grid = torch.linspace(0, 3, 10001)
        quadrature = torch.trapezoid(material.strength(grid), grid)
        self.assertAlmostEqual(total, float(direct[2]), 12)
        self.assertAlmostEqual(total, float(quadrature), 6)
        self.assertAlmostEqual(float(tau.detach()), 0.0, 12)
        self.assertEqual(float(virgin.accumulated_slip), 0.0)

    def test_contact_loss_included_in_unique_return_bound(self):
        with self.assertRaises(ValueError):
            Interface(2, 2, 0.4, 1, "exponential", contact_length=3)

    def test_mobilized_return_hardening_softening_and_work(self):
        for contact_length in [None, 3.0]:
            material = Interface(10, 2, 0.4, 2, "exponential", contact_length, 0.2, 0.25)
            virgin = State.zero(torch.zeros(1))
            state, total = virgin, 0.0
            slopes = []
            for slip in [0.03, 0.1, 0.2, 0.4, 0.8, 2.0, 3.5]:
                s = torch.tensor([slip], requires_grad=True)
                tau, trial, work = return_map(s, state, material)
                dl = trial.accumulated_slip-state.accumulated_slip
                yield_value = tau.abs()-material.strength(trial.accumulated_slip)
                self.assertLessEqual(float(yield_value.detach()), 1e-12)
                self.assertLess(abs(float((dl*yield_value).detach())), 1e-11)
                self.assertGreaterEqual(float(work.detach()), -1e-14)
                gradient = torch.autograd.grad(tau, s)[0]
                high = return_map(torch.tensor([slip+1e-5]), state, material)[0]
                low = return_map(torch.tensor([slip-1e-5]), state, material)[0]
                self.assertAlmostEqual(float(gradient), float((high-low)/2e-5), 6)
                slopes.append(float(gradient))
                total += float(work.detach())
                state = trial.committed()
            direct = return_map(torch.tensor([3.5]), virgin, material)
            self.assertAlmostEqual(total, float(direct[2]), 12)
            grid = torch.linspace(0, float(state.accumulated_slip), 20001)
            self.assertAlmostEqual(total, float(torch.trapezoid(material.strength(grid), grid)), 6)
            self.assertGreater(slopes[2], 0)
            self.assertLess(slopes[4], 0)
            self.assertEqual(float(virgin.accumulated_slip), 0.0)

    def test_delayed_softening_return_work_and_unloading(self):
        for contact_length, mobilization in [(None, None), (3.0, None), (3.0, 0.2)]:
            material = Interface(10, 2, 0.4, 1, "exponential", contact_length,
                                 mobilization, 0.25 if mobilization else 1.0, 0.7)
            virgin = State.zero(torch.zeros(1))
            state, total = virgin, 0.0
            for slip in [0.03, 0.4, 0.85, 1.2, 2.0, 3.5]:
                s = torch.tensor([slip], requires_grad=True)
                tau, trial, work = return_map(s, state, material)
                dl = trial.accumulated_slip-state.accumulated_slip
                yield_value = tau.abs()-material.strength(trial.accumulated_slip)
                self.assertLessEqual(float(yield_value.detach()), 1e-12)
                self.assertLess(abs(float((dl*yield_value).detach())), 1e-11)
                self.assertGreaterEqual(float(work.detach()), -1e-14)
                gradient = torch.autograd.grad(tau, s)[0]
                high = return_map(torch.tensor([slip+1e-5]), state, material)[0]
                low = return_map(torch.tensor([slip-1e-5]), state, material)[0]
                self.assertAlmostEqual(float(gradient), float((high-low)/2e-5), 6)
                if slip == 0.4:
                    unloaded = return_map(torch.tensor([slip-0.01]), trial.committed(), material)
                    self.assertTrue(torch.equal(unloaded[1].accumulated_slip, trial.accumulated_slip))
                    self.assertAlmostEqual(float(unloaded[0]), float(tau.detach())-0.1, 12)
                    if contact_length is None:
                        self.assertAlmostEqual(float(tau.detach()), 2.0, 12)
                        self.assertAlmostEqual(float(gradient), 0.0, 12)
                total += float(work.detach())
                state = trial.committed()
            direct = return_map(torch.tensor([3.5]), virgin, material)
            self.assertAlmostEqual(total, float(direct[2]), 12)
            # Integrate the defined strength independently across the onset corner.
            grid = torch.linspace(0, float(state.accumulated_slip), 20001)
            self.assertAlmostEqual(total, float(torch.trapezoid(material.strength(grid), grid)), 6)
            self.assertEqual(float(virgin.accumulated_slip), 0.0)

    def test_delayed_onset_after_contact_exit(self):
        material = Interface(10, 2, 0.4, 1, "exponential", 3.0, softening_onset=5.0)
        tau, _, work = return_map(torch.tensor([6.0]), State.zero(torch.zeros(1)), material)
        self.assertAlmostEqual(float(tau), 0.0, 12)
        self.assertAlmostEqual(float(work), 3.0, 12)


if __name__=="__main__":
    unittest.main()
