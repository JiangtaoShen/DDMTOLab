from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
from ddmtolab.Algorithms.STSO.BO import BO
from ddmtolab.Algorithms.MTSO.MTBO import MTBO
from ddmtolab.Methods.animation_generator import create_optimization_animation

problem = CEC17MTSO_10D().P1()
BO(problem, n_initial=20, max_nfes=100, name='BO').optimize()
MTBO(problem, n_initial=20, max_nfes=100, name='MTBO').optimize()
animation = create_optimization_animation(max_nfes=100, merge=2, title='BO and MTBO on CEC17MTSO-10D-P1')


