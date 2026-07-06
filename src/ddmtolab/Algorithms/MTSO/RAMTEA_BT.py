"""
RA-MTEA-BT -- faithful Python port of RAMTEA_BT.m (Blind/unconditional Transfer ablation).

Identical to RAMTEA but knowledge transfer is permanently ON: every generation each task
injects every task's RBF-surrogate-best solution, regardless of similarity (MATLAB
RAMTEA_BT.m removes S and always takes the transfer branch). NOTE wmax = 30 in the MATLAB
source (RAMTEA / NT use 50). See _ramtea_core.py.

Notes
-----
Author: Jiangtao Shen
"""
from ddmtolab.Methods.Algo_Methods.algo_utils import get_algorithm_information
from ddmtolab.Algorithms.MTSO._ramtea_core import ramtea_core


class RAMTEA_BT:
    """RAMTEA with unconditional (blind) transfer (faithful to RAMTEA_BT.m, wmax=30)."""

    algorithm_information = {
        'n_tasks': '[2, K]', 'dims': 'unequal', 'objs': 'equal', 'n_objs': '1',
        'cons': 'equal', 'n_cons': '0', 'expensive': 'True',
        'knowledge_transfer': 'True', 'n_initial': 'unequal', 'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, wmax=30, Nin=50,
                 save_data=True, save_path='./Data', name='RA-MTEA-BT', disable_tqdm=True,
                 **kwargs):
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.wmax = wmax            # MATLAB RAMTEA_BT.m: wmax = 30
        self.Nin = Nin
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        return ramtea_core(
            self.problem, self.n_initial, self.max_nfes, self.wmax, mode='both',
            Nin=self.Nin, save_data=self.save_data, save_path=self.save_path,
            name=self.name, disable_tqdm=self.disable_tqdm,
        )
