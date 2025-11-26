import unittest

import numpy as np

from DLR_rt.examples.d2x1.dlr_2x1d_periodic import integrate
from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr

# ToDo: With large collisions term this probably does not work. After rewriting code,
#       change this as well, such that we only look at adveciton, not collision

class ModelTestCase(unittest.TestCase):
    def setUp(self):
        dt = 1e-3
        r = 8
        t_f = 1.0
        N = 16      # grid size for x, mu and phi

        grid1 = Grid_2x1d(N, N, N, r)
        lr01 = setInitialCondition_2x1d_lr(grid1)
        f01 = lr01.U @ lr01.S @ lr01.V.T
        lr1, time = integrate(lr01, grid1, t_f, dt)
        f1 = lr1.U @ lr1.S @ lr1.V.T

        self.fro_norm1 = np.linalg.norm(f01-f1, 'fro')

    def test_frobenius_norm(self):
        '''
        Test if error in frobenius norm from inital value 
        and end value at t=1 is not huge.
        '''
        self.assertGreater(1.0, self.fro_norm1)


if __name__ == "__main__":
    unittest.main()
