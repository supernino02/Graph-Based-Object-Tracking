import numpy as np
from typing import List,Optional

class KALMANObject:
    #measures matrix
    H: np.ndarray = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype='float64')

    #error measure matrix
    R: np.ndarray = np.array([
        [100, 0  ],
        [0,   100]
    ], dtype='float64')

    #process noise
    Q: np.ndarray = np.array([
        [0.01, 0, 0, 0],
        [0, 0.01, 0, 0],
        [0, 0, 0.01, 0],
        [0, 0, 0, 0.01],
    ], dtype='float64')

    dt = 1 #1fps 
    PHI: np.ndarray = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype='float64')    

    #to assign them more freely
    @classmethod
    def set_kalman_matrices(cls,
        H:   List|np.ndarray|None = None,
        R:   List|np.ndarray|None = None,
        Q:   List|np.ndarray|None = None,
        PHI: List|np.ndarray|None = None
    ) -> None:
        if H is not None:   cls.H =   np.array(H,   dtype='float64')
        if R is not None:   cls.R =   np.array(R,   dtype='float64')
        if Q is not None:   cls.Q =   np.array(Q,   dtype='float64')
        if PHI is not None: cls.PHI = np.array(PHI, dtype='float64')

        n = cls.PHI.shape[0]
        m = cls.H.shape[0]

        if cls.R.shape != (m, m):
            raise ValueError(f"Invalid R shape {cls.R.shape}, expected {(m, m)}")
        if cls.Q.shape != (n, n):
            raise ValueError(f"Invalid Q shape {cls.Q.shape}, expected {(n, n)}")
        if cls.H.shape[1] != n:
            raise ValueError(f"Invalid H shape {cls.H.shape}, expected (*, {n})")
        

    def __init__(self, init_state=None, init_cov=None):
        n = self.PHI.shape[0] if self.PHI is not None else 0

        #nx1
        if init_state is not None:
            self.state = np.array(init_state, dtype='float64').reshape(n, 1)
        else:
            self.state = np.zeros((n, 1), dtype='float64')

        #nxn
        if init_cov is not None:
            self.P = np.array(init_cov, dtype='float64')
        else:
            self.P = np.eye(n, dtype='float64') * 1000

    def update(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        #predict
        pred_state = self.PHI @ self.state
        pred_P = self.PHI @ self.P @ self.PHI.T + self.Q

        measurement = np.array(measurement, dtype='float64').reshape(self.H.shape[0], 1)
        S = self.H @ pred_P @ self.H.T + self.R
        K = pred_P @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ pred_state
        new_state = pred_state + K @ innovation
        new_P = (np.eye(pred_P.shape[0]) - K @ self.H) @ pred_P
        return new_state, new_P
    
    @staticmethod
    def merge_states(
        estimates: List[tuple[np.ndarray, np.ndarray]],
        weights: List[float]
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if not estimates or not weights or len(estimates) != len(weights):
            raise ValueError("Estimates and weights must be non-empty and of equal length")

        all_cov_none = all(P is None for (_, P) in estimates)

        # Accumulators
        weighted_state_sum = np.zeros_like(estimates[0][0])
        weighted_cov_sum = None if all_cov_none else np.zeros_like(estimates[0][1])
        total_weight = 0.0

        for (x, P), w in zip(estimates, weights):
            weighted_state_sum += w * x
            if not all_cov_none:
                weighted_cov_sum += w * P # type: ignore
            total_weight += w

        if total_weight <= 0:
            raise ValueError("Sum of weights must be positive")

        x_fused = weighted_state_sum / total_weight
        P_fused = None if all_cov_none else weighted_cov_sum / total_weight # type: ignore

        return x_fused, P_fused