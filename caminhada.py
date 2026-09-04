"""Operacoes compartilhadas da caminhada quantica confinada."""

import numpy as np


def estado_inicial(N):
    """Retorna o estado inicial localizado no sitio central."""
    origin = (N - 1) // 2
    psiL = np.zeros(N, dtype=complex)
    psiR = np.zeros(N, dtype=complex)
    psiL[origin] = 1.0 / np.sqrt(2.0)
    psiR[origin] = 1.0j / np.sqrt(2.0)
    return psiL, psiR


def passo(psiL, psiR):
    """Executa a coin e o shift de um passo da caminhada."""
    N = len(psiL)
    cL = np.zeros(N, dtype=complex)
    cR = np.zeros(N, dtype=complex)

    cL[1:-1] = (psiL[1:-1] + psiR[1:-1]) / np.sqrt(2.0)
    cR[1:-1] = (psiL[1:-1] - psiR[1:-1]) / np.sqrt(2.0)

    cL[0] = psiR[0]
    cR[0] = psiL[0]
    cL[-1] = psiR[-1]
    cR[-1] = psiL[-1]

    outL = np.zeros(N, dtype=complex)
    outR = np.zeros(N, dtype=complex)
    outL[:-1] = cL[1:]
    outR[1:] = cR[:-1]
    return outL, outR
