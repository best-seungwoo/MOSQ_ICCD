# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Estimator V2 implementation for Aer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers import EstimatorPubLike, PrimitiveResult, PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.quantum_info import Pauli

from qiskit_aer import AerSimulator


@dataclass
class Options:
    """Options for :class:`~.EstimatorV2`."""

    default_precision: float = 0.0
    """The default precision to use if none are specified in :meth:`~run`.
    """

    backend_options: dict = field(default_factory=dict)
    """backend_options: Options passed to AerSimulator."""

    run_options: dict = field(default_factory=dict)
    """run_options: Options passed to run."""


class EstimatorV2(BaseEstimatorV2):
    """Evaluates expectation values for provided quantum circuit and observable combinations.

    Run a fast simulation using Aer.
    Sampling from a normal distribution ``N(expval, precison)`` when set to ``precision``.

    * ``default_precision``: The default precision to use if none are specified in :meth:`~run`.
      Default: 0.0.

    * ``backend_options``: Options passed to AerSimulator.
      Default: {}.

    * ``run_options``: Options passed to :meth:`AerSimulator.run`.
      Default: {}.
    """

    def __init__(
        self,
        *,
        options: dict | None = None,
    ):
        """
        Args:
            options: The options to control the default precision (``default_precision``),
                the backend options (``backend_options``), and
                the runtime options (``run_options``).
        """
        self._options = Options(**options) if options else Options()
        method = "density_matrix" if "noise_model" in self.options.backend_options else "automatic"
        self._backend = AerSimulator(method=method, **self.options.backend_options)
        #SW
        self._sim_exp_etc_time = 0
        self._sim_time = 0
        self._exp_time = 0
        self._transpiled_circuit = None

    def from_backend(self, backend, **options):
        """use external backend"""
        self._backend.from_backend(backend, **options)

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[EstimatorPub]):
        for i, pub in enumerate(pubs):
            if pub.precision < 0.0:
                raise ValueError(
                    f"The {i}-th pub has precision less than 0 ({pub.precision}). ",
                    "But precision should be equal to or larger than 0.",
                )

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs])

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        import time #SW
        etc_start_time = time.time() #SW
        circuit = pub.circuit.copy()
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision
        
        #SW: add MOSQ
        if self._transpiled_circuit: #use a cache
            trans_circuit = self._transpiled_circuit
        else:
            #MOSQ
            trans_circuit = [] # transpiled quantum circuit
            qreg = circuit[0].qubits[0]._register # Quantum Register
            for i_pauli in range(len(circuit)):
                subcirc_inst = circuit[i_pauli] #CircuitInstruction
                oper_subcirc = subcirc_inst.operation #Gate
                def_oper = oper_subcirc.definition #QuantumCircuit = list of CircuitInstructions
                if len(def_oper) == 1: # for U3 gates
                    trans_circuit.append(subcirc_inst)
                else: # for pauli strings (ex: exp(it IIXY))
                    for i in range(len(def_oper)):
                        trans_circuit.append(def_oper[i])
            
            from qiskit.circuit import QuantumCircuit
            trans_circuit = QuantumCircuit.from_instructions(instructions=trans_circuit, qubits=qreg)
            
            # save expval
            paulis = {pauli for obs_dict in observables.ravel() for pauli in obs_dict.keys()}
            for pauli in paulis:
                trans_circuit.save_expectation_value(
                    Pauli(pauli), qubits=range(circuit.num_qubits), label=pauli
                )
            self._transpiled_circuit = trans_circuit
            # print(trans_circuit)

        # calculate broadcasting of parameters and observables
        param_shape = parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, observables)

        parameter_binds = {}
        param_array = parameter_values.as_array(circuit.parameters)
        parameter_binds = {p: param_array[..., i].ravel() for i, p in enumerate(circuit.parameters)}
            
        # run simulation
        sim_start_time = time.time() #SW
        result = self._backend.run(
            trans_circuit, parameter_binds=[parameter_binds], **self.options.run_options
            # circuit, parameter_binds=[parameter_binds], **self.options.run_options
        ).result()
        sim_end_time = time.time() #SW
        self._sim_time += (sim_end_time - sim_start_time) #SW
        
        # calculate expectation values (evs) and standard errors (stds)
        exp_start_time = time.time() #SW
        flat_indices = list(param_indices.ravel())
        evs = np.zeros_like(bc_param_ind, dtype=float)
        stds = np.full(bc_param_ind.shape, precision)
        for pauli, coeff in bc_obs[0].items():
            expval = result.data(0)[pauli]
            evs[0] += expval * coeff
        data_bin_cls = self._make_data_bin(pub)
        data_bin = data_bin_cls(evs=evs, stds=stds)
        exp_end_time = time.time() #SW
        self._exp_time += (exp_end_time - exp_start_time) #SW
        
        #SW: additional metadata
        time_expval = result.results[0].metadata['time_expval']
        time_MOSQ = result.results[0].metadata['time_MOSQ']
        time_HS = result.results[0].metadata['time_HS']
        time_SDGH = result.results[0].metadata['time_SDGH']
        time_RZ = result.results[0].metadata['time_RZ']
        time_CX = result.results[0].metadata['time_CX']
        time_S = result.results[0].metadata['time_S']
        time_SDG = result.results[0].metadata['time_SDG']
        time_H = result.results[0].metadata['time_H']
        
        etc_end_time = time.time() #SW
        self._sim_exp_etc_time += (etc_end_time - etc_start_time) #SW
        
        return PubResult(
            data_bin,
            metadata={"target_precision": precision, "simulator_metadata": result.metadata,
                      "time_expval": time_expval,
                      "time_MOSQ": time_MOSQ,
                      "time_HS": time_HS,
                      "time_SDGH": time_SDGH,
                      "time_RZ": time_RZ,
                      "time_CX": time_CX,
                      "time_S": time_S,
                      "time_SDG": time_SDG,
                      "time_H": time_H,
                      },
        )
