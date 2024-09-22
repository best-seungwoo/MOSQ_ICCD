from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
import numpy as np
from scipy.optimize import minimize
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit_nature.second_q.operators import FermionicOp
import time
import sys
import json
import os

start_time = time.time()

molecule = sys.argv[1]
cache_dir = "./cache"
json_filepath = f"{cache_dir}/{molecule}.json"
fermionic_op_filepath = f"{cache_dir}/{molecule}_fermionic_op.txt"

def deserialize_problem(data, fermionic_op_filepath):
    # Deserialize problem attributes
    num_spatial_orbitals = int(data['num_spatial_orbitals'])
    num_particles = tuple(map(int, data['num_particles'].strip("()").split(',')))
    nuclear_repulsion_energy = np.float64(data['nuclear_repulsion_energy'])
    reference_energy = np.float64(data['reference_energy'])

    with open(fermionic_op_filepath, 'r') as f:
        fermionic_op_str = f.read()
    
    # Parse fermionic_op_str to FermionicOp object
    def parse_fermionic_op_string(fermionic_op_str):
        lines = fermionic_op_str.strip().split("\n")
        data = {}
        max_index = -1
        for line in lines[2:]:
            if line.startswith("+ "):
                line = line[2:]
            elif line.startswith("  "):
                line = line[2:]

            coeff_str, label = line.split(" * ")
            coeff = float(coeff_str.strip())
            label = label.strip().strip("()").strip()
            data[label] = coeff

            for term in label.split():
                index = int(term[2:])
                if index > max_index:
                    max_index = index

        num_spin_orbitals = max_index + 1
        return data, num_spin_orbitals

    fermionic_op_data, num_spin_orbitals = parse_fermionic_op_string(fermionic_op_str)
    fermionic_op = FermionicOp(fermionic_op_data, num_spin_orbitals)

    return fermionic_op, num_spatial_orbitals, num_particles, nuclear_repulsion_energy, reference_energy

def load_problem_from_json(json_filepath, fermionic_op_filepath):
    with open(json_filepath, 'r') as file:
        data = json.load(file)
    return deserialize_problem(data, fermionic_op_filepath)

# Ensure cache directory exists
os.makedirs(cache_dir, exist_ok=True)

# Check if cache file exists
if os.path.exists(json_filepath) and os.path.exists(fermionic_op_filepath):
    fermionic_op, num_spatial_orbitals, num_particles, nuclear_repulsion_energy, reference_energy = load_problem_from_json(json_filepath, fermionic_op_filepath)
else:
    raise FileNotFoundError(f"Cache files not found for molecule {molecule}")

# Map the problem to a fermionic operator and then to a qubit operator
jw_mapper = JordanWignerMapper()
qubit_jw_op = jw_mapper.map(fermionic_op)
hamiltonian = qubit_jw_op

# Define the ansatz using UCCSD
ansatz_UCCSD = UCCSD(
    num_spatial_orbitals,
    num_particles,
    jw_mapper,
    initial_state=HartreeFock(
        num_spatial_orbitals,
        num_particles,
        jw_mapper,
    ),
)
print("num_qubits: " + str(ansatz_UCCSD.num_qubits))

ansatz = ansatz_UCCSD

# zero initialization
initial_point = np.zeros(ansatz.num_parameters)

# random initialization
# np.random.seed(10)
# initial_point = np.random.rand(ansatz.num_parameters)

# qiskit-aer/EstimatorV2
ansatz = ansatz.decompose().decompose()
options = {
    'backend_options': {}, # specify any backend options here
    'run_options': {}      # specify any run options here
}
from qiskit_aer.primitives import EstimatorV2
estimator = EstimatorV2(options=options)

# Convert SparsePauliOp to ObservablesArray
observables_array = ObservablesArray.coerce([hamiltonian])

iteration = 0
time_taken_execute = 0
sim_time = 0
exp_time = 0
sim_exp_etc_time = 0
time_expval = 0
time_Diag = 0
time_Fuse2 = 0
time_Fuse3 = 0
time_Fuse4 = 0
time_Fuse5 = 0

# Define the objective function for the optimizer
def objective_function(params):
    global iteration
    global time_taken_execute
    global sim_time
    global exp_time
    global sim_exp_etc_time
    global time_expval
    global time_Diag
    global time_Fuse2
    global time_Fuse3
    global time_Fuse4
    global time_Fuse5
    iteration += 1
    parameter_binds = BindingsArray({param: val for param, val in zip(ansatz.parameters, params)})
    estimator_pub = EstimatorPub(circuit=ansatz, observables=observables_array, parameter_values=parameter_binds)
    
    job = estimator.run([estimator_pub])

    result = job.result()
    evs = result[0].data['evs']
    formatted_evs = np.array2string(evs, precision=16, floatmode='unique')
    print(f"{iteration}: {formatted_evs}")
    time_taken_execute += result[0].metadata['simulator_metadata']['time_taken_execute']
    sim_time = estimator._sim_time
    exp_time = estimator._exp_time
    sim_exp_etc_time = estimator._sim_exp_etc_time
    
    time_expval += result[0].metadata['time_expval']
    time_Diag += result[0].metadata['time_Diag']
    time_Fuse2 += result[0].metadata['time_Fuse2']
    time_Fuse3 += result[0].metadata['time_Fuse3']
    time_Fuse4 += result[0].metadata['time_Fuse4']
    time_Fuse5 += result[0].metadata['time_Fuse5']
    
# Perform the optimization
maxit = 1
if molecule in ["H2", "LiH", "BeH2"]:
    maxit = 10
result = minimize(objective_function, initial_point, method='COBYLA'
        ,options = {'maxiter': maxit}
        )

print("T_sim: " + str(sim_time - time_expval))
print("T_exp: " + str(exp_time + time_expval))
print("T_etc: " + str(sim_exp_etc_time - exp_time - sim_time))
print("T_opt = T_tot - T_sim - T_exp - T_etc")
print("time_Diag: " + str(time_Diag))
print("time_Fuse2: " + str(time_Fuse2))
print("time_Fuse3: " + str(time_Fuse3))
print("time_Fuse4: " + str(time_Fuse4))
print("time_Fuse5: " + str(time_Fuse5))
print("total iteration: " + str(iteration))