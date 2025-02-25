/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _statevector_state_hpp
#define _statevector_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/config.hpp"
#include "framework/json.hpp"
#include "framework/utils.hpp"
#include "qubitvector.hpp"
#include "simulators/chunk_utils.hpp"
#include "simulators/state.hpp"

#ifdef AER_THRUST_SUPPORTED
#include "qubitvector_thrust.hpp"
#endif

namespace AER {

namespace Statevector {

//SW: Timer type
using myclock_t = std::chrono::high_resolution_clock;

using OpType = Operations::OpType;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {OpType::gate,
     OpType::measure,
     OpType::reset,
     OpType::initialize,
     OpType::barrier,
     OpType::bfunc,
     OpType::roerror,
     OpType::matrix,
     OpType::diagonal_matrix,
     OpType::multiplexer,
     OpType::kraus,
     OpType::qerror_loc,
     OpType::sim_op,
     OpType::set_statevec,
     OpType::save_expval,
     OpType::save_expval_var,
     OpType::save_probs,
     OpType::save_probs_ket,
     OpType::save_amps,
     OpType::save_amps_sq,
     OpType::save_state,
     OpType::save_statevec,
     OpType::save_statevec_dict,
     OpType::save_densmat,
     OpType::jump,
     OpType::mark},
    // Gates
    {
        "u1",   "u2",    "u3",     "u",       "U",     "CX",       "cx",
        "cz",   "cy",    "cp",     "cu1",     "cu2",   "cu3",      "swap",
        "id",   "p",     "x",      "y",       "z",     "h",        "s",
        "sdg",  "t",     "tdg",    "r",       "rx",    "ry",       "rz",
        "rxx",  "ryy",   "rzz",    "rzx",     "ccx",   "ccz",      "cswap",
        "mcx",  "mcy",   "mcz",    "mcu1",    "mcu2",  "mcu3",     "mcswap",
        "mcr",  "mcrx",  "mcry",   "mcrz",    "sx",    "sxdg",     "csx",
        "mcsx", "csxdg", "mcsxdg", "delay",   "pauli", "mcx_gray", "cu",
        "mcu",  "mcp",   "ecr",    "mcphase", "crx",   "cry",      "crz",
        "mcu",  "mcp",   "ecr",    "mcphase", "crx",   "cry",      "crz",
        "H+S",  "SDG+H", "MOSQ",   "MOSQ_CR", //SW
    });

// Allowed gates enum class
enum class Gates {
  id,
  h,
  s,
  sdg,
  t,
  tdg,
  rxx,
  ryy,
  rzz,
  rzx,
  mcx,
  mcy,
  mcz,
  mcr,
  mcrx,
  mcry,
  mcrz,
  mcp,
  mcu2,
  mcu3,
  mcu,
  mcswap,
  mcsx,
  mcsxdg,
  pauli,
  ecr,
  hs, //SW
  sdgh, //SW
  mosq, //SW
  mosq_cr //SW
};

//=========================================================================
// QubitVector State subclass
//=========================================================================

template <class statevec_t = QV::QubitVector<double>>
class State : public QuantumState::State<statevec_t> {
public:
  using BaseState = QuantumState::State<statevec_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return statevec_t::name(); }

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  void apply_op(const Operations::Op &op, ExperimentResult &result,
                RngEngine &rng, bool final_op = false) override;

  // memory allocation (previously called before inisitalize_qreg)
  bool allocate(uint_t num_qubits, uint_t block_bits,
                uint_t num_parallel_shots = 1) override;

  // Initializes an n-qubit state to the all |0> state
  void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit state
  void initialize_statevector(uint_t num_qubits, statevec_t &&state);

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is independent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  virtual void set_config(const Config &config) override;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  std::vector<SampleVector> sample_measure(const reg_t &qubits, uint_t shots,
                                           RngEngine &rng) override;

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string &pauli) override;
  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  // Initialize OpenMP settings for the underlying QubitVector class
  void initialize_omp();

  auto move_to_vector(void);
  auto copy_to_vector(void);

  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------

  // Applies a supported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(const Operations::Op &op);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  virtual void apply_measure(const reg_t &qubits, const reg_t &cmemory,
                             const reg_t &cregister, RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits, RngEngine &rng);

  // Initialize the specified qubits to a given state |psi>
  // by applying a reset to the these qubits and then
  // computing the tensor product with the new state |psi>
  // /psi> is given in params
  void apply_initialize(const reg_t &qubits, const cvector_t &params,
                        RngEngine &rng);

  void initialize_from_vector(const cvector_t &params);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const Operations::Op &op);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t &vmat);

  // apply diagonal matrix
  void apply_diagonal_matrix(const reg_t &qubits, const cvector_t &diag);

  // Apply a vector of control matrices to given qubits (identity on all other
  // qubits)
  void apply_multiplexer(const reg_t &control_qubits,
                         const reg_t &target_qubits,
                         const std::vector<cmatrix_t> &mmat);

  // Apply stacked (flat) version of multiplexer matrix to target qubits (using
  // control qubits to select matrix instance)
  void apply_multiplexer(const reg_t &control_qubits,
                         const reg_t &target_qubits, const cmatrix_t &mat);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &krausops,
                   RngEngine &rng);

  // Return the reduced density matrix for the simulator
  cmatrix_t density_matrix(const reg_t &qubits);

  // Apply the global phase
  void apply_global_phase();

  // //SW
  // double time_taken = 0.0;

protected:
  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current state of the statevector simulator
  // If `last_op` is True this will use move semantics to move the simulator
  // state to the results, otherwise it will use copy semantics to leave
  // the current simulator state unchanged.
  void apply_save_statevector(const Operations::Op &op,
                              ExperimentResult &result, bool last_op);

  // Save the current state of the statevector simulator as a ket-form map.
  void apply_save_statevector_dict(const Operations::Op &op,
                                   ExperimentResult &result);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(const Operations::Op &op,
                                 ExperimentResult &result);

  // Helper function for computing expectation value
  void apply_save_probs(const Operations::Op &op, ExperimentResult &result);

  // Helper function for saving amplitudes and amplitudes squared
  void apply_save_amplitudes(const Operations::Op &op,
                             ExperimentResult &result);

  //-----------------------------------------------------------------------
  // Measurement Helpers
  //-----------------------------------------------------------------------

  // Return vector of measure probabilities for specified qubits
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  // TODO: move to private (no longer part of base class)
  rvector_t measure_probs(const reg_t &qubits) const;

  // Sample the measurement outcome for qubits
  // return a pair (m, p) of the outcome m, and its corresponding
  // probability p.
  // Outcome is given as an int: Eg for two-qubits {q0, q1} we have
  // 0 -> |q1 = 0, q0 = 0> state
  // 1 -> |q1 = 0, q0 = 1> state
  // 2 -> |q1 = 1, q0 = 0> state
  // 3 -> |q1 = 1, q0 = 1> state
  std::pair<uint_t, double> sample_measure_with_prob(const reg_t &qubits,
                                                     RngEngine &rng);

  void measure_reset_update(const std::vector<uint_t> &qubits,
                            const uint_t final_state, const uint_t meas_state,
                            const double meas_prob);

  // Helper function to convert a vector to a reduced density matrix
  template <class T>
  cmatrix_t vec2density(const reg_t &qubits, const T &vec);

  //-----------------------------------------------------------------------
  // Single-qubit gate helpers
  //-----------------------------------------------------------------------

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const uint_t qubit, const complex_t phase);

  //-----------------------------------------------------------------------
  // Multi-controlled u3
  //-----------------------------------------------------------------------

  // Apply N-qubit multi-controlled single qubit gate specified by
  // 4 parameters u4(theta, phi, lambda, gamma)
  // NOTE: if N=1 this is just a regular u4 gate.
  void apply_gate_mcu(const reg_t &qubits, const double theta, const double phi,
                      const double lambda, const double gamma);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // OpenMP qubit threshold
  int omp_qubit_threshold_ = 14;

  // QubitVector sample measure index size
  int sample_measure_index_size_ = 10;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-10;

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;
};

//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

template <class statevec_t>
const stringmap_t<Gates> State<statevec_t>::gateset_(
    {                         // 1-qubit gates
     {"delay", Gates::id},    // Delay gate
     {"id", Gates::id},       // Pauli-Identity gate
     {"x", Gates::mcx},       // Pauli-X gate
     {"y", Gates::mcy},       // Pauli-Y gate
     {"z", Gates::mcz},       // Pauli-Z gate
     {"s", Gates::s},         // Phase gate (aka sqrt(Z) gate)
     {"sdg", Gates::sdg},     // Conjugate-transpose of Phase gate
     {"h", Gates::h},         // Hadamard gate (X + Z / sqrt(2))
     {"t", Gates::t},         // T-gate (sqrt(S))
     {"tdg", Gates::tdg},     // Conjguate-transpose of T gate
     {"p", Gates::mcp},       // Parameterized phase gate
     {"sx", Gates::mcsx},     // Sqrt(X) gate
     {"sxdg", Gates::mcsxdg}, // Inverse Sqrt(X) gate
     {"H+S", Gates::hs}, //SW
     {"SDG+H", Gates::sdgh}, //SW
     /* 1-qubit rotation Gates */
     {"r", Gates::mcr},   // R rotation gate
     {"rx", Gates::mcrx}, // Pauli-X rotation gate
     {"ry", Gates::mcry}, // Pauli-Y rotation gate
     {"rz", Gates::mcrz}, // Pauli-Z rotation gate
     /* Waltz Gates */
     {"u1", Gates::mcp},  // zero-X90 pulse waltz gate
     {"u2", Gates::mcu2}, // single-X90 pulse waltz gate
     {"u3", Gates::mcu3}, // two X90 pulse waltz gate
     {"u", Gates::mcu3},  // two X90 pulse waltz gate
     {"U", Gates::mcu3},  // two X90 pulse waltz gate
     /* 2-qubit gates */
     {"CX", Gates::mcx},       // Controlled-X gate (CNOT)
     {"cx", Gates::mcx},       // Controlled-X gate (CNOT)
     {"cy", Gates::mcy},       // Controlled-Y gate
     {"cz", Gates::mcz},       // Controlled-Z gate
     {"cp", Gates::mcp},       // Controlled-Phase gate
     {"cu1", Gates::mcp},      // Controlled-u1 gate
     {"cu2", Gates::mcu2},     // Controlled-u2 gate
     {"cu3", Gates::mcu3},     // Controlled-u3 gate
     {"cu", Gates::mcu},       // Controlled-u4 gate
     {"cp", Gates::mcp},       // Controlled-Phase gate
     {"swap", Gates::mcswap},  // SWAP gate
     {"rxx", Gates::rxx},      // Pauli-XX rotation gate
     {"ryy", Gates::ryy},      // Pauli-YY rotation gate
     {"rzz", Gates::rzz},      // Pauli-ZZ rotation gate
     {"rzx", Gates::rzx},      // Pauli-ZX rotation gate
     {"csx", Gates::mcsx},     // Controlled-Sqrt(X) gate
     {"csxdg", Gates::mcsxdg}, // Controlled-Sqrt(X)dg gate
     {"ecr", Gates::ecr},      // ECR Gate
     {"crx", Gates::mcrx},     // Controlled X-rotation gate
     {"cry", Gates::mcry},     // Controlled Y-rotation gate
     {"crz", Gates::mcrz},     // Controlled Z-rotation gate
     /* 3-qubit gates */
     {"ccx", Gates::mcx},      // Controlled-CX gate (Toffoli)
     {"ccz", Gates::mcz},      // Controlled-CZ gate
     {"cswap", Gates::mcswap}, // Controlled SWAP gate (Fredkin)
     /* Multi-qubit controlled gates */
     {"mcx", Gates::mcx},       // Multi-controlled-X gate
     {"mcy", Gates::mcy},       // Multi-controlled-Y gate
     {"mcz", Gates::mcz},       // Multi-controlled-Z gate
     {"mcr", Gates::mcr},       // Multi-controlled R-rotation gate
     {"mcrx", Gates::mcrx},     // Multi-controlled X-rotation gate
     {"mcry", Gates::mcry},     // Multi-controlled Y-rotation gate
     {"mcrz", Gates::mcrz},     // Multi-controlled Z-rotation gate
     {"mcphase", Gates::mcp},   // Multi-controlled-Phase gate
     {"mcp", Gates::mcp},       // Multi-controlled-Phase gate
     {"mcu1", Gates::mcp},      // Multi-controlled-u1
     {"mcu2", Gates::mcu2},     // Multi-controlled-u2
     {"mcu3", Gates::mcu3},     // Multi-controlled-u3
     {"mcu", Gates::mcu},       // Multi-controlled-u4
     {"mcswap", Gates::mcswap}, // Multi-controlled SWAP gate
     {"mcsx", Gates::mcsx},     // Multi-controlled-Sqrt(X) gate
     {"mcsxdg", Gates::mcsxdg}, // Multi-controlled-Sqrt(X)dg gate
     {"pauli", Gates::pauli},   // Multi-qubit Pauli gate
     {"MOSQ", Gates::mosq},   //SW
     {"MOSQ_CR", Gates::mosq_cr}, //SW
     {"mcx_gray", Gates::mcx}});

//=========================================================================
// Implementation: Base class method overrides
//=========================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------

template <class statevec_t>
void State<statevec_t>::initialize_qreg(uint_t num_qubits) {
  initialize_omp();

  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize();

  apply_global_phase();
}

template <class statevec_t>
void State<statevec_t>::initialize_statevector(uint_t num_qubits,
                                               statevec_t &&state) {
  if (state.num_qubits() != num_qubits) {
    throw std::invalid_argument("QubitVector::State::initialize: initial state "
                                "does not match qubit number");
  }

  BaseState::qreg_ = std::move(state);

  apply_global_phase();
}

template <class statevec_t>
void State<statevec_t>::initialize_omp() {
  BaseState::qreg_.set_omp_threshold(omp_qubit_threshold_);
  if (BaseState::threads_ > 0) // set allowed OMP threads in qubitvector
    BaseState::qreg_.set_omp_threads(BaseState::threads_);
}

template <class statevec_t>
bool State<statevec_t>::allocate(uint_t num_qubits, uint_t block_bits,
                                 uint_t num_parallel_shots) {
  if (BaseState::max_matrix_qubits_ > 0)
    BaseState::qreg_.set_max_matrix_bits(BaseState::max_matrix_qubits_);
  if (BaseState::max_sampling_shots_ > 0)
    BaseState::qreg_.set_max_sampling_shots(BaseState::max_sampling_shots_);

  BaseState::qreg_.set_target_gpus(BaseState::target_gpus_);
#ifdef AER_CUSTATEVEC
  BaseState::qreg_.cuStateVec_enable(BaseState::cuStateVec_enable_);
#endif
  BaseState::qreg_.chunk_setup(block_bits, num_qubits, 0, 1);

  return true;
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

template <class statevec_t>
void State<statevec_t>::apply_global_phase() {
  if (BaseState::has_global_phase_)
    BaseState::qreg_.apply_diagonal_matrix(
        {0}, {BaseState::global_phase_, BaseState::global_phase_});
}

template <class statevec_t>
size_t State<statevec_t>::required_memory_mb(
    uint_t num_qubits, const std::vector<Operations::Op> &ops) const {
  (void)ops; // avoid unused variable compiler warning
  return BaseState::qreg_.required_memory_mb(num_qubits);
}

template <class statevec_t>
void State<statevec_t>::set_config(const Config &config) {
  BaseState::set_config(config);

  // Set threshold for truncating states to be saved
  json_chop_threshold_ = config.zero_threshold;
  BaseState::qreg_.set_json_chop_threshold(json_chop_threshold_);

  // Set OMP threshold for state update functions
  omp_qubit_threshold_ = config.statevector_parallel_threshold;

  // Set the sample measure indexing size
  if (config.statevector_sample_measure_opt) {
    int index_size = config.statevector_sample_measure_opt;
    BaseState::qreg_.set_sample_measure_index_size(index_size);
  }
}

template <class statevec_t>
auto State<statevec_t>::move_to_vector(void) {
  return std::move(BaseState::qreg_.move_to_vector());
}

template <class statevec_t>
auto State<statevec_t>::copy_to_vector(void) {
  return BaseState::qreg_.copy_to_vector();
}

//=========================================================================
// Implementation: apply operations
//=========================================================================
template <class statevec_t>
void State<statevec_t>::apply_op(const Operations::Op &op,
                                 ExperimentResult &result, RngEngine &rng,
                                 bool final_op) {
  // printf("apply_op\n");
  auto timer_start = myclock_t::now();
  auto timer_stop = myclock_t::now();
  if (BaseState::creg().check_conditional(op)) {
    switch (op.type) {
    case OpType::barrier:
    case OpType::nop:
    case OpType::qerror_loc:
      break;
    case OpType::reset:
      apply_reset(op.qubits, rng);
      break;
    case OpType::initialize:
      apply_initialize(op.qubits, op.params, rng);
      break;
    case OpType::measure:
      apply_measure(op.qubits, op.memory, op.registers, rng);
      break;
    case OpType::bfunc:
      BaseState::creg().apply_bfunc(op);
      break;
    case OpType::roerror:
      BaseState::creg().apply_roerror(op, rng);
      break;
    case OpType::gate:
      apply_gate(op);
      break;
    case OpType::matrix:
      apply_matrix(op);
  //SW
  timer_stop = myclock_t::now(); // stop timer
  if(op.qubits.size()==2) this->time_Fuse2 += std::chrono::duration<double>(timer_stop - timer_start).count();
  else if(op.qubits.size()==3) this->time_Fuse3 += std::chrono::duration<double>(timer_stop - timer_start).count();
  else if(op.qubits.size()==4) this->time_Fuse4 += std::chrono::duration<double>(timer_stop - timer_start).count();
  else if(op.qubits.size()==5) this->time_Fuse5 += std::chrono::duration<double>(timer_stop - timer_start).count();
      break;
    case OpType::diagonal_matrix:
      apply_diagonal_matrix(op.qubits, op.params);
      //SW
      timer_stop = myclock_t::now(); // stop timer
      this->time_Diag += std::chrono::duration<double>(timer_stop - timer_start).count();
      break;
    case OpType::multiplexer:
      apply_multiplexer(op.regs[0], op.regs[1],
                        op.mats); // control qubits ([0]) & target qubits([1])
      break;
    case OpType::kraus:
      apply_kraus(op.qubits, op.mats, rng);
      break;
    case OpType::sim_op:
      if (op.name == "begin_register_blocking") {
        BaseState::qreg_.enter_register_blocking(op.qubits);
      } else if (op.name == "end_register_blocking") {
        BaseState::qreg_.leave_register_blocking();
      }
      break;
    case OpType::set_statevec:
      initialize_from_vector(op.params);
      break;
    case OpType::save_expval:
    case OpType::save_expval_var:
      //SW: start T_exp
      timer_start = myclock_t::now(); // start timer
      BaseState::apply_save_expval(op, result);
      //SW: start T_exp
      timer_stop = myclock_t::now(); // stop timer
      this->time_taken += std::chrono::duration<double>(timer_stop - timer_start).count();
      // std::cout << this->time_taken << std::endl;
      break;
    case OpType::save_densmat:
      apply_save_density_matrix(op, result);
      break;
    case OpType::save_state:
    case OpType::save_statevec:
      apply_save_statevector(op, result, final_op);
      break;
    case OpType::save_statevec_dict:
      apply_save_statevector_dict(op, result);
      break;
    case OpType::save_probs:
    case OpType::save_probs_ket:
      apply_save_probs(op, result);
      break;
    case OpType::save_amps:
    case OpType::save_amps_sq:
      apply_save_amplitudes(op, result);
      break;
    default:
      throw std::invalid_argument("QubitVector::State::invalid instruction \'" +
                                  op.name + "\'.");
    }
  }
}

//=========================================================================
// Implementation: Save data
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_save_probs(const Operations::Op &op,
                                         ExperimentResult &result) {
  // get probs as hexadecimal
  auto probs = measure_probs(op.qubits);
  if (op.type == Operations::OpType::save_probs_ket) {
    // Convert to ket dict
    result.save_data_average(BaseState::creg(), op.string_params[0],
                             Utils::vec2ket(probs, json_chop_threshold_, 16),
                             op.type, op.save_type);
  } else {
    result.save_data_average(BaseState::creg(), op.string_params[0],
                             std::move(probs), op.type, op.save_type);
  }
}

template <class statevec_t>
double State<statevec_t>::expval_pauli(const reg_t &qubits,
                                       const std::string &pauli) {
  return BaseState::qreg_.expval_pauli(qubits, pauli);
}

template <class statevec_t>
void State<statevec_t>::apply_save_statevector(const Operations::Op &op,
                                               ExperimentResult &result,
                                               bool last_op) {
  if (op.qubits.size() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  std::string key =
      (op.string_params[0] == "_method_") ? "statevector" : op.string_params[0];

  if (last_op) {
    auto v = move_to_vector();
    result.save_data_pershot(BaseState::creg(), key, std::move(v),
                             OpType::save_statevec, op.save_type);
  } else {
    result.save_data_pershot(BaseState::creg(), key, copy_to_vector(),
                             OpType::save_statevec, op.save_type);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_save_statevector_dict(const Operations::Op &op,
                                                    ExperimentResult &result) {
  if (op.qubits.size() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  auto state_ket = BaseState::qreg_.vector_ket(json_chop_threshold_);
  std::map<std::string, complex_t> result_state_ket;
  for (auto const &it : state_ket) {
    result_state_ket[it.first] = it.second;
  }
  result.save_data_pershot(BaseState::creg(), op.string_params[0],
                           std::move(result_state_ket), op.type, op.save_type);
}

template <class statevec_t>
void State<statevec_t>::apply_save_density_matrix(const Operations::Op &op,
                                                  ExperimentResult &result) {
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    reduced_state[0] = BaseState::qreg_.norm();
  } else {
    reduced_state = density_matrix(op.qubits);
  }

  result.save_data_average(BaseState::creg(), op.string_params[0],
                           std::move(reduced_state), op.type, op.save_type);
}

template <class statevec_t>
void State<statevec_t>::apply_save_amplitudes(const Operations::Op &op,
                                              ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save_amplitudes instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  if (op.type == Operations::OpType::save_amps) {
    Vector<complex_t> amps(size, false);
    for (int_t i = 0; i < size; ++i) {
      amps[i] = BaseState::qreg_.get_state(op.int_params[i]);
    }
    result.save_data_pershot(BaseState::creg(), op.string_params[0],
                             std::move(amps), op.type, op.save_type);
  } else {
    rvector_t amps_sq(size, 0);
    for (int_t i = 0; i < size; ++i) {
      amps_sq[i] = BaseState::qreg_.probability(op.int_params[i]);
    }
    result.save_data_average(BaseState::creg(), op.string_params[0],
                             std::move(amps_sq), op.type, op.save_type);
  }
}

template <class statevec_t>
cmatrix_t State<statevec_t>::density_matrix(const reg_t &qubits) {
  return vec2density(qubits, copy_to_vector());
}

template <class statevec_t>
template <class T>
cmatrix_t State<statevec_t>::vec2density(const reg_t &qubits, const T &vec) {
  const size_t N = qubits.size();
  const size_t DIM = 1ULL << N;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Return full density matrix
  cmatrix_t densmat(DIM, DIM);
  if ((N == BaseState::qreg_.num_qubits()) && (qubits == qubits_sorted)) {
    const int_t mask = QV::MASKS[N];
#pragma omp parallel for if (2 * N > (size_t)omp_qubit_threshold_ &&           \
                             BaseState::threads_ > 1)                          \
    num_threads(BaseState::threads_)
    for (int_t rowcol = 0; rowcol < int_t(DIM * DIM); ++rowcol) {
      const int_t row = rowcol >> N;
      const int_t col = rowcol & mask;
      densmat(row, col) = complex_t(vec[row]) * complex_t(std::conj(vec[col]));
    }
  } else {
    const size_t END = 1ULL << (BaseState::qreg_.num_qubits() - N);
    // Initialize matrix values with first block
    {
      const auto inds = QV::indexes(qubits, qubits_sorted, 0);
      for (size_t row = 0; row < DIM; ++row)
        for (size_t col = 0; col < DIM; ++col) {
          densmat(row, col) =
              complex_t(vec[inds[row]]) * complex_t(std::conj(vec[inds[col]]));
        }
    }
    // Accumulate remaining blocks
    for (size_t k = 1; k < END; k++) {
      // store entries touched by U
      const auto inds = QV::indexes(qubits, qubits_sorted, k);
      for (size_t row = 0; row < DIM; ++row)
        for (size_t col = 0; col < DIM; ++col) {
          densmat(row, col) +=
              complex_t(vec[inds[row]]) * complex_t(std::conj(vec[inds[col]]));
        }
    }
  }
  return densmat;
}

//=========================================================================
// Implementation: Matrix multiplication
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_gate(const Operations::Op &op) {
  // printf("gate name: %s\n", op.name.c_str());
  auto timer_start = myclock_t::now();
  auto timer_stop = myclock_t::now();
  // CPU qubit vector does not handle chunk ID inside kernel, so modify op here
  if (BaseState::num_global_qubits_ > BaseState::qreg_.num_qubits() &&
      !BaseState::qreg_.support_global_indexing()) {
    reg_t qubits_in, qubits_out;
    if (op.name[0] == 'c' || op.name.find("mc") == 0) {
      Chunk::get_inout_ctrl_qubits(op, BaseState::qreg_.num_qubits(), qubits_in,
                                   qubits_out);
    }
    if (qubits_out.size() > 0) {
      uint_t mask = 0;
      for (uint_t i = 0; i < qubits_out.size(); i++) {
        mask |= (1ull << (qubits_out[i] - BaseState::qreg_.num_qubits()));
      }
      if ((BaseState::qreg_.chunk_index() & mask) == mask) {
        Operations::Op new_op = Chunk::correct_gate_op_in_chunk(op, qubits_in);
        apply_gate(new_op);
      }
      return;
    }
  }

  // uint_t num_par_SW;//SW

  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
        "QubitVectorState::invalid gate instruction \'" + op.name + "\'.");
  switch (it->second) {
  case Gates::mcx:
    // Includes X, CX, CCX, etc
    BaseState::qreg_.apply_mcx(op.qubits);
    timer_stop = myclock_t::now(); // stop timer
    this->time_CX += std::chrono::duration<double>(timer_stop - timer_start).count();
    break;
  case Gates::mcy:
    // Includes Y, CY, CCY, etc
    BaseState::qreg_.apply_mcy(op.qubits);
    break;
  case Gates::mcz:
    // Includes Z, CZ, CCZ, etc
    BaseState::qreg_.apply_mcphase(op.qubits, -1);
    break;
  case Gates::mcr:
    BaseState::qreg_.apply_mcu(op.qubits,
                               Linalg::VMatrix::r(op.params[0], op.params[1]));
    break;
  case Gates::mcrx:
    BaseState::qreg_.apply_rotation(op.qubits, QV::Rotation::x,
                                    std::real(op.params[0]));
    break;
  case Gates::mcry:
    BaseState::qreg_.apply_rotation(op.qubits, QV::Rotation::y,
                                    std::real(op.params[0]));
    break;
  case Gates::mcrz:
    // num_par_SW = op.params.size();
    // printf("size: %lu\n", num_par_SW);
    // for(uint_t iter = 0; iter < num_par_SW; iter++){
    //   printf("%lu: %f + %fi\n", iter, std::real(op.params[iter]), std::imag(op.params[iter]));
    // }
    // std::cout << "(phase): " << op.params[0] << std::endl;
    BaseState::qreg_.apply_rotation(op.qubits, QV::Rotation::z,
                                    std::real(op.params[0]));
    timer_stop = myclock_t::now(); // stop timer
    this->time_RZ += std::chrono::duration<double>(timer_stop - timer_start).count();
    break;
  case Gates::rxx:
    BaseState::qreg_.apply_rotation(op.qubits, QV::Rotation::xx,
                                    std::real(op.params[0]));
    break;
  case Gates::ryy:
    BaseState::qreg_.apply_rotation(op.qubits, QV::Rotation::yy,
                                    std::real(op.params[0]));
    break;
  case Gates::rzz:
    BaseState::qreg_.apply_rotation(op.qubits, QV::Rotation::zz,
                                    std::real(op.params[0]));
    break;
  case Gates::rzx:
    BaseState::qreg_.apply_rotation(op.qubits, QV::Rotation::zx,
                                    std::real(op.params[0]));
    break;
  case Gates::ecr:
    BaseState::qreg_.apply_matrix(op.qubits, Linalg::VMatrix::ECR);
  case Gates::id:
    break;
  case Gates::h:
    apply_gate_mcu(op.qubits, M_PI / 2., 0., M_PI, 0.);
    timer_stop = myclock_t::now(); // stop timer
    this->time_H += std::chrono::duration<double>(timer_stop - timer_start).count();
    break;
  case Gates::sdgh: //SW
    apply_gate_mcu(op.qubits, M_PI / 2., 0., M_PI / 2., 0.);
    timer_stop = myclock_t::now(); // stop timer
    this->time_SDGH += std::chrono::duration<double>(timer_stop - timer_start).count();
    break;
  case Gates::hs: //SW
    apply_gate_mcu(op.qubits, M_PI / 2., M_PI / 2., M_PI, 0.);
    timer_stop = myclock_t::now(); // stop timer
    this->time_HS += std::chrono::duration<double>(timer_stop - timer_start).count();
    break;
  case Gates::s:
    apply_gate_phase(op.qubits[0], complex_t(0., 1.));
    timer_stop = myclock_t::now(); // stop timer
    this->time_S += std::chrono::duration<double>(timer_stop - timer_start).count();
    break;
  case Gates::sdg:
    apply_gate_phase(op.qubits[0], complex_t(0., -1.));
    timer_stop = myclock_t::now(); // stop timer
    this->time_SDG += std::chrono::duration<double>(timer_stop - timer_start).count();
    break;
  case Gates::t: {
    const double isqrt2{1. / std::sqrt(2)};
    apply_gate_phase(op.qubits[0], complex_t(isqrt2, isqrt2));
  } break;
  case Gates::tdg: {
    const double isqrt2{1. / std::sqrt(2)};
    apply_gate_phase(op.qubits[0], complex_t(isqrt2, -isqrt2));
  } break;
  case Gates::mcswap:
    // Includes SWAP, CSWAP, etc
    BaseState::qreg_.apply_mcswap(op.qubits);
    break;
  case Gates::mcu3:
    // Includes u3, cu3, etc
    apply_gate_mcu(op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                   std::real(op.params[2]), 0.);
    break;
  case Gates::mcu:
    // Includes u3, cu3, etc
    apply_gate_mcu(op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                   std::real(op.params[2]), std::real(op.params[3]));
    break;
  case Gates::mcu2:
    // Includes u2, cu2, etc
    apply_gate_mcu(op.qubits, M_PI / 2., std::real(op.params[0]),
                   std::real(op.params[1]), 0.);
    break;
  case Gates::mcp:
    // Includes u1, cu1, p, cp, mcp etc
    BaseState::qreg_.apply_mcphase(op.qubits,
                                   std::exp(complex_t(0, 1) * op.params[0]));
    break;
  case Gates::mcsx:
    // Includes sx, csx, mcsx etc
    BaseState::qreg_.apply_mcu(op.qubits, Linalg::VMatrix::SX);
    break;
  case Gates::mcsxdg:
    BaseState::qreg_.apply_mcu(op.qubits, Linalg::VMatrix::SXDG);
    break;
  case Gates::pauli:
    BaseState::qreg_.apply_pauli(op.qubits, op.string_params[0]);
    break;
  case Gates::mosq: //SW
    // num_par_SW = op.params.size();
    // printf("size: %lu\n", num_par_SW);
    // for(uint_t iter = 0; iter < num_par_SW; iter++){
    //   printf("%lu: %f + %fi\n", iter, std::real(op.params[iter]), std::imag(op.params[iter]));
    // }
    // std::cout << "(operating qubit num): " << op.qubits.size() << std::endl;
    // std::cout << "(phase): " << op.params[0] << std::endl;
    BaseState::qreg_.apply_MOSQ(op.qubits, std::exp(complex_t(0, 1) * op.params[0]));
    timer_stop = myclock_t::now(); // stop timer
    this->time_MOSQ += std::chrono::duration<double>(timer_stop - timer_start).count();
    break;
  case Gates::mosq_cr: //SW
    // num_par_SW = op.params.size();
    // printf("size: %lu\n", num_par_SW);
    // for(uint_t iter = 0; iter < num_par_SW; iter++){
    //   printf("%lu: %f + %fi\n", iter, std::real(op.params[iter]), std::imag(op.params[iter]));
    // }
    // std::cout << "(operating qubit num): " << op.qubits.size() << std::endl;
    // std::cout << "(phase): " << op.params[0] << std::endl;
    BaseState::qreg_.apply_MOSQ_CR(op.qubits, std::exp(complex_t(0, 1) * op.params[0]), op.params[1], op.params[2], op.params[3]);
    timer_stop = myclock_t::now(); // stop timer
    this->time_MOSQ_CR += std::chrono::duration<double>(timer_stop - timer_start).count();
    break;
  default:
    // We shouldn't reach here unless there is a bug in gateset
    throw std::invalid_argument(
        "QubitVector::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

template <class statevec_t>
void State<statevec_t>::apply_multiplexer(const reg_t &control_qubits,
                                          const reg_t &target_qubits,
                                          const cmatrix_t &mat) {
  if (control_qubits.empty() == false && target_qubits.empty() == false &&
      mat.size() > 0) {
    cvector_t vmat = Utils::vectorize_matrix(mat);
    BaseState::qreg_.apply_multiplexer(control_qubits, target_qubits, vmat);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_matrix(const Operations::Op &op) {
  if (op.qubits.empty() == false && op.mats[0].size() > 0) {
    if (Utils::is_diagonal(op.mats[0], .0)) {
      apply_diagonal_matrix(op.qubits, Utils::matrix_diagonal(op.mats[0]));
    } else {
      BaseState::qreg_.apply_matrix(op.qubits,
                                    Utils::vectorize_matrix(op.mats[0]));
    }
  }
}

template <class statevec_t>
void State<statevec_t>::apply_matrix(const reg_t &qubits,
                                     const cvector_t &vmat) {
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    apply_diagonal_matrix(qubits, vmat);
  } else {
    BaseState::qreg_.apply_matrix(qubits, vmat);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_diagonal_matrix(const reg_t &qubits,
                                              const cvector_t &diag) {
  if (BaseState::num_global_qubits_ > BaseState::qreg_.num_qubits() &&
      !BaseState::qreg_.support_global_indexing()) {
    reg_t qubits_in = qubits;
    cvector_t diag_in = diag;
    Chunk::block_diagonal_matrix(BaseState::qreg_.chunk_index(),
                                 BaseState::qreg_.num_qubits(), qubits_in,
                                 diag_in);
    BaseState::qreg_.apply_diagonal_matrix(qubits_in, diag_in);
  } else {
    BaseState::qreg_.apply_diagonal_matrix(qubits, diag);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_gate_mcu(const reg_t &qubits, double theta,
                                       double phi, double lambda,
                                       double gamma) {
  BaseState::qreg_.apply_mcu(qubits,
                             Linalg::VMatrix::u4(theta, phi, lambda, gamma));
}

template <class statevec_t>
void State<statevec_t>::apply_gate_phase(uint_t qubit, complex_t phase) {
  cvector_t diag = {{1., phase}};
  apply_diagonal_matrix(reg_t({qubit}), diag);
}

//=========================================================================
// Implementation: Reset, Initialize and Measurement Sampling
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_measure(const reg_t &qubits, const reg_t &cmemory,
                                      const reg_t &cregister, RngEngine &rng) {
  // Actual measurement outcome
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Implement measurement update
  measure_reset_update(qubits, meas.first, meas.first, meas.second);
  const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());
  BaseState::creg().store_measure(outcome, cmemory, cregister);
}

template <class statevec_t>
rvector_t State<statevec_t>::measure_probs(const reg_t &qubits) const {
  return BaseState::qreg_.probabilities(qubits);
}

template <class statevec_t>
void State<statevec_t>::apply_reset(const reg_t &qubits, RngEngine &rng) {
  // Simulate unobserved measurement
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Apply update to reset state
  measure_reset_update(qubits, 0, meas.first, meas.second);
}

template <class statevec_t>
std::pair<uint_t, double>
State<statevec_t>::sample_measure_with_prob(const reg_t &qubits,
                                            RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);
  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class statevec_t>
void State<statevec_t>::measure_reset_update(const std::vector<uint_t> &qubits,
                                             const uint_t final_state,
                                             const uint_t meas_state,
                                             const double meas_prob) {
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement
  // final_state

  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    BaseState::qreg_.apply_diagonal_matrix(qubits, mdiag);

    // If it doesn't agree with the reset state update
    if (final_state != meas_state)
      BaseState::qreg_.apply_mcx(qubits);
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    BaseState::qreg_.apply_diagonal_matrix(qubits, mdiag);

    // If it doesn't agree with the reset state update
    // This function could be optimized as a permutation update
    if (final_state != meas_state) {
      // build vectorized permutation matrix
      cvector_t perm(dim * dim, 0.);
      perm[final_state * dim + meas_state] = 1.;
      perm[meas_state * dim + final_state] = 1.;
      for (size_t j = 0; j < dim; j++) {
        if (j != final_state && j != meas_state)
          perm[j * dim + j] = 1.;
      }
      // apply permutation to swap state
      apply_matrix(qubits, perm);
    }
  }
}

template <class statevec_t>
std::vector<SampleVector> State<statevec_t>::sample_measure(const reg_t &qubits,
                                                            uint_t shots,
                                                            RngEngine &rng) {
  uint_t i;
  // Generate flat register for storing
  std::vector<double> rnds;
  rnds.reserve(shots);
  reg_t allbit_samples(shots, 0);

  for (i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));

  allbit_samples = BaseState::qreg_.sample_measure(rnds);

  // Convert to SampleVector format
  int_t npar = BaseState::threads_;
  if (npar > shots)
    npar = shots;
  std::vector<SampleVector> all_samples(shots, SampleVector(qubits.size()));

  auto convert_to_bit_lambda = [this, &allbit_samples, &all_samples, shots,
                                qubits, npar](int_t i) {
    uint_t ishot, iend;
    ishot = shots * i / npar;
    iend = shots * (i + 1) / npar;
    for (; ishot < iend; ishot++) {
      SampleVector allbit_sample;
      allbit_sample.from_uint(allbit_samples[ishot], qubits.size());
      all_samples[ishot].map(allbit_sample, qubits);
    }
  };
  Utils::apply_omp_parallel_for((npar > 1), 0, npar, convert_to_bit_lambda,
                                npar);

  return all_samples;
}

template <class statevec_t>
void State<statevec_t>::apply_initialize(const reg_t &qubits,
                                         const cvector_t &params_in,
                                         RngEngine &rng) {
  auto sorted_qubits = qubits;
  std::sort(sorted_qubits.begin(), sorted_qubits.end());
  // apply global phase here
  cvector_t tmp;
  if (BaseState::has_global_phase_) {
    tmp.resize(params_in.size());
    auto apply_global_phase = [&tmp, &params_in, this](int_t i) {
      tmp[i] = params_in[i] * BaseState::global_phase_;
    };
    Utils::apply_omp_parallel_for(
        (qubits.size() > (uint_t)omp_qubit_threshold_), 0, params_in.size(),
        apply_global_phase, BaseState::threads_);
  }
  const cvector_t &params = tmp.empty() ? params_in : tmp;
  if (qubits.size() == BaseState::qreg_.num_qubits()) {
    // If qubits is all ordered qubits in the statevector
    // we can just initialize the whole state directly
    if (qubits == sorted_qubits) {
      initialize_from_vector(params);
      return;
    }
  }
  // Apply reset to qubits
  apply_reset(qubits, rng);

  // Apply initialize_component
  BaseState::qreg_.initialize_component(qubits, params);
}

template <class statevec_t>
void State<statevec_t>::initialize_from_vector(const cvector_t &params) {
  BaseState::qreg_.initialize_from_vector(params);
}

//=========================================================================
// Implementation: Multiplexer Circuit
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_multiplexer(const reg_t &control_qubits,
                                          const reg_t &target_qubits,
                                          const std::vector<cmatrix_t> &mmat) {
  // (1) Pack vector of matrices into single (stacked) matrix ... note: matrix
  // dims: rows = DIM[qubit.size()] columns = DIM[|target bits|]
  cmatrix_t multiplexer_matrix = Utils::stacked_matrix(mmat);

  // (2) Treat as single, large(r), chained/batched matrix operator
  apply_multiplexer(control_qubits, target_qubits, multiplexer_matrix);
}

//=========================================================================
// Implementation: Kraus Noise
//=========================================================================
template <class statevec_t>
void State<statevec_t>::apply_kraus(const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats,
                                    RngEngine &rng) {
  // Check edge case for empty Kraus set (this shouldn't happen)
  if (kmats.empty())
    return; // end function early

  // Choose a real in [0, 1) to choose the applied kraus operator once
  // the accumulated probability is greater than r.
  // We know that the Kraus noise must be normalized
  // So we only compute probabilities for the first N-1 kraus operators
  // and infer the probability of the last one from 1 - sum of the previous

  double r = rng.rand(0., 1.);
  double accum = 0.;
  double p;
  bool complete = false;

  // Loop through N-1 kraus operators
  for (size_t j = 0; j < kmats.size() - 1; j++) {

    // Calculate probability
    cvector_t vmat = Utils::vectorize_matrix(kmats[j]);

    p = BaseState::qreg_.norm(qubits, vmat);
    accum += p;
    // check if we need to apply this operator
    if (accum > r) {
      // rescale vmat so projection is normalized
      Utils::scalar_multiply_inplace(vmat, 1 / std::sqrt(p));
      // apply Kraus projection operator
      apply_matrix(qubits, vmat);
      complete = true;
      break;
    }
  }

  // check if we haven't applied a kraus operator yet
  if (complete == false) {
    // Compute probability from accumulated
    complex_t renorm = 1 / std::sqrt(1. - accum);
    auto vmat = Utils::vectorize_matrix(renorm * kmats.back());
    apply_matrix(qubits, vmat);
  }
}

//-------------------------------------------------------------------------
} // namespace Statevector
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
