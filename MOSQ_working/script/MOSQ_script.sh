output_DIR="./data/"
MOSQ_src_DIR="../src_code_MOSQ/"
cpp_src_DIR="../qiskit-aer/src/"
py_src_DIR="../MOSQ_venv/lib/python3.8/site-packages/qiskit_aer/"

#copy MOSQ source codes
cp ${MOSQ_src_DIR}_cobyla_py.py ${py_src_DIR}../scipy/optimize/
cp ${MOSQ_src_DIR}indexes.hpp ${cpp_src_DIR}simulators/statevector/
cp ${MOSQ_src_DIR}operations.hpp ${cpp_src_DIR}framework/
cp ${MOSQ_src_DIR}qubitvector.hpp ${cpp_src_DIR}simulators/statevector/
cp ${MOSQ_src_DIR}statevector_state.hpp ${cpp_src_DIR}simulators/statevector/
cp ${MOSQ_src_DIR}state.hpp ${cpp_src_DIR}simulators/

#baseline results
cp ${MOSQ_src_DIR}fusion_OFF.hpp ${cpp_src_DIR}transpile/fusion.hpp

cd ../qiskit-aer/build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j
cd ..
python3 setup.py bdist_wheel
pip uninstall qiskit-aer -y
pip install dist/*.whl
cd ../MOSQ_working

cp ${MOSQ_src_DIR}aer_compiler.py ${py_src_DIR}backends/
cp ${MOSQ_src_DIR}estimator_v2.py ${py_src_DIR}primitives/

for molecule in NH3 #H2 LiH BeH2 NH3 CH4 LiF MgH2 N2H2 CH3N C2H4 CO2 C2H6
do
    echo "Baseline" > "${output_DIR}Baseline_${molecule}.txt"
    python3 ./code/Baseline.py $molecule >> "${output_DIR}Baseline_${molecule}.txt"
done

#fusion results
cp ${MOSQ_src_DIR}fusion_ON.hpp ${cpp_src_DIR}transpile/fusion.hpp

cd ../qiskit-aer/build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j
cd ..
python3 setup.py bdist_wheel
pip uninstall qiskit-aer -y
pip install dist/*.whl
cd ../MOSQ_working

cp ${MOSQ_src_DIR}aer_compiler.py ${py_src_DIR}backends/
cp ${MOSQ_src_DIR}estimator_v2.py ${py_src_DIR}primitives/

for molecule in NH3 #H2 LiH BeH2 NH3 CH4 LiF MgH2 N2H2 CH3N C2H4 CO2 C2H6
do
    echo "Fusion" > "${output_DIR}Fusion_${molecule}.txt"
    python3 ./code/Fusion.py $molecule >> "${output_DIR}Fusion_${molecule}.txt"
done

#MOSQ results
cp ${MOSQ_src_DIR}fusion_OFF.hpp ${cpp_src_DIR}transpile/fusion.hpp

cd ../qiskit-aer/build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j
cd ..
python3 setup.py bdist_wheel
pip uninstall qiskit-aer -y
pip install dist/*.whl
cd ../MOSQ_working

cp ${MOSQ_src_DIR}aer_compiler.py ${py_src_DIR}backends/
cp ${MOSQ_src_DIR}estimator_v2.py ${py_src_DIR}primitives/

for molecule in NH3 #C2H6 #H2 LiH BeH2 NH3 CH4 LiF MgH2 N2H2 CH3N C2H4 CO2 C2H6
do
    echo "MOSQ" > "${output_DIR}MOSQ_${molecule}.txt"
    python3 ./code/MOSQ.py $molecule >> "${output_DIR}MOSQ_${molecule}.txt"
done