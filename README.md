## Spin-spin decoherence
This code simulates the dynamics of a closed spin chain using the Bethe ansatz for an XXX chain model. There is a figure of merit comparing the simulations for different implementations of the algorithms. The times are ordered like the legend of the figure `final_plot.png`
### Experiment
The simulated experiment is the spin echo experiment. The observable is the $x$ component of the magnetization of the rotated spin.
### Algorithm
The simulation has been run using two algorithms:
1. Using the _**analytical solution**_, which required exponential resources as the amount of particles in the chain increases.
2. Using a polinomially expensive algorithm in a _**truncated Hilbert space**_ where the available excitations are considered to be limited given an external magnetic field that is strong enough. You can find more details about this algorithm in the paper by [A. Lunghi and S. Sanvito, *Electronic Spin-Spin Decoherence Contribution in Molecular Qubits by Quantum Unitary Dynamics*, Journal of magnetism and Magnetic materials **487** (2019)](https://doi.org/10.1016/j.jmmm.2019.165325)

### Variables and results
After running the codes there are some clear results:
1. Increase of the decoherence due to:
* More spins in the chain
* Longer distance interaction (which strength decay can be tuned)
* Stronger spin-spin interaction
2. The implemented algorithm improves significantly the performance when the analytical solution begins to compile too slowly. Considering 2 excitations is usually enough to have almost no numerical error in the simulation, and compiles around 10 times faster.


#### Future work
The algorithm can be personalized even further:
* Considering an even more truncated hilbert space, with excitations only close to the rotated spin in the spin chain
* Comparing how the distance and magnitude of the interaction constant affects the decoherence in a quantitative way
* Instead of using the parameters for an isotropic material and an XXX spin chain, the g-factor and dipolar interaction matrices can be tuned to simulate real molecules or lattices.
