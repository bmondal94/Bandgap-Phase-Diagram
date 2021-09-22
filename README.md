# Bandgap Phase Diagram
Bandgap phase diagram for ternary and higher order semiconductor materials.
* https://bmondal94.github.io/Bandgap-Phase-Diagram/

## Computational setup
* Periodic DFT using VASP-5.4.4
* Geometry optimization: PBE-D3 (BJ), PAW basis set 
* Electronic properties: m-BJ, PAW basis set, spin-orbit coupling 
* Super cell : 6x6x6, 10 SQS [[1]](https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/node74.html), Γ-only, band unfolding [[2]](https://github.com/rubel75/fold2Bloch-VASP),[[3]](https://github.com/band-unfolding/bandup)

