# Bandgap Phase Diagram
Bandgap phase diagram for semiconductor materials. For details visit here:
* https://bmondal94.github.io/Bandgap-Phase-Diagram/

   <img src="./ImageFolder/BandgapPhaseDiagram.png" style="width:300px;height:300px;">

## Computational setup
General computational setup.
* Periodic DFT using VASP-5.4.4
* Geometry optimization: PBE-D3 (BJ), PAW basis set 
* Electronic properties: m-BJ, PAW basis set, spin-orbit coupling 
* Super cell : 6x6x6, 10 SQS [[1]](https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/node74.html), Γ-only, band unfolding [[2]](https://github.com/rubel75/fold2Bloch-VASP),[[3]](https://github.com/band-unfolding/bandup)

## References
* III-V semiconductors bandgap phase diagram
    *  Binary compounds: [arxIv](http://arxiv.org/abs/2208.10596), [NOMAD repository](https://doi.org/10.17172/NOMAD/2022.08.20-2)

Others are work in progress. Please contact to [Badal Mondal](mailto:badalmondal.chembgc@gmail.com).
