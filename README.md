# Bandgap Phase Diagram
Bandgap phase diagram for semiconductor materials. For details visit here:

* https://bmondal94.github.io/Bandgap-Phase-Diagram/

   <img src="./ImageFolder/BandgapPhaseDiagram.png" style="width:300px;height:300px;">

## Introduction
The tuning of the type and size of bandgaps of III-V semiconductors is a major goal for optoelectronic applications. Varying the relative composition of several III- or V-components in compound semiconductors is one of the major approaches here. Alternatively, straining the system can be used to modify the bandgaps. By combining these two approaches, bandgaps can be tuned over a wide range of values, and direct or indirect semiconductors can be designed. However, an optimal choice of composition and strain to a target bandgap requires complete material-specific composition, strain, and bandgap knowledge. Exploring the vast chemical space of all possible combinations of III- and V-elements with variation in composition and strain is experimentally not feasible. We thus developed a density-functional-theory-based predictive computational approach for such an exhaustive exploration. This enabled us to construct the 'bandgap phase diagram' by mapping the bandgap in terms of its magnitude and nature over the whole composition-strain space. Further, we have developed efficient machine-learning models to accelerate such mapping in multinary systems. We show the application and great benefit of this new predictive mapping on device design. 

## General computational setup
General computational setup.

* Periodic DFT using VASP-5.4.4
* Geometry optimization: PBE-D3 (BJ), PAW basis set 
* Electronic properties: m-BJ, PAW basis set, spin-orbit coupling 
* Super cell : 6x6x6, 10 SQS [[1]](https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/node74.html), Γ-only, band unfolding [[2]](https://github.com/rubel75/fold2Bloch-VASP),[[3]](https://github.com/band-unfolding/bandup)
* SVM(rbf) machine learning models

## References
* III-V semiconductors bandgap phase diagram
    *  Binary compounds: [arXiv](http://arxiv.org/abs/2208.10596), [Mondal et. al, Phys. Scr., (2023)](https://doi.org/10.1088/1402-4896/acd08b), [NOMAD repository](https://doi.org/10.17172/NOMAD/2022.08.20-2)
    *  Ternary compounds: [arXiv](http://arxiv.org/abs/2302.14547), [NOMAD repository](https://doi.org/10.17172/NOMAD/2023.02.27-1)
    *  Quaternary compounds
        *  GaAsPSb system: [arXiv](https://doi.org/10.48550/arXiv.2305.03666), [NOMAD repository](https://doi.org/10.17172/NOMAD/2023.05.03-1)
    
## License
* [MIT License](LICENSE)

Please contact to [Badal Mondal](mailto:badalmondal.chembgc@gmail.com,badal.mondal@physik.uni-marburg.de,badal.mondal@studserv.uni-leipzig.de) for further details.
