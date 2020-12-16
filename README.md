# FTS_polyampholyte_water
The field theoretic simulation (FTS) code for simulating a system of an polyampholyte plus a small molecule that can be for explict solvent (water) or salt ions.

The code is based on the theory described in  
McCarty J et al, J Phys Chem Lett 10 1644 (2019)  
Lin Y et al, eLife 8 e42571 (2019)  
Danielsen S P O et al, PNAS 116 8224 (2019)  
Pal T, Wessén J, Das S, & Chan H S, arXiv:2006.12776 (2020)  

ver0: polyampholyte and small molecule must be neutral. 

Direct single sequence canonical ensemble calculation:

<tt>python3 FTS_polyampholyte_water [seq name]</tt>

A folder below directory <tt>./results/</tt> will be constrcuted and all field snapshots will be stored in the folder as .txt files

To analyze the field snapshots, e.g. calculating sysytem mass densities and charge densities, do:

<tt>python3 CL_io_control [directory of the field snapshots] </tt>


