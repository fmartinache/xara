XARA: a python package for eXtreme Angular Resolution Astronomy
===============================================================

It is a python module desgigned to create, extract and manipulate Kernel-phase
data structures, using the theory presented by 

- Martinache (2010), ApJ, 724, 464.  
- Martinache (2013), PASP, 125, 422.

The module is constructed around two main classes:
-------------------------------------------------

- KPI: Kernel-phase Information: object that encodes the information defined by
  the linear model that describes the optical system of interest.

- KPO: Kernel-phase Object: the data structure that, in addition to
  a KPI data structure, also contains Ker-phase data extracted
  from actual images, using the model contained within the KPI object,
  along with some additional information extracted from the data fits
  files headers.


