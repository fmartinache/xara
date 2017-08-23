XARA: a python package for eXtreme Angular Resolution Astronomy
===============================================================

It is a python module designed to create, extract and manipulate Kernel-phase
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


The modules comes with a utility program
----------------------------------------

ker_model_builder is a GUI (based on PyQt4 and pyqtgraph) that is handy to
design discrete models for circular apertures featuring either offset spider
vanes or central obstruction.

Unlike the module itself, this program is optional: as a consequence, the
required modules for its proper execution: PyQt4 and pyqtgraph are not
included, and should be installed manually.

Note:
----

Than on some systems, after installation is complete, the call for the
"ker_model_builder" command may fail due to apparent permission
limitations. The exact error message encountered is:

"IOError: [Errno 13] Permission denied: '/usr/local/lib/python2.7/dist-packages/xara-0.1-py2.7.egg/EGG-INFO/requires.txt'"

The current fix is to set the permissions right on this requirement file:

"sudo chmod a+r /usr/local/lib/python2.7/dist-packages/xara-0.1-py2.7.egg/EGG-INFO/requires.txt"

