XARA: a python package for eXtreme Angular Resolution Astronomy
===============================================================

It is a python module designed to create, extract and manipulate Kernel-phase
data structures, using the theory introduced by:

- Martinache (2010), ApJ, 724, 464 
- Martinache (2013), PASP, 125, 422.

and more recently refined by: Martinache et al (2020), A&A, 636, A72

Links to these papers:
- https://ui.adsabs.harvard.edu/abs/2010ApJ...724..464M
- https://ui.adsabs.harvard.edu/abs/2013PASP..125..422M
- https://doi.org/10.1051/0004-6361/201936981

Tutorial/documentation
---------------

A growing documentation featuring several tutorial examples and a fair amount
of howto explanation is available here: http://frantzmartinache.eu/xara_doc/

Acknowledgement
---------------

XARA is a development carried out in the context of the KERNEL project. KERNEL
has received funding from the European Research Council (ERC) under the
European Union's Horizon 2020 research and innovation program (grant agreement
CoG - 683029). For more information about the KERNEL project, visit:
http://frantzmartinache.eu/index.php/category/kernel/

Recommandation for installation:
-------------------------------

>> python setup.py install --user


It is also recommended to add ~/.local/bin/ to the path:

>> export PATH=$HOME/.local/bin/:$PATH

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

If installing the module using super user privileges, on some systems,
after installation is complete, the call for the "ker_model_builder"
command may fail due to apparent permission limitations when not using
super user privileges. The exact error message encountered is:

"IOError: [Errno 13] Permission denied: '/usr/local/lib/python2.7/dist-packages/xara-0.1-py2.7.egg/EGG-INFO/requires.txt'"

The current fix is to set the permissions right on this requirement file:

"sudo chmod a+r /usr/local/lib/python2.7/dist-packages/xara-0.1-py2.7.egg/EGG-INFO/requires.txt"

The other option is to use the recommended installation mode described above!
