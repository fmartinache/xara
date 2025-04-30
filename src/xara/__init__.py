#!/usr/bin/env python

'''------------------------------------------------------------------
                XARA: Extreme Angular Resolution Astronomy
    ------------------------------------------------------------------
    ---
    XARA is a python module to create, and extract Kernel-phase data
    structures, using the theory of Martinache, 2010, ApJ, 724, 464.
    ----

    The module is constructed around two main classes:
    -------------------------------------------------

    - KPI: Kernel-Phase Information

      An object packing the data structures that guide the
      interpretation of images from an inteferometric point of view,
      leading to applications like kernel-phase and/or wavefront
      sensing

    - KPO: Kernel-Phase Observation

      An object that contains a KPI along with optional data extracted
      from the Fourier transform of images, using the KPI model and a
      handful additional pieces of information: wavelength, pixel scale,
      detector position angle and epoch to enable their interpretation
      ---------------------------------------------------------------- '''

__version__ = "1.0.0"

from xara import (
    core as core,
    fitting as fitting,
    iwfs as iwfs,
    kpi as kpi,
    kpo as kpo,
)
