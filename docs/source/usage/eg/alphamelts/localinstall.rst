Local Installation of alphaMELTS
----------------------------------

pyrolite can download and manage its own version of alphaMELTS (without any real
'installation', *per-se*), and use this for :mod:`~pyrolite.util.alphamelts.automation`
purposes.

.. literalinclude:: ../../../../_examples/alphamelts/install.py
  :language: python

.. warning:: This 'local install' method still requires that you have Perl installed,
          as it uses the Perl :code:`run_alphamelts.command` script. If you're on
          Windows, you can use `Strawberry Perl <http://strawberryperl.com/>`__
          for this purpose.


.. seealso::

  Examples:
    `alphaMELTS Environment Configuration <environment.html>`__,
    `Automating alphaMELTS Runs <automation.html>`__,
    `Handling Outputs from Melts: Tables <tables.html>`__,
    `Compositional Uncertainty Propagation for alphaMELTS Experiments <montecarlo.html>`__
