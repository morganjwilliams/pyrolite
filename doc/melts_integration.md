
# An alphaMELTS Wrapper

alphaMELTS programs are installed, instantiated and updated using the Perl script `run_alphamelts.command`, `install.command` and `update.command` files, which are together the source of the Perl dependency for Melts. Perl can optionally be installed using conda (`conda install perl`), which will provide a much more reproducible environment than binary distributions from the [Perl website](http://www.perl.org/get.html).


### Melts (doesn't need to) Install

The melts install process serves only to make the executables findable, but each of the functions can be achieved directly with the executables.
The  `install.command` script makes shortcuts to the MELTS program, and tries to add the folder to the user's `PATH` environment variable. The installed link files are executable batch file (`.bat`) wrappers around the alphaMELTS scripts. Once the path is set, they should be findable with `where`: e.g. `where alphamelts` should return `<install_dir>\alphamelts.bat`.

TODO: provide a similar function by writing the location information to a simple text file.

### Input Files

The `file_format.command` script is principally used to fix incorrect line endings in input files. It is used in the manner of `file_format.command <melts_file_1.melts, ..., melts_file_n.melts>` or alternatively on all files as `file_format.command *.melts`.

Melts files (`.melts`) are simply text data files with the following format:
```
Title: Allan et al. 1989 (see Ghiorso 1997)
Initial Composition: SiO2 48.68
Initial Composition: TiO2 1.01
...
Initial Composition: H2O 0.20
Initial Temperature: 1200.00
Final Temperature: 1000.00
Increment Temperature: 3.00
Initial Pressure: 500.00
Final Pressure: 500.00
Increment Pressure: 0.00
dp/dt: 0.00
log fo2 Path: None
Mode: Fractionate Solids
```

#### Settings File

#### Enrichment File

#### Table File

#### Trace Data File

### Running alphaMELTS

The call signature of alphaMELTS is as follows:

```
run_alphamelts.command [-h] [-p output_path] [-f settings_file] [-b batch_file | -a] [-m melts_file] [-o output_file] [-l log_file]
```

#### Environment Variables

Note that `run_alphamelts.command` only passes your chosen settings to alphamelts indirectly – either by setting environment variables or by making entries in text files. If you are in batch or automatic mode it will redirect the `alphamelts` standard input by executing `alphamelts < batch_file`, which passes the `melts_file` name. Otherwise, the user must still prompt `alphamelts` to read in the relevant text or binary files (even when the `-m` switch is used); a warning will be given if no file has been input.

Here the most practical approach would be to manage the environment, e.g. using `environs` (`pip install environs`).

#### Batch Files
Under the alphaMELTS formulation, batch files are simply a list of sequential arguments which would be passed to the executable.  Automatic mode generates a similar file, ‘auto_batch.txt’, for simple calculations (no adjusting of parameters) that can be renamed and modified for later use.

### Output Files

Text file output is written automatically with one or more lines for each of the calculated equilibria along the thermodynamic path. The values are recorded at each step after equilibrium conditions are achieved and just before any fractionations (i.e. melt extraction or solid fractionation). A title, taken from the original melts_file, is printed first; six space-delimited files with names ending in `_tbl.txt` are written.
