CovEst
======

Tool that estimates coverage (and genome size) of dna sequence from
reads.

.. image:: https://badge.fury.io/py/covest.svg
    :target: https://badge.fury.io/py/covest
.. image:: https://travis-ci.org/mhozza/covest.svg?branch=master
    :target: https://travis-ci.org/mhozza/covest

Important
---------

Prediction models have been renamed in version 0.5.9 to match names in publications.

Changes are as follows:

Basic -> E

Repeats -> RE

Basic_Polymorphism -> EP

Repeats_Polymorphism_Equal -> ERNPE

Requirements
------------
- python 3.4+
- python3-dev
- gcc

Installation:
------------
``pip install -e .`` from the project directory

Usage
-----

type ``covest --help`` for the usage.

Basic Usage:
~~~~~~~~~~~~
``covest histogram -m model -k K -r read_length``

-  You can specify the read file using ``-s reads.fa`` parameter for more precise genome size computation.
-  default *K* is 21
-  default *read length* is 100
-  currently, the supported models are:

   -  basic: for simple genomes without repeats
   -  repeat: for genomes with repetitive sequences

Input Histogram Specification:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The input histogram can be generated from the read data using `jellyfish <http://www.cbcb.umd.edu/software/jellyfish/>`__.

-  ``jellyfish count -m K -C reads.fa -o reads_table.jf -s 1000000000``
-  ``jellyfish histo table.jf -o reads.hist``

The format of the histogram is just list of lines. Each lines contains an index and value separated by space.

Output Specification:
~~~~~~~~~~~~~~~~~~~~~
CovEst outputs it's results in simple subset of YAML format for best human readability and possibility of machine processing.

The output are lines containing ``key: value``. The most important keys are ``coverage`` and ``genome_size`` (or ``genome_size_reads`` if reads size was specified).

Other included tools
--------------------

-  ``geset.py`` tool for estimation genome size from reads size and known
   coverage
-  ``reads_size.py`` tool for computation of the total reads size
-  ``kmer_hist.py`` custom khmer histogram computation, it is much slower than other tools, so use it only if you have no other option.
-  ``read_sampler.py`` script for subsampling reads, useful if you have very high coverage data and want to make it smaller.
-  ``fasta_length.py`` get total length of all sequences in fasta file.

Copyright and citation
----------------------
This section is applicable to original CovEst (0.5.6) by M. Hozza.

Original CovEst is licenced under `GNU GPLv3 <http://www.gnu.org/licenses/gpl-3.0.en.html>`__ license.

CovEst is research software, so you should cite us when you use it in scientific publications!
   Hozza, M., Vinař, T., & Brejová, B. (2015, September). How Big is that Genome? Estimating Genome Size and Coverage from k-mer Abundance Spectra. In String Processing and Information Retrieval (pp. 199-209). Springer International Publishing.
