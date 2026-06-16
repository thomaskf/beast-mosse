#!/bin/bash
# Reproduce the MoSSE positive-likelihood bug at a single fixed state.
# EvalState parses the sim8 XML (data + model + tree), overrides the tree
# heights + the 10 parameters from the given state file, then evaluates the
# posterior/likelihood N times in one JVM and prints the result.
#
# REQUIRES: BEAST 2.7 + the beast-mosse package + the native JNI lib (libtest,
# built on FFTW). On Gadi this is provided by the modules + env.sh:
#     source /scratch/hu35/txw813/mosse_no_punc/env.sh
#   (loads beast/2.7.6, fftw3, gsl; sets LD_LIBRARY_PATH so libtest.so is found)
# On another machine, set the classpath to your BEAST + mosse jars and make sure
# the native lib (libtest.so / libtest.dylib) is on java.library.path / LD_LIBRARY_PATH.
set -e

# Build the classpath from the BEAST install + installed packages (Gadi layout).
# Adjust these globs for your machine if different.
CP=$(echo /apps/beast/2.7.6/lib/*.jar ~/.beast/2.7/*/lib/*.jar 2>/dev/null | tr ' ' ':')
if [ -z "$CP" ] || [ "$CP" = ":" ]; then
  echo "!! Could not find BEAST jars. Edit CP in this script to point to your"
  echo "   BEAST 2.7 lib + beast-mosse package jars (must include mosse.jar, jblas)."
  exit 1
fi

echo "Classpath: $CP"
javac -cp "$CP" EvalState.java

echo
echo "==== STATE_1600 (divergent: diffusion=0.003375, dt=0.003; sd=sqrt(dt*diff)~0.0032 << dx~0.028) ===="
echo "     EXPECT a POSITIVE / garbage logP (the bug). Logged value was +897.63."
echo "     Across separate JVM launches it varies in {+897, -5296, -5949, -7680, ...} (FFTW-plan dependent)."
java -cp "$CP:." -Dmosse.threads=8 EvalState sim8_seq1_0.001_73.xml state_1600.txt 5

echo
echo "==== STATE_1500 (healthy, 100 samples earlier) ===="
echo "     EXPECT a normal negative logP (logged value was -3027 likelihood)."
java -cp "$CP:." -Dmosse.threads=8 EvalState sim8_seq1_0.001_73.xml state_1500.txt 3
