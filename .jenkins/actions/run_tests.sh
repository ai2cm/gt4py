#!/bin/bash
envloc=$1
RUN_CMD_FILE=$2
maxsleep=9000
if [ "`which sbatch`" != "" ] ; then
. ${envloc}/env/machineEnvironment.sh
# check if SLURM script exists
script="${envloc}/env/submit.${host}.slurm"
test -f ${script} || exitError 1252 ${LINENO} "cannot find script ${script}"


# load slurm tools
if [ ! -f ${envloc}/env/slurmTools.sh ] ; then
    exitError 1203 ${LINENO} "could not find ${envloc}/env/slurmTools.sh"
fi
. ${envloc}/env/slurmTools.sh


# setup SLURM job
out="gt4py_tests_${BUILD_ID}.out"
/bin/sed -i 's|<NAME>|jenkins-gt4py-tests|g' ${script}
/bin/sed -i 's|<NTASKS>|1|g' ${script}
/bin/sed -i 's|<NTASKSPERNODE>|'"${nthreads}"'|g' ${script}
/bin/sed -i 's|<CPUSPERTASK>|1|g' ${script}
/bin/sed -i 's|<OUTFILE>|'"${out}"'|g' ${script}
/bin/sed -i -e '/<CMD>/ r ${RUN_CMD_FILE}' ${script}
/bin/sed -i 's|<CMD>|'""'|g' ${script}
/bin/sed -i 's|<PARTITION>|'"cscsci"'|g' ${script}

# submit SLURM job
launch_job ${script} ${maxsleep}
if [ $? -ne 0 ] ; then
  exitError 1251 ${LINENO} "problem launching SLURM job ${script}"
fi

# echo output of SLURM job
cat ${out}
rm ${out}

else
   while read in; do $in; done < ${RUN_CMD_FILE}
fi
