import os, uuid, yaml
import numpy as np
from argparse import ArgumentParser
from CondorJobSubmitter import CondorJobSubmitter

def write_job_script(configpath, scriptdir, outdir, number_runs, imagepath):

    if not os.path.exists(scriptdir):
        os.makedirs(scriptdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    scriptpath = os.path.join(scriptdir, str(uuid.uuid4()) + ".sh")

    sceleton = """#!/bin/bash
export ROOTDIR={rootdir}
singularity exec -H ${{_CONDOR_SCRATCH_DIR}} -B {rootdir} -B {outdir} -B {configpath} --cleanenv --env ROOTDIR="{rootdir}" {imagepath} python3.10 {rootdir}/notebooks/pymc_batch/run_mcmc.py --outdir {outdir} --config {configpath} --number_runs {number_runs}
"""
    
    with open(scriptpath, 'w') as scriptfile:
        scriptfile.write(sceleton.format(rootdir = os.environ["ROOTDIR"],
                                         configpath = os.path.realpath(os.path.abspath(configpath)),
                                         outdir = os.path.realpath(os.path.abspath(outdir)),
                                         imagepath = os.path.realpath(os.path.abspath(imagepath)),
                                         number_runs = number_runs))

    return scriptpath

def RunMCMCCampaign(campaigndir, dryrun, sigfrac, sig_mean, dset_length, number_runs, imagepath):

    if not os.path.exists(campaigndir):
        os.makedirs(campaigndir)

    configpath = os.path.join(campaigndir, "config.yaml")
    with open(configpath, 'w') as configfile:
        configfile.write(yaml.dump({"sigfrac": sigfrac, 
                                    "sig_mean": sig_mean, 
                                    "dset_length": dset_length}
                               ))

    job = write_job_script(configpath, 
                           scriptdir = os.path.join(campaigndir, "submit"),
                           outdir = os.path.join(campaigndir, "output"),
                           imagepath = imagepath,
                           number_runs = number_runs)

    CondorJobSubmitter.submit_job(job, opts = {"request_cpus": "2", "+queue": "\"short\""}, 
                                  dryrun = dryrun)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--campaigndir", action = "store", dest = "campaigndir")
    parser.add_argument("--dryrun", action = "store_true", dest = "dryrun")
    parser.add_argument("--imagepath", action = "store", dest = "imagepath")
    args = vars(parser.parse_args())

    sigfrac_truth = 0.2
    sig_means = np.linspace(-1.0, 3.0, 18)

    number_jobs_per_point = 40
    number_runs_per_job = 10
    dset_length = 100
    
    for sig_mean in sig_means:
        cur_campaigndir = os.path.join(args["campaigndir"], str(uuid.uuid4()))
        for cur in range(number_jobs_per_point):
            RunMCMCCampaign(campaigndir = cur_campaigndir, dryrun = args["dryrun"], 
                            sigfrac = sigfrac_truth, sig_mean = float(sig_mean), dset_length = dset_length,
                            imagepath = args["imagepath"], number_runs = number_runs_per_job)
