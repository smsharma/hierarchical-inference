import os, uuid, sys
from argparse import ArgumentParser
sys.path.append(os.environ["ROOTDIR"])
from notebooks.pymc_batch.CondorJobSubmitter import CondorJobSubmitter

def write_job_script(scriptdir, outdir, imagepath):

    if not os.path.exists(scriptdir):
        os.makedirs(scriptdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    scriptpath = os.path.join(scriptdir, str(uuid.uuid4()) + ".sh")

    sceleton = """#!/bin/bash
export ROOTDIR={rootdir}
singularity exec -H ${{_CONDOR_SCRATCH_DIR}} -B {rootdir} -B {outdir} --cleanenv --env ROOTDIR="{rootdir}" {imagepath} python3.6 {rootdir}/notebooks/pytorch_batch/train_deepset.py --outdir {outdir}
"""

    with open(scriptpath, 'w') as scriptfile:
        scriptfile.write(sceleton.format(rootdir = os.environ["ROOTDIR"],
                                         imagepath = os.path.abspath(imagepath),
                                         outdir = os.path.abspath(outdir)))

    return scriptpath

def RunTrainingCampaign(campaigndir, imagepath, dryrun):

    if not os.path.exists(campaigndir):
        os.makedirs(campaigndir)

    job = write_job_script(scriptdir = os.path.join(campaigndir, "submit"),
                           outdir = os.path.join(campaigndir, "output"),
                           imagepath = imagepath)

    CondorJobSubmitter.submit_job(job, opts = {"request_cpus": "1", "request_gpus": "1"}, 
                                dryrun = dryrun)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--campaigndir", action = "store", dest = "campaigndir")
    parser.add_argument("--dryrun", action = "store_true", dest = "dryrun")
    parser.add_argument("--imagepath", action = "store", dest = "imagepath")
    args = vars(parser.parse_args())
    
    RunTrainingCampaign(**args)
