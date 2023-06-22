import os, time
import subprocess as sp

class CondorJobSubmitter:

    # create the .submit file and submit it to the batch
    @staticmethod
    def submit_job(job_script_path, job_threshold = 400, opts = {}, dryrun = False):

        job_script_id = os.path.splitext(job_script_path)[0]

        # check what kind of file we got
        if not os.path.splitext(job_script_path)[1] == ".submit":
            # need to create the submit file
            job_script_base, _ = os.path.splitext(job_script_path)
            job_dir = os.path.dirname(job_script_path)
            submit_file_path = job_script_base + ".submit"

            while True:
                try:
                    with open(submit_file_path, 'w') as submit_file:
                        submit_file.write("executable = " + job_script_path + "\n")
                        submit_file.write("universe = vanilla\n")
                        submit_file.write("notification = never\n")
                        submit_file.write(f"output = {job_script_id}.out\n")
                        submit_file.write(f"error = {job_script_id}.err\n")
                        submit_file.write(f"log = {job_script_id}.log\n")

                        for cur_opt_name, cur_opt_val in opts.items():
                            submit_file.write("{} = {}\n".format(cur_opt_name, cur_opt_val))

                        submit_file.write("queue 1")
                
                    break
                except:
                    print("problem writing job script -- retrying")
                    time.sleep(10)

        else:
            # are given the submit file directly
            submit_file_path = job_script_path

        while True:
            running_jobs = CondorJobSubmitter.queued_jobs()
            if running_jobs < job_threshold:
                break
            print("have {} jobs running - wait a bit".format(running_jobs))
            time.sleep(30)

        while True:
            try:
                # call the job submitter
                if not dryrun:
                    sp.check_output(["condor_submit", submit_file_path])
                print("submitted '" + submit_file_path + "'")
                break
            except:
                print("Problem submitting job - retrying in 10 seconds!")
                time.sleep(10)

    @staticmethod
    def queued_jobs(queue_status = "condor_q"):
        while True:
            try:
                running_jobs = len(sp.getoutput([queue_status]).split('\n')) - 6
                return running_jobs
            except sp.CalledProcessError:
                print("{} error - retrying!".format(queue_status))
                time.sleep(10)
