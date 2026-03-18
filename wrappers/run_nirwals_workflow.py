#!/usr/bin/env python

import os
import json
import importlib
from astropy.io import fits
import glob

def load_json(path):
    with open(path) as f:
        return json.load(f)
    

def fix_primary_header_keys(work_dir):
    '''
    nirwalsreduce.py expects header "AR-ANGLE" that does not exist in our raw data files. 
    Manually replace with header "CAMANG", which provides the same information
    '''

    product_dir = os.path.join(work_dir, 'nirwals', 'product')
    product_files = glob.glob(os.path.join(product_dir, '*.fits'))
    
    for product_file in product_files:
        basename = os.path.basename(product_file)
        
        try:
            with fits.open(product_file, mode='update') as hdul:
                header = hdul['PRIMARY'].header
                header_keys = list(header.keys())
                
                # add AR-ANGLE if missing (use CAMANG value)
                if 'AR-ANGLE' not in header_keys and 'CAMANG' in header_keys:
                    header['AR-ANGLE'] = (header['CAMANG'], 'Articulation angle [degrees] (copied from CAMANG)')

        except Exception as e:
            print(f"Error processing {basename}: {e}")

# created this, but still need to implement it into main function !!!!!
def add_bpm_header(work_dir):
    '''
     nirwalsreduce.py expects header "BPM" that does not exist in our raw data files. 
     Manually create header using the bpm folder in the config directory
     '''

    # get BPM data from fits file
    bpm_dir = os.path.join('nirwals_pipeline','salt_pkgs', 'configs', 'nirwals', 'bpm')
    bpm_file = glob.glob(os.path.join(bpm_dir, 'NIRWALSBpm*.fits'))[0]
    with fits.open(bpm_file) as hdul:
        bpm_arr = hdul[1].data

    # loop through raw data files
    product_dir = os.path.join(work_dir, 'nirwals', 'product')
    product_files = glob.glob(os.path.join(product_dir, '*.fits'))
    for product_file in product_files:
        basename = os.path.basename(product_file)
        
        try:
            # add "BPM" hdu
            with fits.open(product_file, mode='update') as hdul:
                hdu_names = [hdu.name for hdu in hdul]   # list of hdu extension names
                if 'BPM' in hdu_names:
                    continue  # if this product file doesn't have it, check others
                else:
                    # create "BPM" hdu
                    bpm_hdu = fits.ImageHDU(data=bpm_arr, name='BPM')
                    bpm_hdu.header['COMMENT'] = 'bad-pixel mask'
                    hdul.append(bpm_hdu) # add to hdu list

        except Exception as e:
            print(f"Error processing {basename}: {e}")


def run_workflow(workflow_file, obs_date, param_dir, config_dir, work_dir, 
                 saltdata_dir='', with_stdout=True, only_stdout=False):
                # need saltdata_dir for nirwalsreduce to read, but do not have salt archive on local machine
    
    # setting dirs as absolute paths
    param_dir = os.path.abspath(param_dir)
    config_dir = os.path.abspath(config_dir)
    work_dir = os.path.abspath(work_dir)

    # if no work directory, make one
    os.makedirs(work_dir, exist_ok=True)

    # retireving workflow
    workflow = load_json(workflow_file)

    # retrieving log file
    log_file = workflow.get("log_file", "")

    # looping through tasks in workflow
    for task in workflow["tasks"]:
        
        # terminal output for debugging
        print(f"\n[WORKFLOW] {workflow_file}")
        print(f"   --- task: {task['name']}")
        print(f"   --- type = {task['type']}")
        print(f"   --- status = {task['status']}\n")

        # if task is set to 0, do not execute --> go to next task
        if task["status"] != 1:
            continue
        
        # if task is another workflow, call wrapper again (recursion)
        if task["type"] == "workflow":

            # terminal output for debugging
            print(f"          --> recursing into sub-workflow: {task['name']}\n")
            
            # retrieve sub-workflow file
            sub_workflow = os.path.join(os.path.dirname(workflow_file), task["name"])
            # recurse into sub-workflow
            run_workflow(sub_workflow, obs_date, param_dir, config_dir, work_dir, saltdata_dir, with_stdout, only_stdout,)

        # if task is a module, execute
        elif task["type"] == "module":
            
            # terminal output for debugging
            print(f"          --> calling module: {task['name']}\n")

            # translating workflow terms to primary functions in modules and saving as variables
            if task["name"].endswith(".prepare_data"):
                module_path = "nirwals_pipeline.salt_pkgs.code.saltreduce.saltreduce.science.nirwals.nirwalsprepare"
                func_name = "prepare_data"
            # ''
            elif task["name"].endswith(".reduce_data"):
                module_path = "nirwals_pipeline.salt_pkgs.code.saltreduce.saltreduce.science.nirwals.nirwalsreduce"
                func_name = "reduce_data"

                fix_primary_header_keys(work_dir)  # replace "CAMANG" hdu header with "AR-ANGLE" (pipeline expects this for camera ange)
                add_bpm_header(work_dir)  # add in 'BPM' hdu extension to product file using fits file in config directory

            else:
                raise ImportError(f"Unknown pipeline module: {task["name"]}")
            
            # importing module
            try:
                module = importlib.import_module(module_path)
            except Exception as e:
                print(f"Exception while importing module from path:\n{module_path}\n")
                raise

            # retrieving primary function from module
            try:
                func = getattr(module, func_name)
            except Exception as e:
                print(f"Unable to retrieve function [{func_name}] from module path:\n{module_path}\n")
                raise

            # saving args
            kwargs = {
                "params": task["params"],
                "actions": task["actions"],
                "param_dir": param_dir,
                "config_dir": config_dir,
                "work_dir": work_dir,
                "saltdata_dir": saltdata_dir,
                "with_stdout": with_stdout,
                "only_stdout": only_stdout
            }

            # terminal output for debugging
            print("\nkwargs passed to module:")
            for k, v in kwargs.items():
                print(f"   {k}: {v}")
            print()

            # temporarily change directory
            old_cwd = os.getcwd()
            os.chdir(work_dir)

            try:
                # calling primary function in module
                func(obs_date, log_file, **kwargs)

            # switch back to old directory
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    import sys

    # if input is not able to be passed, give instructions
    num_args = len(sys.argv)-1  # excluding initial python module command
    if num_args != 5:
        print('\n... not the right number of arguments...')
        print('you passed',num_args, ' - expects 5:',end='\n\n')
        print(sys.argv)
        print()
        print("Expected inputs:")
        print("<workflow.json>")
        print("<obs_date>")
        print("<param_dir>")
        print("<config_dir>")
        print("<work_dir>\n")
        sys.exit(1)

    # 0th arg --> python command to run module
    workflow_file = sys.argv[1]
    obs_date = sys.argv[2]
    param_dir = sys.argv[3]
    config_dir = sys.argv[4]
    work_dir = sys.argv[5]

    run_workflow(workflow_file=workflow_file, obs_date=obs_date, param_dir=param_dir, config_dir=config_dir, work_dir=work_dir)


