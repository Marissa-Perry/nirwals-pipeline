import os
import json
import importlib

def load_json(path):
    with open(path) as f:
        return json.load(f)


def run_workflow(obs_date, workflow_filepath=None, with_stdout=True, only_stdout=False, run_primary=True):

    # setting dirs as absolute paths
    param_rel_path = os.path.join('nirwals_pipeline','salt_pkgs','params')
    param_dir = os.path.abspath(param_rel_path)
    config_rel_path = os.path.join('nirwals_pipeline','salt_pkgs','configs')
    config_dir = os.path.abspath(config_rel_path)
    work_rel_path = os.path.join('nirwals_pipeline',str(obs_date))  # work directory is under the observation date directory
    work_dir = os.path.abspath(work_rel_path)
    saltdata_dir = ''  # need saltdata_dir for nirwalsreduce to read, but do not have salt archive on local machine

    # if no work directory, make one
    os.makedirs(work_dir, exist_ok=True)

    # run primary reductions
    if run_primary:
        module_path = "nirwals_pipeline.salt_pkgs.code.saltreduce.primary.primary_reductions"
        print(f"\n[WORKFLOW] primary reductions")
        print(f"   --- module: {module_path}\n")
        primary = importlib.import_module(module_path)
        primary.run(obs_date)

    # retireving workflow
    if not workflow_filepath:
        # if one isn't passed, start with the main workflow file for the science DRP
        workflow_filepath = os.path.join('nirwals_pipeline','salt_pkgs','workflows','nirwals','workflow_science_nirwals.json')
    workflow = load_json(workflow_filepath)

    # retrieving log file
    log_file = workflow.get("log_file", "")

    # looping through tasks in workflow
    for task in workflow["tasks"]:
        
        # terminal output for debugging
        print(f"\n[WORKFLOW] {workflow_filepath}")
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
            sub_workflow = os.path.join(os.path.dirname(workflow_filepath), task["name"])
            # recurse into sub-workflow (do not re-run primary reductions)
            run_workflow(obs_date, workflow_filepath=sub_workflow, with_stdout=with_stdout, only_stdout=only_stdout, run_primary=False)

        # if task is a module, execute
        elif task["type"] == "module":
            
            # terminal output for debugging
            print(f"          --> calling module: {task['name']}\n")

            # translating workflow terms to primary functions in modules and saving as variables
            if task["name"].endswith(".prepare_data"):
                # module_path = "nirwals_pipeline.salt_pkgs.code.saltreduce.saltreduce.science.nirwals.nirwalsprepare"
                module_path = "nirwals_pipeline.salt_pkgs.code.saltreduce.science.nirwals.nirwalsprepare"
                func_name = "prepare_data"
            # ''
            elif task["name"].endswith(".reduce_data"):
                # module_path = "nirwals_pipeline.salt_pkgs.code.saltreduce.saltreduce.science.nirwals.nirwalsreduce"
                module_path = "nirwals_pipeline.salt_pkgs.code.saltreduce.science.nirwals.nirwalsreduce"
                func_name = "reduce_data"
            else:
                raise ImportError(f"Unknown pipeline module: {task['name']}")
            
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
    if num_args != 1:
        print('\n... not the right number of arguments...')
        print('you passed',num_args, ' - expects 1:',end='\n\n')
        print(sys.argv)
        print()
        print("Expected inputs:")
        print("<obs_date>\n")
        sys.exit(1)

    # 0th arg --> python command to run module
    obs_date = sys.argv[1]

    run_workflow(obs_date=obs_date)


