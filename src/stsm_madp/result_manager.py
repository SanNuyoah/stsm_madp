import os
import sys
sys.dont_write_bytecode = True


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RESULTS_RUN = os.path.join(PACKAGE_ROOT, "results", "run")
VALID_VARIANTS = ("baseline", "stsm")


def get_results_run_root(results_root=None):
    root = (
        results_root or
        os.environ.get("STSM_RESULTS_RUN") or
        DEFAULT_RESULTS_RUN)
    return os.path.abspath(root)


def get_result_dir(robot, variant, results_root=None, create=True):
    robot = str(robot or "").strip().lower()
    variant = str(variant or "").strip().lower()
    if not robot:
        raise ValueError("robot is required")
    if variant not in VALID_VARIANTS:
        raise ValueError("invalid result variant={}".format(variant))
    path = os.path.join(get_results_run_root(results_root), robot, variant)
    if create and not os.path.isdir(path):
        os.makedirs(path)
    return path


def result_path(robot, variant, filename, results_root=None, create=True):
    if not filename:
        raise ValueError("filename is required")
    return os.path.join(
        get_result_dir(robot, variant, results_root=results_root, create=create),
        filename)
