"""
Main entry point for the L2O project.
This script acts as a wrapper for the l2o package.
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("l2o", run_name="__main__", alter_sys=True)
