"""
Test script for MOEA/DD algorithm implementation
"""
import numpy as np
from Methods.mtop import MTOP
from Problems.STMO.ZDT import ZDT1, ZDT2, ZDT3
from Problems.STMO.DTLZ import DTLZ1, DTLZ2, DTLZ3
from Algorithms.STMO.MOEADD import MOEADD


def test_moeadd_zdt1():
    """Test MOEA/DD on ZDT1 (2-objective problem)"""
    print("\n" + "="*60)
    print("Testing MOEA/DD on ZDT1 (2-objective)")
    print("="*60)

    # Create problem
    problem = MTOP()
    zdt1 = ZDT1(30)
    problem.add_task(zdt1.obj_func, zdt1.dim, zdt1.lower_bound, zdt1.upper_bound)

    # Run MOEA/DD
    algorithm = MOEADD(problem, n=100, max_nfes=10000, delta=0.9,
                       save_data=False, disable_tqdm=False)
    results = algorithm.optimize()

    # Print results
    print(f"\nFinal population size: {len(results.objs[0])}")
    print(f"Objective range: [{results.objs[0].min(axis=0)}, {results.objs[0].max(axis=0)}]")
    print("✓ ZDT1 test completed successfully")

    return results


def test_moeadd_dtlz2():
    """Test MOEA/DD on DTLZ2 (3-objective problem)"""
    print("\n" + "="*60)
    print("Testing MOEA/DD on DTLZ2 (3-objective)")
    print("="*60)

    # Create problem
    problem = MTOP()
    dtlz2 = DTLZ2(12, 3)
    problem.add_task(dtlz2.obj_func, dtlz2.dim, dtlz2.lower_bound, dtlz2.upper_bound)

    # Run MOEA/DD
    algorithm = MOEADD(problem, n=105, max_nfes=10000, delta=0.9,
                       save_data=False, disable_tqdm=False)
    results = algorithm.optimize()

    # Print results
    print(f"\nFinal population size: {len(results.objs[0])}")
    print(f"Objective range: [{results.objs[0].min(axis=0)}, {results.objs[0].max(axis=0)}]")
    print("✓ DTLZ2 test completed successfully")

    return results


def test_moeadd_dtlz1():
    """Test MOEA/DD on DTLZ1 (3-objective with multiple local fronts)"""
    print("\n" + "="*60)
    print("Testing MOEA/DD on DTLZ1 (3-objective, multimodal)")
    print("="*60)

    # Create problem
    problem = MTOP()
    dtlz1 = DTLZ1(7, 3)
    problem.add_task(dtlz1.obj_func, dtlz1.dim, dtlz1.lower_bound, dtlz1.upper_bound)

    # Run MOEA/DD with smaller population for quick test
    algorithm = MOEADD(problem, n=91, max_nfes=5000, delta=0.9,
                       save_data=False, disable_tqdm=False)
    results = algorithm.optimize()

    # Print results
    print(f"\nFinal population size: {len(results.objs[0])}")
    print(f"Objective range: [{results.objs[0].min(axis=0)}, {results.objs[0].max(axis=0)}]")
    print("✓ DTLZ1 test completed successfully")

    return results


def test_algorithm_info():
    """Test algorithm information retrieval"""
    print("\n" + "="*60)
    print("Testing Algorithm Information")
    print("="*60)

    info = MOEADD.get_algorithm_information(print_info=True)
    print("✓ Algorithm info test completed successfully")

    return info


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# MOEA/DD Algorithm Test Suite")
    print("#"*60)

    # Test algorithm info
    test_algorithm_info()

    # Test on different problems
    results_zdt1 = test_moeadd_zdt1()
    results_dtlz2 = test_moeadd_dtlz2()
    results_dtlz1 = test_moeadd_dtlz1()

    print("\n" + "#"*60)
    print("# All tests completed successfully!")
    print("#"*60)
