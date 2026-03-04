from yaw_prototype import *
import time




if __name__ == "__main__":
    start_time = time.perf_counter()
    compute_errors = True
    logical_alg = qubit()
    I = logical_alg.I
    X = logical_alg.X
    Z = logical_alg.Z

    name = "Bit-Flip Code"
    physical_alg = qubit_system(3)
    stabilizer_generators = [Z @ Z @ I, I @ Z @ Z]
    example_syndrome = (1,0)

    # name = "Phase-Flip Code"
    # physical_alg = qubit_system(3)
    # stabilizer_generators = [X @ X @ I, I @ X @ X]
    # example_syndrome = (1,1)

    # name = "Five-Qubit Code"
    # physical_alg = qubit_system(5)
    # stabilizer_generators = [X @ Z @ Z @ X @ I, 
    #                          I @ X @ Z @ Z @ X,
    #                          X @ I @ X @ Z @ Z,
    #                          Z @ X @ I @ X @ Z]
    # example_syndrome = (1,1,0,0)

    # name = "Steane Code"
    # physical_alg = qubit_system(7)
    # stabilizer_generators = [I @ I @ I @ Z @ Z @ Z @ Z, 
    #                          I @ Z @ Z @ I @ I @ Z @ Z,
    #                          Z @ I @ Z @ I @ Z @ I @ Z,
    #                          I @ I @ I @ X @ X @ X @ X,
    #                          I @ X @ X @ I @ I @ X @ X,
    #                          X @ I @ X @ I @ X @ I @ X]
    # example_syndrome = (1,1,1,1,1,1)

    # name = "Shor Code"
    # physical_alg = qubit_system(9)
    # stabilizer_generators = [Z @ Z @ I @ I @ I @ I @ I @ I @ I, 
    #                          I @ Z @ Z @ I @ I @ I @ I @ I @ I,
    #                          I @ I @ I @ Z @ Z @ I @ I @ I @ I,
    #                          I @ I @ I @ I @ Z @ Z @ I @ I @ I,
    #                          I @ I @ I @ I @ I @ I @ Z @ Z @ I,
    #                          I @ I @ I @ I @ I @ I @ I @ Z @ Z,
    #                          X @ X @ X @ X @ X @ X @ I @ I @ I,
    #                          I @ I @ I @ X @ X @ X @ X @ X @ X]
    # example_syndrome = (0,0,0,0,0,0,0,1)

    stabilizer_code = AutoStabilizerCode(stabilizer_generators, logical_alg, physical_alg, name, compute_errors)
    correction_operator = stabilizer_code.lookup_correction(example_syndrome)
    end_time = time.perf_counter()
    time_taken = f"{end_time-start_time:.4f} s"
    print(f"Time taken to generate {stabilizer_code.name}: {time_taken}.")