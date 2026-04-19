import numpy as np


def dominates(obj1, obj2) -> bool:
    """
    MATLAB equivalent:
    all(obj1 <= obj2) && any(obj1 < obj2)
    """
    a = np.asarray(obj1).ravel()
    b = np.asarray(obj2).ravel()
    if a.shape != b.shape:
        raise ValueError("obj1 and obj2 must have the same shape.")
    return bool(np.all(a <= b) and np.any(a < b))


if __name__ == "__main__":
    # Test 1: dominates (True)
    t1 = dominates([1.0, 2.0], [1.0, 3.0])
    # Test 2: not dominates (False)
    t2 = dominates([1.0, 4.0], [1.0, 3.0])
    print("test1 (should be True):", t1)
    print("test2 (should be False):", t2)
