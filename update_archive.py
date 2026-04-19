import numpy as np

from dominates import dominates
from get_mofcn import getMOFcn


def update_archive(archive, particles, archive_size, FUN, numObj):
    """
    Python equivalent of MATLAB update_archive.m.
    """
    particles_arr = np.asarray(particles, dtype=float)
    if particles_arr.ndim == 1:
        particles_arr = particles_arr.reshape(1, -1)

    if particles_arr.ndim != 2:
        raise ValueError("particles must be a 2D array or 1D decision vector.")

    dim = particles_arr.shape[1]
    archive_size = int(archive_size)

    if archive is None or np.size(archive) == 0:
        archive_arr = np.empty((0, dim), dtype=float)
    else:
        archive_arr = np.asarray(archive, dtype=float)
        if archive_arr.ndim == 1:
            archive_arr = archive_arr.reshape(1, -1)
        if archive_arr.shape[1] != dim:
            raise ValueError("archive and particles must have the same decision dimension.")

    # Merge current archive and new solutions
    new_solutions = particles_arr
    fitness1, _, _ = getMOFcn(FUN, particles_arr, numObj)
    new_objs = fitness1

    fitness2 = np.empty((0, new_objs.shape[1]), dtype=float)
    if archive_arr.size != 0:
        fitness2, _, _ = getMOFcn(FUN, archive_arr, numObj)

    archive_solutions = np.vstack([archive_arr, new_solutions])
    archive_objs = np.vstack([fitness2, new_objs])

    # Remove dominated solutions and keep non-dominated ones
    is_dominated = np.zeros(archive_objs.shape[0], dtype=bool)
    for i in range(archive_objs.shape[0]):
        for j in range(archive_objs.shape[0]):
            if i != j and dominates(archive_objs[j, :], archive_objs[i, :]):
                is_dominated[i] = True
                break

    archive_solutions = archive_solutions[~is_dominated, :]
    archive_objs = archive_objs[~is_dominated, :]

    # If archive exceeds size limit, prune by crowding distance
    if archive_solutions.shape[0] > archive_size:
        distances = calculate_crowding_distance(archive_objs)
        sorted_indices = np.argsort(distances)[::-1]  # descending
        archive_solutions = archive_solutions[sorted_indices[:archive_size], :]
        archive_objs = archive_objs[sorted_indices[:archive_size], :]

    # Updated archive only contains non-dominated solutions
    archive = archive_solutions
    return archive


def calculate_crowding_distance(objs):
    """
    MATLAB equivalent of local function calculate_crowding_distance(objs).

    Note:
    If max(objs[:, m]) == min(objs[:, m]), denominator is zero.
    This function intentionally keeps MATLAB-like behavior, which can
    produce inf/nan in that term.
    """
    objs_arr = np.asarray(objs, dtype=float)
    if objs_arr.ndim != 2:
        raise ValueError("objs must be a 2D array.")

    num_objs = objs_arr.shape[1]
    num_solutions = objs_arr.shape[0]
    distances = np.zeros(num_solutions, dtype=float)

    if num_solutions == 0:
        return distances

    for m in range(num_objs):
        sorted_indices = np.argsort(objs_arr[:, m])
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        den = np.max(objs_arr[:, m]) - np.min(objs_arr[:, m])
        if (not np.isfinite(den)) or den <= 1e-12:
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(1, num_solutions - 1):
                distances[sorted_indices[i]] = distances[sorted_indices[i]] + (
                    (objs_arr[sorted_indices[i + 1], m] - objs_arr[sorted_indices[i - 1], m]) / den
                )

    return distances


if __name__ == "__main__":
    # Small 2-objective example (FUN='ZDT1')
    dim = 30
    archive = np.zeros((2, dim), dtype=float)
    archive[0, 0] = 0.1
    archive[1, 0] = 0.9
    archive[1, 1:] = 1.0  # make this one strongly dominated

    particles = np.zeros((2, dim), dtype=float)
    particles[0, 0] = 0.2
    particles[1, 0] = 0.8

    archive_size = 2
    updated = update_archive(archive, particles, archive_size, "ZDT1", 2)
    updated_obj, _, _ = getMOFcn("ZDT1", updated, 2)

    print("updated archive shape:", updated.shape)
    print("updated objectives:")
    print(updated_obj)
