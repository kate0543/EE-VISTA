import numpy as np

from vlga.chromosome import Chromosome


def crossover(chromosome1, chromosome2, chunk_size):
    config1 = chromosome1.get_config()
    config2 = chromosome2.get_config()

    size1 = config1.shape[0]
    size2 = config2.shape[0]

    n_chunk_1 = round(size1 / chunk_size)
    n_chunk_2 = round(size2 / chunk_size)

    # min_size = min(size1, size2)

    while True:
        t1 = np.random.randint(n_chunk_1 + 1)
        r1 = t1 * chunk_size - 1

        t2 = np.random.randint(n_chunk_2 + 1)
        r2 = t2 * chunk_size - 1

        if r1 == -1 and r2 == size2 - 1:
            continue

        if r1 == size1 - 1 and r2 == -1:
            continue

        break

    # 1A: [0, 1, ..., random_position]; 1B: [random_position + 1, size1 - 1]
    # 2A: [0, 1, ..., random_position]; 2B: [random_position + 1, size2 - 1]

    part_1a = config1[:r1 + 1]
    part_1b = config1[r1 + 1:]

    part_2a = config2[:r2 + 1]
    part_2b = config2[r2 + 1:]

    # concat operator return a new array instead of
    # referencing the original array
    new_config1 = np.concatenate((part_1a, part_2b))
    new_config2 = np.concatenate((part_2a, part_1b))

    check_valid_config(new_config1, chunk_size)
    check_valid_config(new_config2, chunk_size)

    new_chromosome1 = Chromosome(new_config1, None)
    new_chromosome2 = Chromosome(new_config2, None)

    return new_chromosome1, new_chromosome2


def mutate(chromosome, chunk_size):
    config = chromosome.get_config()
    size = config.shape[0]
    n_mutations = np.random.randint(size)

    positions = np.random.choice(size, n_mutations, replace=False)
    for position in positions:
        config[position] = 1 - config[position]

    if has_zero_layer(config, chunk_size):
        fill_zero_layer(config, chunk_size)

    check_valid_config(config, chunk_size)

    chromosome.set_config(config)
    chromosome.set_fitness(None)


def check_valid_config(config, chunk_size):
    assert config.shape[0] % chunk_size == 0
    assert config.shape[0] != 0

    if has_zero_layer(config, chunk_size):
        raise Exception("A config has zeros layer!!!")


def has_zero_layer(config, chunk_size):
    cnt = -1
    while True:
        sum = 0
        for i in range(chunk_size):
            cnt += 1
            sum += config[cnt]

        if sum == 0:
            return True

        if cnt == config.shape[0] - 1:
            break

    return False


def fill_zero_layer(config, chunk_size):
    print('Filling zero layer...')
    cnt = -1
    while True:
        sum = 0
        cnt2 = cnt
        for i in range(chunk_size):
            cnt2 += 1
            sum += config[cnt2]

        if sum == 0:
            for i in range(chunk_size):
                cnt += 1
                config[cnt] = 1
        else:
            cnt = cnt2

        if cnt == config.shape[0] - 1:
            break
