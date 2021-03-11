'''Tekne Consulting blogpost --- teknecons.com'''


import numpy as np
from scipy import optimize as opt
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from scipy.spatial import distance as dis
from decorators import inst_to_tuple
from catboost import CatBoostRegressor
import os


this_dir = os.path.dirname(os.path.abspath(__file__))
rg = np.random.default_rng(seed=1234)
toolbox = base.Toolbox()


def initialize_globals(ind_size=2, pop_size=20, mut_prob=0.5, mut_prob_s=0.5,
                       mat_prob=0.5, t_size=4, evo_cycles=100, evo_startegy=[0.2, 0.7],
                       attr_range=[-10, 10]):
    global IND_SIZE
    IND_SIZE = ind_size
    global POP_SIZE
    POP_SIZE = pop_size
    global MUT_PROB
    MUT_PROB = mut_prob
    global MUT_PROB_S
    MUT_PROB_S = mut_prob_s
    global MAT_PROB
    MAT_PROB = mat_prob
    global T_SIZE
    T_SIZE = t_size
    global EVO_CYCLES
    EVO_CYCLES = evo_cycles
    global EVO_STRATEGY
    EVO_STRATEGY = evo_startegy
    global ATTR_RANGE
    ATTR_RANGE = attr_range
    return(True)


def target_fun(coord):
    xy = np.array(coord).reshape(-1, 2)
    model = CatBoostRegressor()
    model.load_model(os.path.join(this_dir, 'tree_approx_model.cbm'))
    return(model.predict(xy).ravel()[0])


'''evaluation of fitness'''


@inst_to_tuple  # deap requires evaluate to return tupe, but final optimizer requires number
def evaluate(individual, target_function):
    z = target_function(individual)
    return(z)


'''modify if statement for legit feasibility check'''


def feasible(individual):
    if -10 <= individual[0] <= 10 and -10 <= individual[1] <= 10:
        return(True)
    else:
        return(False)


'''penalty for being outside feasibility region'''


def distance(individual):
    dist = dis.euclidean([np.average(ATTR_RANGE), np.average(ATTR_RANGE)], individual)
    return(dist**2)


'''creating individuals with evolution strategy'''


def generate_ev_s(ind_class, strategy_class, ind_size, strategy_range, attributes_range):
    ind = ind_class(rg.uniform(*attributes_range) for _ in range(ind_size))
    ind.strategy = strategy_class(rg.uniform(*strategy_range) for _ in range(ind_size))
    return(ind)


'''ensuring strategy >= minimum'''


def check_strategy(strategy_range):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if strategy_range[0] > s:
                        child.strategy[i] = strategy_range[0]
                    elif strategy_range[1] < s:
                        child.strategy[i] = strategy_range[1]
                    else:
                        continue
            return(children)
        return(wrappper)
    return(decorator)


'''custom mutation composed of mutGaussian and strategy mutation'''


@ inst_to_tuple
def custom_mutation(individual, mu, sigma):
    ind = toolbox.clone(individual)
    new_strategy = [i + rg.normal(0, 0.1) if rg.uniform() < MUT_PROB_S else i for i in ind.strategy]
    ind.strategy = new_strategy
    del ind.fitness.values
    for i, s in enumerate(ind.strategy):
        if rg.uniform() < s:
            ind[i] += rg.normal(mu, sigma)
    return(ind)


def set_env():
    '''setting the environment'''
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, strategy=None)
    creator.create("Strategy", list)
    return(True)


def main(target_function=target_fun, penalty=True, **globals_dict):
    initialize_globals(**globals_dict)

    '''registering object and function in environment'''
    toolbox.register("individual", generate_ev_s, creator.Individual, creator.Strategy,
                     IND_SIZE, EVO_STRATEGY, ATTR_RANGE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, target_function=target_function)
    if penalty:
        toolbox.decorate("evaluate", tools.DeltaPenalty(
            feasible, 1000, distance))  # out of region penalty
    toolbox.register("mate", tools.cxESTwoPoint)
    toolbox.decorate("mate", check_strategy(EVO_STRATEGY))
    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=MUT_PROB) #mutESLogNormal has low efficiency
    toolbox.register("mutate", custom_mutation, mu=0, sigma=1)
    toolbox.decorate("mutate", check_strategy(EVO_STRATEGY))
    toolbox.register("select", tools.selTournament, tournsize=T_SIZE)  # selBest as an alternative?

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    strategy_stats = tools.Statistics(lambda ind: ind.strategy)
    stats = tools.MultiStatistics(fitness=fitness_stats, strategy=strategy_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    result, log = algorithms.eaMuPlusLambda(pop, toolbox, POP_SIZE, POP_SIZE, MUT_PROB, MAT_PROB,
                                            ngen=EVO_CYCLES, stats=stats, halloffame=hof, verbose=False)
    result.sort(key=lambda x: x.fitness.values)
    return(result, log, hof)


if __name__ == "__main__":
    set_env()
    result, log, hof = main()
    print("the best individuals: {}".format(result))
    print("hall of fame: {}".format(hof))
    print("logbook: {}".format(log))
    optimal = opt.minimize(target_fun, x0=hof[0])
    print("optimized function value is {} for coordinates: {}".format(optimal['fun'], optimal['x']))
