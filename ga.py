#!/usr/bin/env python
import numpy as np
import random
import multiprocessing as mp
import sys


class Individual:
    """
    Individual(x0=np.zeros(2), ndim=2, bounds=np.array((np.ones(2) * 10 * -1, np.ones(2) * 10)), type_create='uniform')
    
    Cria os individuos que compoem a população da classe Genetic 
    
    Atributos
    ----------
    fitness : None or flout
        é o melhor valor do individuo
    size_individual : int 
        é o tamanho do indiviuo 
    create_individual : class 'fffit.ga.Individual'
        cria o individuo com suas caracteristicas
    
    Parametros
    ----------
    type_create : str, optional
        Define qual será o tipo de criação da população inicial
    x0 : np.ndarray, optional
        Define qual o ponto inicial até os limites de onde vai criar o valor do indviduo
    bounds : numpy.ndarray
        Define até onde pode ir os valores dos individuo, como inferio e superior
    ndim : integer
        Define quantos dimensões tem o indiviuo ou seja o tamanho do array
    sigma :float, opcional
        Define a probabilidade do individuo ter mutação
    
    
    Exemplos
    --------
    >>> from fffit import ga
    >>> import numpy as np
    >>> ranges = 2
    >>> ndim = 2
    >>> bounds = np.array((np.ones(2) * 10 * -1, np.ones(2) * 10))
    >>> individual = ga.Individual(x0=np.zeros(2), ndim=2, bounds=bounds, sigma=None, type_create='uniform')
    <fffit.ga.Individual object at 0x7f8968797c18>
    >>> individual.chromosome
    array([ 5.7287427 , -0.54066483])
    """
    def __init__(self, type_create='uniform', x0=None, bounds=None, ndim=None, sigma=None):
        self.fitness = None
        self.size_individual = ndim
        self.create_individual(type_create, x0, bounds, sigma)
        super().__init__()

    def create_individual(self, type_create, x0, bounds, sigma):
        """
        create_individual(type_create=uniform, x0=np.zeros(2), np.array((np.ones(2) * 10 * -1, np.ones(2) * 10)), sigma=None)
        
        Cria os chromosome que pertence a classe Individual

        Parametros
        ----------
        type_create : str, optional
            Define qual será o tipo de criação da população inicial
        x0 : np.ndarray, optional
            Define qual o ponto inicial até os limites de onde vai criar o valor do indviduo
        bounds : numpy.ndarray
            Define até onde pode ir os valores dos individuo, como inferio e superior
        sigma :float, opcional
            Define a probabilidade do individuo ter mutação

    
        """

        if type_create == 'gaussian':
            if sigma is not None and not (np.shape(sigma) == (self.size_individual,) or isinstance(sigma, float) or isinstance(sigma, int)):
                raise ValueError(f'sigma bust be a single float or an array with {self.size_individual} entries.')
            self.chromosome = np.random.normal(x0, sigma, size=self.size_individual)
        elif type_create == 'integer':
            if bounds is None or np.shape(bounds) != (2, self.size_individual):
                raise ValueError(f'bounds must be of shape (2, {self.size_individual}). Instead, got {bounds}.')
            self.chromosome = np.random.randint(bounds[0], bounds[1], size=self.size_individual)
        elif type_create == 'uniform':
            if bounds is None or np.shape(bounds) != (2, self.size_individual):
                raise ValueError(f'bounds must be of shape (2, {self.size_individual}). Instead, got {bounds}.')
            self.chromosome = np.random.uniform(bounds[0], bounds[1], size=self.size_individual)
        else:
            raise ValueError(f'Invalid individual creation type: {type_create}')


class Genetic(object):

    """
    Genetic(maxiter=1000, goal=0, cross_over='one_point',
                       mutation_probability=0.01, mutation='uniform',
                       selection_method='elitism',num_parents=2,
                       num_elitism=10, bounds=np.array((np.ones(2) * 10 * -1, np.ones(2) * 10)))
    
    É a classe que é responsavel por realizar a criar a população, 
    selecionar os parentes da população, realizar os cruzamentos e mutação. 
    
    Parametros
    ----------
    goal : float, opcional
        Define o valor a qual queremos nos aproximar
    bounds : numpy.ndarray
        Define até onde pode ir os valores dos individuo, como inferio e superior
    mutation_probability : float
        Define a probabilidade de ocorrer a mutação
    selection_probability : float
        Define a probabilidade de ocorrer a seleção de pais
    sigma : float
        Define a probabilidade do individuo ter mutação
    num_parents : integer
        Define o numero de parentes que será esolhido no metodo da seleção 
    num_elitism : integer
        Define o numero de pais que será preservado entre as gerações
    maxiter : integer,opcional
        Define o numero de interações maximo que o algoritmo irá fazer para encontrar o resultado
    selection_method : str, opcional
        Define o metodo de seleção que será usado para escolher os pais da proxima geração
    cross_over : str, opcional
        Define o metodo de cruzamento que será usado
    mutation : str, opcional
        Define o metodo de mutação que será usado
    submit_to_cluster : bool, opcional
        Define se a meta-hurística será executada no cluster
    
    Exemplos
    --------
    >>> from fffit import ga
    >>> bounds = np.array((np.ones(2) * 10 * -1, np.ones(2) * 10))
    >>> ga.Genetic(maxiter=1000, goal=0, cross_over='one_point',
                       mutation_probability=0.01, mutation='uniform',
                       selection_method='elitism',num_parents=2,
                       num_elitism=10, bounds=bounds)
                       
    >>> 
    """
    def __init__(self, goal=1.0 ,bounds = None,
                 mutation_probability=0.5, selection_probability=0.5,
                 sigma=0.5, num_parents=2,num_elitism=2, maxiter=None,
                 selection_method='elitism', cross_over='uniform',
                 mutation='gaussian', submit_to_cluster=False):
        """Inicializar o objeto PSO"""
        self.num_parents = num_parents
        self.num_elitism = num_elitism
        self.mutation_probability = mutation_probability
        self.selection_probability = selection_probability
        self.sigma = sigma  # Mutation size
        self.population = []
        self.model = np.array(np.ones(5))
        self.bounds = bounds
        self.ncpu = 1
        self.step_number = 0
        self.maxiter = maxiter
        self.submit_to_cluster = submit_to_cluster
        self.goal = goal
        self.fitness = None
        self.selection_method = selection_method
        self.x0 = []
        self.size_population = None
        self.cont_new_population = 0
        self.step_evaluation = 0
        self.improving = True

        if cross_over == 'uniform':
            self.cross_over = self.cross_over_uniform
        elif cross_over == 'two_points':
            self.cross_over = self.cross_over_two_points
        elif cross_over == 'one_point':
            self.cross_over = self.cross_over_one_point
        else:
            raise ValueError(f'Invalid crossover: {cross_over}.')

        if mutation == 'binary':
            self.mutation = self.mutation_binary
        elif mutation == 'gaussian':
            self.mutation = self.mutation_gaussian
        elif mutation == 'uniform':
            self.mutation = self.mutation_uniform
        else:
            raise ValueError(f'Invalid mutation: {mutation}.')

    def populate(self, size_population, bounds=None, x0=None, ndim=None, sigma=None, type_create='uniform'):
        """
        Retorna uma lista consistindo de vários indivíduos que formam a população.


        Return:
            :return: void

        """

        self.size_population = size_population
        if x0 is None and ndim is None:
            raise ValueError('Either x0 or ndim bust be given')
        elif x0 is not None:
            self.x0 = x0
            self.ndim = len(x0)
        else:
            self.ndim = ndim
        self.population = [Individual(x0=x0, ndim=self.ndim, bounds=bounds, sigma=sigma, type_create=type_create) for i in range(size_population)]

    def calculate_pop_fitness(self, func):
        """
        calcula a aptidão da população e retorna em lista da classe genética.


        Returns:
            :return: void

        """
        for individual in self.population:
            func(individual)

    def calculate_best_fitness(self):
        """Calcula a melhor aptidão entre todas as populações e salva na lista da classe genética.


        Returns:
            :return: void

        """
        self.fitness = max([k.fitness for k in self.population])
        return self.fitness

    def calculate_avg_fitness(self):
        """Calcula a aptidão de media entre todas as populações e salva na lista da classe genética.


        Returns:
            :return: void

        """
        return sum([k.fitness for k in self.population]) / np.size(self.population)

    @staticmethod
    def sorted_population(population):
        """
        Selecione o gene a ser realizado a mutação.


        Args:
            :param population:(:obj:`list`): Lists one containing a population.


        Returns:
            :return: score(:obj:`list`): An element of the population list select.
        """
        return sorted(population, key=lambda k: k.fitness)

    def roulette_selection(self):
        """
        Esta função realiza a seleção por releta do gene a ser realizado a mutação.


        Args:
            :param population: population:(:obj:`list`): Lists one containing a population.



        Returns:
            :return: selected(:obj:`list`): An element of the population list select.
        """

        population = self.sorted_population(self.population)
        parents = []
        sum_score = sum([k.fitness for k in population])
        if not np.all(np.sign([k.fitness for k in population]) == np.sign(sum_score)):
            raise ValueError('Not all fitnesses have the same sign. This is not supported')
        if sum_score < 0:
            # If sum_score is negative, we assume all fitnesses are negative as
            # well (we're actually trying to minimize something, so we need to make
            # sure that only positive values are sent as roulette wheel fractions.
            for individual in population:
                individual.fitness = individual.fitness - sum_score
        for _ in range(self.num_parents):
            sum_score = sum([k.fitness for k in population])
            selection_criteria = 0
            key = np.random.uniform(0, sum_score)
            for idx in range(len(population)):
                selection_criteria += population[idx].fitness
                if selection_criteria > key:
                    parents.append(population.pop(idx))
                    break
        return parents

    def random_selection(self):
        """
        Esta função realiza uma seleção do torneio e retorno gene vencedor.


        Args:
            :param population:(:obj:`list`): Lists one containing a population.


        Returns:
            :return: sub_population(:obj:`list`): An element of the sub_population list select.
        """
        parents = []
        for k in range(self.num_parents):
            sub_population = random.sample(self.population, 2)
            sub_population = self.sorted_population(sub_population)
            parents.append(sub_population[0])
        return parents

    def elitism_selection(self):
        """
        Função que realiza seleção elitista.

        Args:
            :param population:(:obj:`list`): Lists one containing a population.


        Returns:
            :return:population(:obj:`list`): An element of the population list select.
        """
        population = self.sorted_population(self.population)
        parents = population[-self.num_parents:]
        return parents

    def selection(self):
        """[summary]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if self.selection_method == 'roulette':
            parents = self.roulette_selection()
            return parents
        elif self.selection_method == 'random':
            parents = self.random_selection()
            return parents
        elif self.selection_method == 'elitism':
            parents = self.elitism_selection()
            return parents
        else:
            raise ValueError(f'Invalid {self.selection_method} selection method.')

    def cross_over_one_point(self, population):
        """
        Esta função realiza o cruzamento de um ponto no gene.


        Args:
            :param population:(:obj:`list`): Lists one containing a population.
            :param parents :(:obj:`float`): parents gene to carry out the mutation.


        Returns:

        """
        population = self.sorted_population(population)
        local_parents = self.selection()
        for individual in population[:-self.num_elitism]:
            locus = np.random.randint(0, individual.size_individual)
            parent_1 = np.random.randint(self.num_parents)
            parent_2 = np.random.randint(self.num_parents)
            while parent_2 == parent_1:
                parent_2 = np.random.randint(self.num_parents)
            individual.chromosome[:locus] = local_parents[parent_1].chromosome[:locus].copy()
            individual.chromosome[locus:] = local_parents[parent_2].chromosome[locus:]
        self.population = population

    def cross_over_two_points(self, population):
        """
        Esta função realiza o cruzamento de dois pontos no gene.


        Args:
            :param population:(:obj:`list`): Lists one containing a population.
            :param parents :(:obj:`float`): parents gene to carry out the mutation.


        Returns:

        """
        population = self.sorted_population(population)
        local_parents = self.selection()
        for individual in population[:-self.num_elitism]:
            while True:
                locus = np.random.randint(0, individual.size_individual)
                locusTwo = np.random.randint(locus, individual.size_individual)
                if locus != locusTwo:
                    break
            individual.fitness = None
            parent_1 = np.random.randint(self.num_parents)
            parent_2 = np.random.randint(self.num_parents)
            while parent_2 == parent_1:
                parent_2 = np.random.randint(self.num_parents)
            individual.chromosome = local_parents[parent_1].chromosome.copy()
            # To avoid messing with local_parents in the next line.
            individual.chromosome[locus:locusTwo] = local_parents[parent_2].chromosome[locus:locusTwo]
        self.population = population

    def cross_over_uniform(self, population):
        """
        Esta função realiza o cruzamento uniforme no gene.


        Args:
            :param population:(:obj:`list`): Lists one containing a population.
            :param parents :(:obj:`float`): parents gene to carry out the mutation.


        Returns:

        """
        population = self.sorted_population(population)
        local_parents = self.selection();
        for individual in population[:-self.num_elitism]:
            parent_1 = np.random.randint(self.num_parents)
            parent_2 = np.random.randint(self.num_parents)
            while parent_2 == parent_1:
                parent_2 = np.random.randint(self.num_parents)
            for j in range(individual.size_individual):
                drawn = np.random.choice((parent_1, parent_2))
                individual.chromosome[j] = local_parents[drawn].chromosome[j]

        self.population = population

    def mutation_uniform(self, population):
        """
        A função realiza a mutação uniforme e retorna a população modificada.


         Args:
            :param population:(:obj:`list`): Lists one containing a population.



        Returns:

        """
        for individual in population:
            for locus in range(individual.size_individual):
                if random.random() <= self.mutation_probability:
                    individual.chromosome[locus] = np.random.uniform(
                        self.bounds[0][locus],self.bounds[1][locus], size=1)

    def mutation_binary(self, population):
        """

        A função realiza a mutação binária e retorna a população com a modificação,
        vale ressaltar que esta mutação só é válida para população binária.


         Args:
            :param population:(:obj:`list`): Lists one containing a population.



        Returns:

        """
        for individual in population:
            for locus in range(individual.size_individual):
                if random.random() <= self.mutation_probability:
                    if individual.chromosome[locus] == 1:
                        individual.chromosome[locus] = 0
                    elif individual.chromosome[locus] == 0:
                        individual.chromosome[locus] = 1

    def mutation_gaussian(self, population):
        """

        A função realiza a mutação de Gausiana e retorna a população com a modificada.


        Args:
            :param population:(:obj:`list`): Lists one containing a population.



        Returns:

                """
        for individual in population:
            for locus in range(individual.size_individual):
                if random.random() <= self.mutation_probability:
                    individual.chromosome[locus] = np.random.normal(individual.chromosome[locus],
                                                                   self.sigma)

    def update_swarm(self):
        """
        Atualize a população realizando cruzamento, mutação
        
        Returns:
            :return population(:obj:`list` of :obj:`Particles`): returns a list of
            swarms.
        """
        if self.fitness is None:
            logging.error("Cannot update the population before calculating Fitness")
            raise RuntimeError("Updated the population before calculating Fitness")
        self.cross_over(self.population)
        self.mutation(self.population)
        if self.submit_to_cluster:
            self.curr_iter['update'] += 1

    def evaluate_single_fitness_test(self, func,
                                     enum_particles=False, add_step_num=False,
                                     **kwargs):
        """
        Execute a função fornecida como o teste de aptidão para todas as partículas.

        Parametros:
        -----------
        fun : callable
            The fitness test function to be minimized:

                ``func(individual.ichromosome, **kwargs) -> float``.

        enum_particles : boolean
            If `True`, the population will be enumerated and the individual index will
            be passed to `func` as keyword `part_idx`, added to `kwargs`

        add_step_num : boolean
            If `True`, the current step number will be passed to `func`
            as keyword `step_num`, added to `kwargs`

        **kwargs: Other keywords to the fitness function, will be passed as is.
        """
        if add_step_num:
            kwargs['step_num'] = self.step_number
        if self.ncpu == 1:
            if enum_particles:
                for part_idx, individual in enumerate(self.population):
                    kwargs['part_idx'] = part_idx
                    individual.fitness = func(individual.chromosome, **kwargs)
            else:
                for individual in self.population:
                    individual.fitness = func(individual.chromosome, **kwargs)
        elif self.ncpu > 1:
            with mp.Pool(processes=self.ncpu) as pool:
                argslist = []
                p = []
                for part_idx, individual in enumerate(self.population):
                    argslist.append(dict(kwargs))
                    if enum_particles:
                        argslist[-1]['part_idx'] = part_idx
                for idx, args in enumerate(argslist):
                    p.append(pool.apply_async(func, args=(self.population[idx].chromosome,),kwds=args))
                results = [ r.get() for r in p ]
            for part_idx, individual in enumerate(self.population):
                individual.fitness = results[part_idx]

    def do_full_step(self, func, **kwargs):
        """Execute uma etapa completa de GA.

        Este método passa por todos os outros métodos para realizar uma completa
        Etapa GA, para que possa ser chamada a partir de um loop no método run ()..
        """
        if self.fitness is not None and self.step_number < self.maxiter:
            self.cross_over(self.population)
            self.mutation(self.population)
            for individual in self.population:
                np.clip(individual.chromosome, self.bounds[0], self.bounds[1], out=individual.chromosome)
        if self.submit_to_cluster:
            raise NotImplementedError('Multistep jobs are under revision.')
        else:
            self.evaluate_single_fitness_test(func, **kwargs)
        self.step_number += 1

    def fitness_variation(self, fitness_evaluation):
        """
        Essa função realiza a verificação da variação do fitness entre as populações
        
        Args:
            fitness_evaluation ([type]): [description]
        """
        if(fitness_evaluation != self.fitness):
                self.step_evaluation = 0
        else:
            self.step_evaluation += 1
            print('step_evaluation: {}'.format(self.step_evaluation))
        if(self.step_evaluation > 100):
            self.improving = False


    def run(self, func, DEBUG=None, **kwargs):
        """Execute uma execução de otimização completa.

        Faz a otimização com a execução da atualização das velocidades
        e as coordenadas também verifica o critério interrompido para encontrar fitnnes.

        Parameters
        ----------
        func : callable
            Function that calculates fitnnes.

        Returns
        -------
            The dictionary that stores the optimization results.
        """
        self.func = func
        while (self.step_number < self.maxiter and self.improving):
            self.do_full_step(func, **kwargs)
            fitness_evaluation= self.fitness
            self.calculate_best_fitness()
            #self.fitness_variation(fitness_evaluation)
            if DEBUG is not None:
                with open(DEBUG, 'a') as dbg_file:
                    print(f"# {self.step_number} {self.fitness}")
                    print(f" {self.step_number} {self.fitness}",
                          file=dbg_file)
                    #np.savetxt(dbg_file,
                    #           [(*p.chromosome, p.fitness)
                    #               for p in self.population])
            if self.fitness >= self.goal:
                break
        self.results = {}
        self.results['fitness'] = self.fitness
        return self.results

