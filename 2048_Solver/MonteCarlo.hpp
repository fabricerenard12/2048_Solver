// Monte Carlo methods to compute the best move at each state
// Author: Fabrice Renard
// Date : 23 / 06 / 2023

#ifndef MONTECARLO_H
#define MONTECARLO_H

#include "Game.hpp"
#include <omp.h>

Game move(Game game, Move move);

void simulate(Game& game, double& score, std::mt19937& localGen);

Move performMC(Game game, int numberOfSimulationsPerMove);

#endif // !MONTECARLO_H
