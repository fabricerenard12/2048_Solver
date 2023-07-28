// Monte Carlo methods to compute the best move at each state
// Author: Fabrice Renard
// Date : 23 / 06 / 2023

#include "MonteCarlo.hpp"

Game move(Game game, Move move) {
    // Perform move on game
    switch (move) {
    case Move::LEFT:
        game.moveLeft();
        break;
    case Move::RIGHT:
        game.moveRight();
        break;
    case Move::UP:
        game.moveUp();
        break;
    case Move::DOWN:
        game.moveDown();
        break;
    default:
        break;
    }

    return game;
}

void simulate(Game& game, std::mt19937& localGen, double& localScore) {
    int minValue = 0;
    int maxValue = 3;

    std::uniform_int_distribution<int> intDistribution(minValue, maxValue);

    // Simulate game until end
    while (!game.isGameOver()) {
        Move randomMove = static_cast<Move>(intDistribution(localGen));

        switch (randomMove) {
        case Move::LEFT:
            game.moveLeft();
            break;
        case Move::RIGHT:
            game.moveRight();
            break;
        case Move::UP:
            game.moveUp();
            break;
        case Move::DOWN:
            game.moveDown();
            break;
        default:
            break;
        }
    }

    // Update local score
    localScore += game.getScore();
}

std::map<double, Move, Compare> performMC(Game game, int numberOfSimulationsPerMove) {
    std::random_device rd;
    std::vector<double> scores(4, 0.0);
    std::map<double, Move, Compare> moves;

    // Set number of threads
    int numThreads = omp_get_num_procs();
    omp_set_num_threads(numThreads);

    // Generate random numbers generators
    std::vector<std::mt19937> generators(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        generators[i].seed(rd() + i);
    }

    // Compute best move using Monte Carlo Computation
    for (int j = 0; j < 4; j++) {
        Move currentMove = static_cast<Move>(j);

        #pragma omp parallel for
        for (int i = 0; i < numberOfSimulationsPerMove; i++) {
            int threadIndex = omp_get_thread_num();
            std::mt19937& localGen = generators[threadIndex];
            Game gameCopy = move(game, currentMove);
            double localScore = 0.0;
            simulate(gameCopy, localGen, localScore);

            #pragma omp critical
            scores[j] += localScore;
        }
    }

    // Construct moves map
    for (int i = 0; i < 4; i++) {
        moves.insert({ scores[i], static_cast<Move>(i) });
    }

    return moves;
}

