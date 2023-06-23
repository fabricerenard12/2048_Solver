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

void simulate(Game& game, double& score, std::mt19937& localGen) {
    int minValue = 0;
    int maxValue = 3;

    std::uniform_int_distribution<int> intDistribution(minValue, maxValue);

    // Simulate game until end
    while (true) {
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

        if (game.isGameOver()) {
            break;
        }
    }

    // Update score
    #pragma omp critical
    score += game.getScore();
}

Move performMC(Game game, int numberOfSimulationsPerMove) {
    std::random_device rd;
    std::vector<double> scores(4, 0.0);

    // Set number of threads
    int numThreads = omp_get_num_procs();
    omp_set_num_threads(numThreads);

    // Generate random numbers generators
    std::vector<std::mt19937> generators(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        generators[i].seed(rd());
    }

    // Compute best move using Monte Carlo Computation
    for (int j = 0; j < 4; j++) {
        Move currentMove = static_cast<Move>(j);
        double& moveScore = scores[j];

        #pragma omp parallel for
        for (int i = 0; i < numberOfSimulationsPerMove; i++) {
            int threadIndex = omp_get_thread_num();
            std::mt19937& localGen = generators[threadIndex];
            Game gameCopy = move(game, currentMove);
            simulate(gameCopy, moveScore, localGen);
        }
    }

    // Get best move from Monte Carlo Computation
    double bestScore = 0;
    Move bestMove;

    for (int i = 0; i < 4; i++) {
        double& moveScore = scores[i];

        if (moveScore > bestScore) {
            bestScore = moveScore;
            bestMove = static_cast<Move>(i);
        }
    }

    return bestMove;
}
