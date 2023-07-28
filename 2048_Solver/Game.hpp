// Game class for the 2048 game logic
// Author: Fabrice Renard
// Date : 23 / 06 / 2023

#ifndef GAME_H
#define GAME_H

#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <QApplication>
#include <QDebug>
#include <vector>

using Grid = std::vector<std::vector<int>>;
enum class Move { LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3 };

struct Compare {
	bool operator() (const int& a, const int& b) const {
		return a > b;
	}
};

class Game : public QObject {
	Q_OBJECT
public:
	Game(int gridSize);
	Game(const Game& other);
	bool moveLeft();
	bool moveRight();
	bool moveUp();
	bool moveDown();
	int getScore();
	Grid getGrid();
	int getGridSize();
	bool isGameOver();
	friend bool operator==(const Game& left, const Game& right);

public slots:
	void handleKeyPress(char key, std::map<double, Move, Compare> bestMoves, std::shared_ptr<Game> game);

private:
	bool merge();
	bool compress();
	void flip();
	void transpose();
	bool addTile();
	bool makeMove(Move move);

	Grid grid_;
	int gridSize_;
	int score_;
	std::random_device rd_;
	std::mt19937 gen_;

	friend class GameWindow;
};

#endif // !GAME_H
