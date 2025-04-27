# Chess Console-Based Game with Min-Max Algorithm & Alpha-Beta Pruning

## Overview

This project implements a console-based chess game in Python, where you can play against the computer. The computer uses the **Min-Max Algorithm with Alpha-Beta Pruning** to determine the best moves. The game is played on a standard 8x8 chessboard, and features detection of checkmate, stalemate, and draw conditions. 

## Features

- **Play Against Computer**: Compete against an AI that uses the Min-Max algorithm with Alpha-Beta pruning.
- **Move Display**: Displays the chessboard and pieces after each move.
- **Move History**: Tracks and shows the moves made by both players.
- **Checkmate & Stalemate Detection**: Detects checkmate and stalemate, and declares the winner or a draw accordingly.
- **Command-line Interface**: The game operates entirely through a command-line interface for simplicity.

## Files

1. **chess.py**: The main Python file containing the chess game logic, implementing the game board, move validation, and AI functionality using the Min-Max algorithm with Alpha-Beta pruning.
2. **chess.docs**: A report explaining the implementation of the game, including the details of the Min-Max algorithm, Alpha-Beta pruning, and other features.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AbdulMoiz2493/Chess-ConsoleBased.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd Chess-ConsoleBased
   ```

3. **Run the game**:
   ```bash
   python chess.py
   ```

## How to Play

1. **Start the Game**: Run the game and follow the prompts.
2. **Make a Move**: The game will ask for your move in standard chess notation (e.g., "e2 to e4").
3. **AI Move**: The computer will calculate its move using the Min-Max algorithm and display the board after both moves.
4. **Game End**: The game will detect if the game ends in checkmate, stalemate, or a draw and display the outcome.

## Game Features

- **Min-Max Algorithm**: The computer uses a strategy to evaluate all possible moves and select the best one.
- **Alpha-Beta Pruning**: Optimizes the Min-Max algorithm to skip evaluating branches of the game tree that won't affect the final decision.
- **Checkmate Detection**: The game checks if either player has won by capturing the opponent’s king.
- **Stalemate Detection**: The game recognizes if no legal moves are possible for the player, resulting in a draw.

## Report

A detailed implementation report is available in the `chess.docs` file, which explains the underlying logic of the game, the Min-Max algorithm, Alpha-Beta pruning, and how the game handles various scenarios.

## Contribution

Feel free to fork the repository and submit pull requests for any improvements or features you'd like to contribute to.

## Contact

For questions or further information, you can reach me at:

- **Email**: [abdulmoiz8895@gmail.com](mailto:abdulmoiz8895@gmail.com)
- **Portfolio**: [abdul-moiz.tech](https://abdul-moiz.tech)

## License

This project is open-source and available under the [MIT License](LICENSE).
