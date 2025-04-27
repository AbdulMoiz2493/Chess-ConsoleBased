import copy
import os

# Constants for pieces
WHITE_KING = '♔'
WHITE_QUEEN = '♕'
WHITE_ROOK = '♖'
WHITE_BISHOP = '♗'
WHITE_KNIGHT = '♘'
WHITE_PAWN = '♙'
BLACK_KING = '♚'
BLACK_QUEEN = '♛'
BLACK_ROOK = '♜'
BLACK_BISHOP = '♝'
BLACK_KNIGHT = '♞'
BLACK_PAWN = '♟'
EMPTY = ' '

# Constants for colors and players
WHITE = 'white'
BLACK = 'black'
HUMAN = 'human'
COMPUTER = 'computer'

class Piece:
    def __init__(self, color, symbol, value):
        self.color = color
        self.symbol = symbol
        self.value = value
        self.has_moved = False

    def get_moves(self, board, pos):
        """Return a list of possible moves for this piece"""
        pass

    def get_symbol(self):
        return self.symbol


class King(Piece):
    def __init__(self, color):
        symbol = WHITE_KING if color == WHITE else BLACK_KING
        super().__init__(color, symbol, 100)
    
    def get_moves(self, board, pos):
        moves = []
        row, col = pos
        
        # All 8 directions a king can move
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for d_row, d_col in directions:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if board[new_row][new_col] is None:
                    moves.append((new_row, new_col))
                elif board[new_row][new_col].color != self.color:
                    moves.append((new_row, new_col))
                    
        # Check for castling
        if not self.has_moved:
            # Kingside castling
            if (col + 3 < 8 and 
                board[row][col+1] is None and 
                board[row][col+2] is None and 
                isinstance(board[row][col+3], Rook) and 
                board[row][col+3].color == self.color and 
                not board[row][col+3].has_moved):
                moves.append((row, col+2))  # Castling move
                
            # Queenside castling
            if (col - 4 >= 0 and 
                board[row][col-1] is None and 
                board[row][col-2] is None and 
                board[row][col-3] is None and 
                isinstance(board[row][col-4], Rook) and 
                board[row][col-4].color == self.color and 
                not board[row][col-4].has_moved):
                moves.append((row, col-2))  # Castling move
        
        return moves


class Queen(Piece):
    def __init__(self, color):
        symbol = WHITE_QUEEN if color == WHITE else BLACK_QUEEN
        super().__init__(color, symbol, 9)
    
    def get_moves(self, board, pos):
        moves = []
        row, col = pos
        
        # Combine rook and bishop movements
        # All 8 directions a queen can move
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for d_row, d_col in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * d_row, col + i * d_col
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    if board[new_row][new_col] is None:
                        moves.append((new_row, new_col))
                    elif board[new_row][new_col].color != self.color:
                        moves.append((new_row, new_col))
                        break
                    else:
                        break
                else:
                    break
        
        return moves


class Rook(Piece):
    def __init__(self, color):
        symbol = WHITE_ROOK if color == WHITE else BLACK_ROOK
        super().__init__(color, symbol, 5)
    
    def get_moves(self, board, pos):
        moves = []
        row, col = pos
        
        # All 4 directions a rook can move
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        
        for d_row, d_col in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * d_row, col + i * d_col
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    if board[new_row][new_col] is None:
                        moves.append((new_row, new_col))
                    elif board[new_row][new_col].color != self.color:
                        moves.append((new_row, new_col))
                        break
                    else:
                        break
                else:
                    break
        
        return moves


class Bishop(Piece):
    def __init__(self, color):
        symbol = WHITE_BISHOP if color == WHITE else BLACK_BISHOP
        super().__init__(color, symbol, 3)
    
    def get_moves(self, board, pos):
        moves = []
        row, col = pos
        
        # All 4 diagonal directions a bishop can move
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for d_row, d_col in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * d_row, col + i * d_col
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    if board[new_row][new_col] is None:
                        moves.append((new_row, new_col))
                    elif board[new_row][new_col].color != self.color:
                        moves.append((new_row, new_col))
                        break
                    else:
                        break
                else:
                    break
        
        return moves


class Knight(Piece):
    def __init__(self, color):
        symbol = WHITE_KNIGHT if color == WHITE else BLACK_KNIGHT
        super().__init__(color, symbol, 3)
    
    def get_moves(self, board, pos):
        moves = []
        row, col = pos
        
        # All 8 possible knight moves
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for d_row, d_col in knight_moves:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if board[new_row][new_col] is None:
                    moves.append((new_row, new_col))
                elif board[new_row][new_col].color != self.color:
                    moves.append((new_row, new_col))
        
        return moves


class Pawn(Piece):
    def __init__(self, color):
        symbol = WHITE_PAWN if color == WHITE else BLACK_PAWN
        super().__init__(color, symbol, 1)
        self.en_passant_vulnerable = False
    
    def get_moves(self, board, pos):
        moves = []
        row, col = pos
        
        # Forward direction depends on color
        forward = -1 if self.color == WHITE else 1
        
        # Forward move
        if 0 <= row + forward < 8 and board[row + forward][col] is None:
            moves.append((row + forward, col))
            
            # Double forward from starting position
            if ((self.color == WHITE and row == 6) or (self.color == BLACK and row == 1)) and \
               board[row + 2 * forward][col] is None:
                moves.append((row + 2 * forward, col))
        
        # Diagonal captures
        for d_col in [-1, 1]:
            if 0 <= row + forward < 8 and 0 <= col + d_col < 8:
                # Normal capture
                if (board[row + forward][col + d_col] is not None and 
                    board[row + forward][col + d_col].color != self.color):
                    moves.append((row + forward, col + d_col))
                
                # En passant capture
                elif (board[row][col + d_col] is not None and 
                     isinstance(board[row][col + d_col], Pawn) and 
                     board[row][col + d_col].color != self.color and 
                     board[row][col + d_col].en_passant_vulnerable):
                    moves.append((row + forward, col + d_col))  # En passant move
        
        return moves


class ChessGame:
    def __init__(self, human_color=WHITE):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = WHITE
        self.human_player = human_color
        self.computer_player = BLACK if human_color == WHITE else WHITE
        self.move_history = []
        self.initialize_board()
        
    def initialize_board(self):
        # Initialize pieces for both sides
        # Black pieces (top of board)
        self.board[0][0] = Rook(BLACK)
        self.board[0][1] = Knight(BLACK)
        self.board[0][2] = Bishop(BLACK)
        self.board[0][3] = Queen(BLACK)
        self.board[0][4] = King(BLACK)
        self.board[0][5] = Bishop(BLACK)
        self.board[0][6] = Knight(BLACK)
        self.board[0][7] = Rook(BLACK)
        
        for col in range(8):
            self.board[1][col] = Pawn(BLACK)
        
        # White pieces (bottom of board)
        self.board[7][0] = Rook(WHITE)
        self.board[7][1] = Knight(WHITE)
        self.board[7][2] = Bishop(WHITE)
        self.board[7][3] = Queen(WHITE)
        self.board[7][4] = King(WHITE)
        self.board[7][5] = Bishop(WHITE)
        self.board[7][6] = Knight(WHITE)
        self.board[7][7] = Rook(WHITE)
        
        for col in range(8):
            self.board[6][col] = Pawn(WHITE)
    
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_board(self):
        """Display the current state of the chess board"""
        self.clear_screen()
        print("   a b c d e f g h")
        print(" +-----------------+")
    
        for i in range(8):
            print(f"{8-i}|", end=" ")
            for j in range(8):
                piece = self.board[i][j]
                if piece is None:
                    # Use checkerboard pattern for empty squares
                    cell = '·' if (i + j) % 2 == 0 else ' '
                else:
                    cell = piece.get_symbol()
                print(cell, end=" ")
            print(f"|{8-i}")
    
        print(" +-----------------+")
        print("   a b c d e f g h")
    
        print(f"\nCurrent player: {'Human' if self.current_player == self.human_player else 'Computer'} ({self.current_player})")
    
        # Display move history (limited to last 10 moves for readability)
        #if self.move_history:
           # print("\nRecent move:")
            #start_idx = max(0, len(self.move_history) - 1)
            #for i in range(start_idx, len(self.move_history)):
            #    move_num = i // 2 + 1
            #    player = "White" if i % 2 == 0 else "Black"
             #   print(f"{move_num}. {player}: {self.move_history[i]}")
    
    def parse_position(self, pos_str):
        """Convert chess notation to board indices (e.g., 'e4' -> (4, 4))"""
        if len(pos_str) != 2:
            return None
        
        col = ord(pos_str[0].lower()) - ord('a')
        row = 8 - int(pos_str[1])
        
        if 0 <= row < 8 and 0 <= col < 8:
            return (row, col)
        return None
    
    def position_to_notation(self, pos):
        """Convert board indices to chess notation (e.g., (4, 4) -> 'e4')"""
        row, col = pos
        return f"{chr(col + ord('a'))}{8 - row}"
    
    def get_piece_at(self, pos):
        """Get the piece at the given position"""
        row, col = pos
        return self.board[row][col]
    
    def find_king(self, color):
        """Find the position of the king of the given color"""
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if isinstance(piece, King) and piece.color == color:
                    return (row, col)
        return None
    
    def is_in_check(self, color):
        """Check if the king of the given color is in check"""
        king_pos = self.find_king(color)
        if king_pos is None:
            return False
        
        opponent_color = BLACK if color == WHITE else WHITE
        
        # Check all opponent pieces to see if they can attack the king
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece is not None and piece.color == opponent_color:
                    moves = piece.get_moves(self.board, (row, col))
                    if king_pos in moves:
                        return True
        
        return False
    
    def make_move(self, from_pos, to_pos):
        """Make a move on the board and return True if successful, False otherwise"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        piece = self.board[from_row][from_col]
        
        if piece is None or piece.color != self.current_player:
            return False
        
        if to_pos not in piece.get_moves(self.board, from_pos):
            return False
        
        # Special handling for castling
        castling = False
        if isinstance(piece, King) and abs(from_col - to_col) == 2:
            castling = True
            # Kingside castling
            if to_col > from_col:
                rook_from = (from_row, 7)
                rook_to = (from_row, 5)
            # Queenside castling
            else:
                rook_from = (from_row, 0)
                rook_to = (from_row, 3)
            
            # Move the rook
            self.board[rook_to[0]][rook_to[1]] = self.board[rook_from[0]][rook_from[1]]
            self.board[rook_from[0]][rook_from[1]] = None
            self.board[rook_to[0]][rook_to[1]].has_moved = True
        
        # Special handling for en passant
        en_passant = False
        if isinstance(piece, Pawn) and from_col != to_col and self.board[to_row][to_col] is None:
            en_passant = True
            # Remove the captured pawn
            self.board[from_row][to_col] = None
        
        # Reset en passant flags
        for row in range(8):
            for col in range(8):
                if self.board[row][col] is not None and isinstance(self.board[row][col], Pawn):
                    self.board[row][col].en_passant_vulnerable = False
        
        # Set en passant flag for double pawn move
        if isinstance(piece, Pawn) and abs(from_row - to_row) == 2:
            piece.en_passant_vulnerable = True
        
        # Make the move
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        piece.has_moved = True
        
        # Pawn promotion (automatically to Queen for simplicity)
        if isinstance(piece, Pawn) and (to_row == 0 or to_row == 7):
            self.board[to_row][to_col] = Queen(piece.color)
        
        # Update move history
        move_notation = self.position_to_notation(from_pos) + self.position_to_notation(to_pos)
        self.move_history.append(move_notation)
        
        # Switch player
        self.current_player = BLACK if self.current_player == WHITE else WHITE
        
        return True
    
    def undo_move(self, from_pos, to_pos, captured_piece=None, was_castling=False, 
                 was_en_passant=False, was_promotion=False, original_piece=None):
        """Undo a move (used during AI evaluation)"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        piece = self.board[to_row][to_col]
        
        # Undo pawn promotion
        if was_promotion:
            self.board[from_row][from_col] = original_piece
        else:
            self.board[from_row][from_col] = piece
            
        self.board[to_row][to_col] = captured_piece
        
        # Undo castling
        if was_castling:
            # Kingside castling
            if to_col > from_col:
                rook_from = (from_row, 5)
                rook_to = (from_row, 7)
            # Queenside castling
            else:
                rook_from = (from_row, 3)
                rook_to = (from_row, 0)
            
            # Move the rook back
            self.board[rook_to[0]][rook_to[1]] = self.board[rook_from[0]][rook_from[1]]
            self.board[rook_from[0]][rook_from[1]] = None
        
        # Undo en passant
        if was_en_passant:
            # Restore the captured pawn
            self.board[from_row][to_col] = Pawn(BLACK if self.current_player == WHITE else WHITE)
        
        # Switch player back
        self.current_player = BLACK if self.current_player == WHITE else WHITE
    
    def get_all_legal_moves(self, color):
        """Get all legal moves for the given color"""
        all_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece is not None and piece.color == color:
                    from_pos = (row, col)
                    for to_pos in piece.get_moves(self.board, from_pos):
                        # Make a temporary copy of the board to check if move is legal
                        temp_game = copy.deepcopy(self)
                        if temp_game.make_move(from_pos, to_pos):
                            # If the move doesn't leave the king in check, it's legal
                            if not temp_game.is_in_check(color):
                                all_moves.append((from_pos, to_pos))
        
        return all_moves
    
    def is_checkmate(self, color):
        """Check if the given color is in checkmate"""
        if not self.is_in_check(color):
            return False
        
        return len(self.get_all_legal_moves(color)) == 0
    
    def is_stalemate(self, color):
        """Check if the given color is in stalemate"""
        if self.is_in_check(color):
            return False
        
        return len(self.get_all_legal_moves(color)) == 0
    
    def evaluate_board(self):
        """Evaluate the current board position (positive for white advantage, negative for black)"""
        score = 0
        
        # Material value
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece is not None:
                    piece_value = piece.value
                    if piece.color == WHITE:
                        score += piece_value
                    else:
                        score -= piece_value
        
        # Check and checkmate
        if self.is_checkmate(WHITE):
            score = -1000
        elif self.is_checkmate(BLACK):
            score = 1000
        elif self.is_in_check(WHITE):
            score -= 5
        elif self.is_in_check(BLACK):
            score += 5
        
        # Mobility (number of legal moves)
        white_moves = len(self.get_all_legal_moves(WHITE))
        black_moves = len(self.get_all_legal_moves(BLACK))
        score += 0.1 * (white_moves - black_moves)
        
        # Center control (bonus for controlling the center squares)
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for row, col in center_squares:
            piece = self.board[row][col]
            if piece is not None:
                if piece.color == WHITE:
                    score += 0.5
                else:
                    score -= 0.5
        
        return score
    
    def minimax(self, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning"""
        if depth == 0:
            return self.evaluate_board(), None
        
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for from_pos, to_pos in self.get_all_legal_moves(WHITE):
                # Make a move
                piece = self.board[from_pos[0]][from_pos[1]]
                captured_piece = self.board[to_pos[0]][to_pos[1]]
                was_castling = isinstance(piece, King) and abs(from_pos[1] - to_pos[1]) == 2
                was_en_passant = (isinstance(piece, Pawn) and from_pos[1] != to_pos[1] and 
                                 captured_piece is None)
                was_promotion = isinstance(piece, Pawn) and (to_pos[0] == 0 or to_pos[0] == 7)
                original_piece = copy.deepcopy(piece) if was_promotion else None
                
                self.make_move(from_pos, to_pos)
                
                # Evaluate this move
                eval_score, _ = self.minimax(depth - 1, alpha, beta, False)
                
                # Undo the move
                self.undo_move(from_pos, to_pos, captured_piece, was_castling, 
                              was_en_passant, was_promotion, original_piece)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (from_pos, to_pos)
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for from_pos, to_pos in self.get_all_legal_moves(BLACK):
                # Make a move
                piece = self.board[from_pos[0]][from_pos[1]]
                captured_piece = self.board[to_pos[0]][to_pos[1]]
                was_castling = isinstance(piece, King) and abs(from_pos[1] - to_pos[1]) == 2
                was_en_passant = (isinstance(piece, Pawn) and from_pos[1] != to_pos[1] and 
                                 captured_piece is None)
                was_promotion = isinstance(piece, Pawn) and (to_pos[0] == 0 or to_pos[0] == 7)
                original_piece = copy.deepcopy(piece) if was_promotion else None
                
                self.make_move(from_pos, to_pos)
                
                # Evaluate this move
                eval_score, _ = self.minimax(depth - 1, alpha, beta, True)
                
                # Undo the move
                self.undo_move(from_pos, to_pos, captured_piece, was_castling, 
                              was_en_passant, was_promotion, original_piece)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (from_pos, to_pos)
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    def get_computer_move(self):
        """Get the best move for the computer using minimax with alpha-beta pruning"""
        # Make sure we're getting moves for the right player
        if self.current_player != self.computer_player:
            print("Error: Attempting to get computer move when it's not computer's turn")
            return None
        
        try:
            # Keep depth limited to prevent long calculation times
            depth = 2
        
            # Start the minimax search
            _, best_move = self.minimax(depth, float('-inf'), float('inf'), 
                                    self.computer_player == WHITE)
                                
            # Validate the move
            if best_move is None:
                # Fallback to a simple legal move if minimax fails
                all_moves = self.get_all_legal_moves(self.computer_player)
                if all_moves:
                    import random
                    # Choose a random move from legal moves for variety
                    best_move = random.choice(all_moves)
                else:
                    print("No legal moves found for computer")
                    return None
                
            # Additional validation - check if move is in legal moves list
            if best_move not in self.get_all_legal_moves(self.computer_player):
                print("Invalid move generated by AI. Selecting random legal move instead.")
                all_moves = self.get_all_legal_moves(self.computer_player)
                if all_moves:
                    import random
                    best_move = random.choice(all_moves)
                else:
                    return None
                
            return best_move
        
        except Exception as e:
            print(f"Error in computer move generation: {e}")
            # Fallback to any legal move
            all_moves = self.get_all_legal_moves(self.computer_player)
            if all_moves:
                import random
                return random.choice(all_moves)
            return None


    def play_game(self):
        """Main game loop"""
        game_over = False
        move_count = 0
        max_moves = 100  # Set a reasonable limit to prevent infinite games

        while not game_over and move_count < max_moves:
            self.display_board()

            # Check for checkmate or stalemate
            if self.is_checkmate(WHITE):
                print("Checkmate! Black wins!")
                game_over = True
                break
            elif self.is_checkmate(BLACK):
                print("Checkmate! White wins!")
                game_over = True
                break
            elif self.is_stalemate(self.current_player):
                print("Stalemate! The game is a draw.")
                game_over = True
                break
            
            # Human's turn
            if self.current_player == self.human_player:
                valid_move = False
                while not valid_move:
                    try:
                        move_input = input("\nEnter your move (e.g., 'e2e4'): ")
                        if move_input.lower() == 'quit':
                            print("Game ended by player.")
                            return

                        if len(move_input) != 4:
                            print("Invalid input format. Please use format like 'e2e4'.")
                            continue
                        
                        from_pos = self.parse_position(move_input[:2])
                        to_pos = self.parse_position(move_input[2:])

                        if from_pos is None or to_pos is None:
                            print("Invalid position. Please use format like 'e2e4'.")
                            continue
                        
                        piece = self.get_piece_at(from_pos)
                        if piece is None or piece.color != self.human_player:
                            print("No valid piece at the starting position.")
                            continue
                        
                        # Check if move is in legal moves
                        if (from_pos, to_pos) not in self.get_all_legal_moves(self.human_player):
                            print("That move is not legal. Try again.")
                            continue

                        valid_move = self.make_move(from_pos, to_pos)

                        if not valid_move:
                            print("Invalid move. Try again.")
                    except Exception as e:
                        print(f"Error: {e}")

            # Computer's turn
            else:
                print("\nComputer is thinking...")
                computer_move = self.get_computer_move()

                if computer_move is None:
                    print("Computer couldn't find a valid move. Game ends in draw.")
                    game_over = True
                    break

                from_pos, to_pos = computer_move
                move_success = self.make_move(from_pos, to_pos)

                if not move_success:
                    print("Computer tried an invalid move. Game ends.")
                    game_over = True
                    break

                print(f"Computer moved: {self.position_to_notation(from_pos)}{self.position_to_notation(to_pos)}")
                input("Press Enter to continue...")

            move_count += 1

        if move_count >= max_moves:
            print("Game ended due to move limit (100 moves). It's a draw.")


if __name__ == "__main__":
    # Ask the user which color they want to play
    color_choice = input("Which color do you want to play? (white/black): ").lower()
    human_color = WHITE if color_choice.startswith('w') else BLACK
    
    game = ChessGame(human_color)
    game.play_game()