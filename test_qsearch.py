import chess

def _quiescence_eval(board, engine, max_depth, depth=0, alpha=-float('inf'), beta=float('inf')):
    is_white = board.turn == chess.WHITE
    
    # Base case: game over
    if board.is_game_over():
        if board.is_checkmate():
            return -1.0 if is_white else 1.0
        return 0.0
        
    # Get static eval
    info = engine.analyse(board, chess.engine.Limit(depth=1))
    sc = info["score"].white()
    cp = 10000 if sc.is_mate() and sc.mate() > 0 else (-10000 if sc.is_mate() else sc.score())
    stand_pat = max(-1.0, min(1.0, cp / 1000.0))
    
    if depth >= max_depth:
        return stand_pat
        
    # In Q-search, you have the option to stand pat (not capture)
    # except when in check!
    in_check = board.is_check()
    if not in_check:
        if is_white:
            if stand_pat > alpha:
                alpha = stand_pat
            if alpha >= beta:
                return alpha
        else:
            if stand_pat < beta:
                beta = stand_pat
            if beta <= alpha:
                return beta
                
    captures = []
    for m in board.legal_moves:
        # In check, all legal moves are forcing. Out of check, only captures and checks
        if in_check or board.is_capture(m) or board.gives_check(m):
            captures.append(m)
            
    if not captures:
        return stand_pat
        
    best_val = stand_pat if not in_check else (-float('inf') if is_white else float('inf'))
    
    for m in captures:
        board.push(m)
        val = _quiescence_eval(board, engine, max_depth, depth + 1, alpha, beta)
        board.pop()
        
        if is_white:
            best_val = max(best_val, val)
            alpha = max(alpha, best_val)
        else:
            best_val = min(best_val, val)
            beta = min(beta, best_val)
            
        if alpha >= beta:
            break
            
    return best_val
