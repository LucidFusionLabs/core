/*
 * $Id: chess.h 1309 2014-10-10 19:20:55Z justin $
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __LFL_CHESS_CHESS_H__
#define __LFL_CHESS_CHESS_H__
namespace LFL {

static const char initialByteBoard[] = 
"RNBQKBNR\n"
"PPPPPPPP\n"
"........\n"
"........\n"
"........\n"
"........\n"
"pppppppp\n"
"rnbqkbnr\n";

typedef string ByteBoard;
typedef unsigned long long BitBoard;

//                                      { 0, pawns,        knights,    bishops,    rooks,      queen,      king      };
static const BitBoard white_initial[] = { 0, 0xff00ULL,     0x42ULL,     0x24ULL,     0x81ULL,     0x10ULL,     0x8ULL     };
static const BitBoard black_initial[] = { 0, 0xff00ULL<<40, 0x42ULL<<56, 0x24ULL<<56, 0x81ULL<<56, 0x10ULL<<56, 0x8ULL<<56 };

string BitBoardToString(BitBoard b) {
    string ret;
    for (int i=7; i>=0; i--) {
        for (int j=7; j>=0; j--) StrAppend(&ret, (bool)(b & (1L<<(i*8+j))));
        StrAppend(&ret, "\n");
    }
    return ret;
}

BitBoard BitBoardFromString(const char *buf, char v='1') {
    BitBoard ret = 0;
    for (int i=7, bi=0; i>=0; i--, bi++) {
        for (int j=7; j>=0; j--, bi++) if (buf[bi] == v) ret |= (1L<<(i*8+j));
    }
    return ret;
}

BitBoard ByteBoardToBitBoard(const ByteBoard &buf, char piece) {
    BitBoard ret = 0;
    for (int i=7, bi=0; i>=0; i--, bi++) {
        for (int j=7; j>=0; j--, bi++) if (buf[bi] == piece) ret |= (1L<<(i*8+j));
    }
    return ret;
}

#include "magic.h"

struct Chess {
    enum { WHITE=0, BLACK=1 };
    static const char *ColorName(int n) {
        static const char *name[] = { "white", "black" };
        CHECK_RANGE(n, 0, 2);
        return name[n];
    }

    enum { ALL=0, PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6 };
    static const char *PieceName(int n) {
        static const char *name[] = { "", "pawn", "knight", "bishop", "rook", "queen", "king" };
        CHECK_RANGE(n, 0, 7);
        return name[n];
    }
    static const char ByteBoardPieceSymbol(int n, bool color) {
        static const char white_piece_value[] = { '.', 'p', 'n', 'b', 'r', 'q', 'k' };
        static const char black_piece_value[] = { '.', 'P', 'N', 'B', 'R', 'Q', 'K' };
        CHECK_RANGE(n, 0, 7);
        return color ? black_piece_value[n] : white_piece_value[n];
    }

    enum {
        H1=0,  G1=1,  F1=2,  E1=3,  D1=4,  C1=5,  B1=6,  A1=7,
        H2=8,  G2=9,  F2=10, E2=11, D2=12, C2=13, B2=14, A2=15,
        H3=16, G3=17, F3=18, E3=19, D3=20, C3=21, B3=22, A3=23,
        H4=24, G4=25, F4=26, E4=27, D4=28, C4=29, B4=30, A4=31,
        H5=32, G5=33, F5=34, E5=35, D5=36, C5=37, B5=38, A5=39,
        H6=40, G6=41, F6=42, E6=43, D6=44, C6=45, B6=46, A6=47,
        H7=48, G7=49, F7=50, E7=51, D7=52, C7=53, B7=54, A7=55,
        H8=56, G8=57, F8=58, E8=59, D8=60, C8=61, B8=62, A8=63
    };
    static int         SquareX   (int p) { return 7 - (p % 8); }
    static int         SquareY   (int p) { return p / 8; }
    static const char *SquareName(int p) {
        static const char *name[] = {
            "H1", "G1", "F1", "E1", "D1", "C1", "B1", "A1",
            "H2", "G2", "F2", "E2", "D2", "C2", "B2", "A2",
            "H3", "G3", "F3", "E3", "D3", "C3", "B3", "A3",
            "H4", "G4", "F4", "E4", "D4", "C4", "B4", "A4",
            "H5", "G5", "F5", "E5", "D5", "C5", "B5", "A5",
            "H6", "G6", "F6", "E6", "D6", "C6", "B6", "A6",
            "H7", "G7", "F7", "E7", "D7", "C7", "B7", "A7",
            "H8", "G8", "F8", "E8", "D8", "C8", "B8", "A8"
        };
        CHECK_RANGE(p, 0, 64);
        return name[p];
    };

    struct Position {
        bool move_color;
        BitBoard white[7], black[7], white_moves[7], black_moves[7];
        Position(const char *b, bool to_move_color=WHITE) { LoadByteBoard(b, to_move_color); }
        Position() { Reset(); }

        void Reset() {
            move_color = 0;
            for (int i=PAWN; i<=KING; i++) white[i] = white_initial[i];
            for (int i=PAWN; i<=KING; i++) black[i] = black_initial[i];
            SetAll(WHITE);
            SetAll(BLACK);
        }

        void LoadByteBoard(const string &b, bool to_move_color=WHITE) {
            move_color = to_move_color;
            for (int i=PAWN; i<=KING; i++) white[i] = ByteBoardToBitBoard(b, ByteBoardPieceSymbol(i, WHITE));
            for (int i=PAWN; i<=KING; i++) black[i] = ByteBoardToBitBoard(b, ByteBoardPieceSymbol(i, BLACK));
            SetAll(WHITE);
            SetAll(BLACK);
        }

        void SetAll(bool color) {
            BitBoard *pieces = Pieces(color);
            pieces[ALL] = pieces[PAWN] | pieces[KNIGHT] | pieces[BISHOP] | pieces[ROOK] | pieces[QUEEN] | pieces[KING];
        }

        BitBoard AllPieces() const { return white[ALL] | black[ALL]; }

              BitBoard *Pieces(bool color)       { return color ? black       : white;       }
              BitBoard *Moves (bool color)       { return color ? black_moves : white_moves; }
        const BitBoard *Pieces(bool color) const { return color ? black       : white;       }
        const BitBoard *Moves (bool color) const { return color ? black_moves : white_moves; }
    };

    static BitBoard PawnMoves(const Position &in, int p, bool black) {
        return 0;
        // return pawnOccupancyMask[p] & ~in.Pieces(black)[ALL] | pawnAttackMask[p] & in.Pieces(!black)[ALL];
    }
    
    static BitBoard KnightMoves(const Position &in, int p, bool black) {
        return knightOccupancyMask[p] & ~in.Pieces(black)[ALL];
    }

    static BitBoard BishopMoves(const Position &in, int p, bool black) {
        static MagicMoves *magic_moves = Singleton<MagicMoves>::Get();
        BitBoard blockers = in.AllPieces() & bishopOccupancyMask[p];
        return magic_moves->BishopMoves(p, blockers, in.Pieces(black)[ALL]);
    }

    static BitBoard RookMoves(const Position &in, int p, bool black) {
        static MagicMoves *magic_moves = Singleton<MagicMoves>::Get();
        BitBoard blockers = in.AllPieces() & rookOccupancyMask[p];
        int magicIndex = MagicHash(p, blockers, rookMagicNumber, rookMagicNumberBits);
        return magic_moves->rookMagicMoves[p][magicIndex] & ~in.Pieces(black)[ALL];
    }

    static BitBoard QueenMoves(const Position &in, int p, bool black) {
        return RookMoves(in, p, black) | BishopMoves(in, p, black);
    }

    static BitBoard KingMoves(const Position &in, int p, bool black) {
        return kingOccupancyMask[p] & ~in.Pieces(black)[ALL];
    }

    static bool InCheck(const Position &in, bool color) {
        return in.Moves(!color)[ALL] & in.Pieces(color)[KING];
    }
};

}; // namespace LFL
#endif // #define __LFL_CHESS_CHESS_H__
