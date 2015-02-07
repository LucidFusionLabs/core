#include "gtest/gtest.h"
#include "lfapp/lfapp.h"
#include "chess.h"

using namespace LFL;

GTEST_API_ int main(int argc, const char **argv) {
    testing::InitGoogleTest(&argc, (char**)argv);
    LFL::FLAGS_default_font = LFL::FakeFont::Filename();
    CHECK_EQ(LFL::app->Create(argc, argv, __FILE__), 0);
    return RUN_ALL_TESTS();
}

TEST(OccupancyMaskTest, Rook) {
    for (int p=0; p<64; p++) {
        BitBoard mask = 0;
        for (int i=p+8; i<56;                    i+=8) mask |= (1L<<i);
        for (int i=p-8; i>=8;                    i-=8) mask |= (1L<<i);
        for (int i=p+1; i%8!=7 && i%8!=0;         i++) mask |= (1L<<i);
        for (int i=p-1; i%8!=7 && i%8!=0 && i>=0; i--) mask |= (1L<<i);
        EXPECT_EQ(rookOccupancyMask[p], mask);
    }
}

TEST(OccupancyMaskTest, Bishop) {
    for (int p=0; p<64; p++) {
        BitBoard mask = 0;
        for (int i=p+9; i%8!=7 && i%8!=0 && i< 56; i+=9) mask |= (1L<<i);
        for (int i=p-9; i%8!=7 && i%8!=0 && i>= 8; i-=9) mask |= (1L<<i);
        for (int i=p+7; i%8!=7 && i%8!=0 && i< 56; i+=7) mask |= (1L<<i);
        for (int i=p-7; i%8!=7 && i%8!=0 && i>= 8; i-=7) mask |= (1L<<i);
        EXPECT_EQ(bishopOccupancyMask[p], mask);
    }
}

TEST(OccupancyMaskTest, Knight) {
    for (int p=0; p<64; p++) {
        BitBoard mask = 0;
        if (p%8!=6 && p%8!=7 && p+10 < 64) mask |= (1L<<(p+10));
        if (          p%8!=7 && p+17 < 64) mask |= (1L<<(p+17));
        if (p%8!=1 && p%8!=0 && p+ 6 < 64) mask |= (1L<<(p+ 6));
        if (          p%8!=0 && p+15 < 64) mask |= (1L<<(p+15));
        if (p%8!=1 && p%8!=0 && p-10 >= 0) mask |= (1L<<(p-10));
        if (          p%8!=0 && p-17 >= 0) mask |= (1L<<(p-17));
        if (p%8!=6 && p%8!=7 && p- 6 >= 0) mask |= (1L<<(p- 6));
        if (          p%8!=7 && p-15 >= 0) mask |= (1L<<(p-15));
        EXPECT_EQ(knightOccupancyMask[p], mask);
    }
}

TEST(OccupancyMaskTest, King) {
    for (int p=0; p<64; p++) {
        BitBoard mask = 0;
        if (            p+8 < 64) mask |= (1L<<(p+8));
        if (            p-8 >= 0) mask |= (1L<<(p-8));
        if (p%8 != 0 && p-1 >= 0) mask |= (1L<<(p-1));
        if (p%8 != 0 && p-9 >= 0) mask |= (1L<<(p-9));
        if (p%8 != 0 && p+7 < 64) mask |= (1L<<(p+7));
        if (p%8 != 7 && p+1 < 64) mask |= (1L<<(p+1));
        if (p%8 != 7 && p+9 < 64) mask |= (1L<<(p+9));
        if (p%8 != 7 && p-7 >= 0) mask |= (1L<<(p-7));
        EXPECT_EQ(kingOccupancyMask[p], mask);
    }
}

TEST(Boards, ByteBoard) {
    for (int i=0; i<64; i++) {
        EXPECT_EQ(  rookOccupancyMask[i], BitBoardFromString(BitBoardToString(  rookOccupancyMask[i]).c_str()));
        EXPECT_EQ(bishopOccupancyMask[i], BitBoardFromString(BitBoardToString(bishopOccupancyMask[i]).c_str()));
    }

    EXPECT_EQ(white_initial[Chess::PAWN],   ByteBoardToBitBoard(initialByteBoard, 'p'));
    EXPECT_EQ(white_initial[Chess::KNIGHT], ByteBoardToBitBoard(initialByteBoard, 'n'));
    EXPECT_EQ(white_initial[Chess::BISHOP], ByteBoardToBitBoard(initialByteBoard, 'b'));
    EXPECT_EQ(white_initial[Chess::ROOK],   ByteBoardToBitBoard(initialByteBoard, 'r'));
    EXPECT_EQ(white_initial[Chess::QUEEN],  ByteBoardToBitBoard(initialByteBoard, 'q'));
    EXPECT_EQ(white_initial[Chess::KING],   ByteBoardToBitBoard(initialByteBoard, 'k'));

    EXPECT_EQ(black_initial[Chess::PAWN],   ByteBoardToBitBoard(initialByteBoard, 'P'));
    EXPECT_EQ(black_initial[Chess::KNIGHT], ByteBoardToBitBoard(initialByteBoard, 'N'));
    EXPECT_EQ(black_initial[Chess::BISHOP], ByteBoardToBitBoard(initialByteBoard, 'B'));
    EXPECT_EQ(black_initial[Chess::ROOK],   ByteBoardToBitBoard(initialByteBoard, 'R'));
    EXPECT_EQ(black_initial[Chess::QUEEN],  ByteBoardToBitBoard(initialByteBoard, 'Q'));
    EXPECT_EQ(black_initial[Chess::KING],   ByteBoardToBitBoard(initialByteBoard, 'K'));

    EXPECT_EQ(Chess::QueenMoves(Chess::Position("........\n"
                                                ".KP.....\n"
                                                ".PP.N...\n"
                                                "P...QP..\n"
                                                "........\n"
                                                "..pp..p.\n"
                                                "ppk.nq..\n"
                                                "........\n"), Chess::E5, Chess::BLACK),
              BitBoardFromString("00000001\n"
                                 "00000010\n"
                                 "00010100\n"
                                 "01110000\n"
                                 "00011100\n"
                                 "00101010\n"
                                 "00001000\n"
                                 "00000000\n"));

    unsigned long long n = 37;
    while (n) {
        unsigned long long i = n & -n;
        INFO("hrm i ", i);
        n ^= i;
    }
}
