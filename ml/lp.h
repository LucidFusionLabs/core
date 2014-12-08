/*
 * $Id: lp.h 1306 2014-11-16 07:13:16Z justin $
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

#ifndef __LFL_ML_LP_H__
#define __LFL_ML_LP_H__
namespace LFL {

struct LinearProgram {
    int M, N;
    vector<int> basic, non_basic;
    unordered_map<int, int> var;
    Matrix A, Ab, Ai, Aj, Abinv, B, C, Cb, Ci, pi, piAi, chat, bhat, ahat;
    double Z0, objective, epsilon = 1e-9;
    struct Flag { enum { Dualize=1 }; };
    struct Result { enum { Final=1, Unbounded=2, Infeasible=3 }; };

    LinearProgram(File *f) {
        IterWordIter words(new FileLineIter(f), true);
        words.ScanN(&M, 1);
        words.ScanN(&N, 1);
        basic    .resize(M);
        non_basic.resize(N);
        if (1)            words.ScanN(&    basic[0],     basic.size());
        if (1)            words.ScanN(&non_basic[0], non_basic.size());
        InitVars();
        MatrixRowIter(&B) words.ScanN( B    .row(i), 1);
        MatrixRowIter(&A) words.ScanN( A    .row(i), N);
        if (1)            words.ScanN(&Z0,           1);
        MatrixRowIter(&C) words.ScanN( C    .row(i), 1);
        objective = Z0;
    }
    LinearProgram(const LinearProgram &lp, int flag=0) {
        bool dualize = flag & Flag::Dualize;
        if (!dualize) { 
            M         = lp.M;
            N         = lp.N;
            Z0        = lp.Z0;
            basic     = lp.basic;
            non_basic = lp.non_basic;
            objective = lp.objective;
            InitVars();
            A.assignL(&lp.A);
            B.assignL(&lp.B);
            C.assignL(&lp.C);
        } else {
            M         =  lp.N;
            N         =  lp.M;
            Z0        = -lp.Z0;
            basic     =  lp.non_basic;
            non_basic =  lp.basic;
            objective = -lp.objective;
            InitVars();
            A.assignL(lp.A.transpose(), mDelA|mNeg);
            B.assignL(&lp.C, mNeg);
            C.assignL(&lp.B, mNeg);
        }
        epsilon = lp.epsilon;
    }
    void InitVars() {
        A.open(M, N);
        B.open(M, 1);
        C.open(N, 1);
        int ind = 0;
        for (auto i : non_basic) var[i] = ind++;
        for (auto i : basic)     var[i] = ind++;
    }
    void InitPivotVars() {
        Ab   .open(M, M);
        Abinv.open(M, M);
        Ai   .open(M, N);
        Aj   .open(M, 1);
        Cb   .open(M, 1);
        Ci   .open(N, 1);
        pi   .open(1, M);
        piAi .open(1, N);
        chat .open(1, N);
        bhat .open(M, 1);
        ahat .open(M, 1);
    }
    void ClearPivotVars() {
        Ab   .clear();
        Abinv.clear();
        Ai   .clear();
        Aj   .clear();
        Cb   .clear();
        Ci   .clear();
        pi   .clear();
        piAi .clear();
        chat .clear();
        bhat .clear();
        ahat .clear();
    }
    void PushConstraints(int n) {
        int var_max = 0;
        for (auto i :     basic) Typed::Max(&var_max, i);
        for (auto i : non_basic) Typed::Max(&var_max, i);
        for (int i=0; i<n; i++) basic.push_back(var_max+1+i);
        M += n;
        A.addrows(n);
        B.addrows(n);
        ClearPivotVars();
    }
    void PopConstraints(int n) {
        for (int i=0; i<n; i++) basic.pop_back();
        M -= n;
        A.addrows(-n);
        B.addrows(-n);
        ClearPivotVars();
    }
    void Print() const {
        INFO("M=", M, ", N=", N, ", Z0=", Z0, " objective=", objective);
        INFO("(non_basic X: ", Vec<int>::str(&non_basic[0], non_basic.size()), ")");
        MatrixRowIter(&A) INFO("X", basic[i], " | ", B.row(i)[0], ", ", Vec<double>::str(A.row(i), A.N));
        INFO("----------------------------------");
        INFO("Z  | ", objective, ", ", Vec<double>::str(C.m, C.M));
    }
    void PrintRevised() const {
        INFO("M=", M, ", N=", N, ", Z0=", Z0, " objective=", objective);
        INFO("basic: ",     Vec<int>::str(&    basic[0],     basic.size()));
        INFO("non_basic: ", Vec<int>::str(&non_basic[0], non_basic.size()));
        Matrix::print(&A, "A");
        INFO("B = [", Vec<double>::str(B.m, B.M), "]");
        INFO("C = [", Vec<double>::str(C.m, C.M), "]");
    }
    void PrintRevisedPivot() const {
        Matrix::print(&Ab,   "Ab");
        Matrix::print(&Ai,   "Ai");
        Matrix::print(&Cb,   "Cb");
        Matrix::print(&Ci,   "Ci");
        Matrix::print(&pi,   "pi");
        Matrix::print(&piAi, "piAi");
        Matrix::print(&chat, "chat");
        Matrix::print(&bhat, "bhat");
        Matrix::print(&ahat, "ahat");
    }
    bool Feasible() const { MatrixRowIter(&B) if (B.row(i)[0] < 0) return 0; return 1; }
    int ComplementaryVariableID(int v) const { return v <= N ? M + v : v - N; }
    int EnteringVariableIndex(double *m, int n) const {
        int min_id = INT_MAX, min_ind = -1;
        for (int i=0; i<n; i++) if (m[i] > epsilon && Typed::Min(&min_id, non_basic[i])) min_ind = i;
        return min_ind;
    }
    void LoadProblemColumn(int var_id, Matrix *Aout, int aj, Matrix *Cout) const {
        int vj = FindOrDie(var, var_id), vN = non_basic.size();
        MatrixRowIter(&A) Aout->row(i)[aj] = (vj < vN) ? Typed::Negate(A.row(i)[vj]) : (vj - vN == i);
        if (1)            Cout->row(aj)[0] = (vj < vN) ? C.row(vj)[0] : 0; 
    }
    void ChangeObjectiveToConstant(double v=-1) { MatrixRowIter(&C) C.row(i)[0] = v; }
    void UpdateDictionaryFromLastPivot() {
        MatrixRowIter(&C) C.row(i)[0] = chat.row(0)[i];
        MatrixRowIter(&B) B.row(i)[0] = bhat.row(i)[0];
        Matrix CopyA(A);
        Matrix::mult(&Abinv, &Ai, &A, mNeg);
        Z0 = objective;
    }
    void Pivot(int *entering_id, int *leaving_id) {
        int Abj = 0, Aij = 0;
        if (!Ab.M) InitPivotVars();
        for (auto v :     basic) LoadProblemColumn(v, &Ab, Abj++, &Cb);
        for (auto v : non_basic) LoadProblemColumn(v, &Ai, Aij++, &Ci);

        Invert(&Ab, &Abinv);
        Matrix::mult(&Cb, &Abinv, &pi, mTrnpA);
        Matrix::mult(&pi, &Ai, &piAi);
        Matrix::sub(Ci.transpose(), &piAi, &chat, mDelA);
        Matrix::mult(&Abinv, &B, &bhat);

        int entering_ind = EnteringVariableIndex(chat.row(0), chat.N), leaving_ind = -1;
        if (entering_ind < 0) { *entering_id = *leaving_id = -1; return; }
        *entering_id = non_basic[entering_ind];

        MatrixIter(&Aj) Aj.row(i)[j] = Ai.row(i)[entering_ind];
        Matrix::mult(&Abinv, &Aj, &ahat, mNeg);

        double min_inc = DBL_MAX, Ahati;
        MatrixIter(&ahat) {
            if ((Ahati = ahat.row(i)[0]) >= -epsilon) continue;
            double inc = bhat.row(i)[0] / -Ahati;
            if (!Typed::Min(&min_inc, inc) &&
                !(Equal(inc, min_inc, epsilon) && basic[i] < *leaving_id)) continue;
            *leaving_id = basic[i];
            leaving_ind = i;
        }
        if (leaving_ind < 0) { *leaving_id = -1; return; }

        objective = Z0 + Vector::dot(pi.m, B.m, pi.N) + chat.row(0)[entering_ind] * min_inc;
        Typed::Swap(non_basic[entering_ind], basic[leaving_ind]);
    }
    static int Optimize(LinearProgram *LP) {
        LP->PrintRevised();
        int entering_id=0, leaving_id=0, pivots=0;
        for (;;pivots++) {
            LP->Pivot(&entering_id, &leaving_id);
            if (entering_id<0 || leaving_id<0) break;
            INFO("X", entering_id, " enters, and X", leaving_id, " leaves, objective=", LP->objective);
        }
        if      (entering_id < 0) { printf("%f\n%d\n", LP->objective, pivots); }
        else if ( leaving_id < 0) { printf("UNBOUNDED\n"); return Result::Unbounded; }
        return Result::Final;
    }
    static void Solve(File *f, bool integer_constrained, int max_iters=10) {
        LinearProgram LP(f);
        if (!LP.Feasible()) {
            INFO("Input LP not feasible, constructing dual");
            LinearProgram OrigLP(LP, 0);
            LP.ChangeObjectiveToConstant();
            LinearProgram DualLP(LP, LinearProgram::Flag::Dualize);
            if (Optimize(&DualLP) == Result::Unbounded) { printf("INFEASIBLE\n"); return; }
            DualLP.UpdateDictionaryFromLastPivot();
            LP = LinearProgram(DualLP, LinearProgram::Flag::Dualize);
            LP.Z0 = OrigLP.Z0;
            MatrixRowIter(&LP.C) LP.C.row(i)[0] = 0;
            MatrixRowIter(&LP.C) {
                int vj = FindOrDie(LP.var, OrigLP.non_basic[i]), vN = LP.non_basic.size();
                if (vj < vN)              LP.C.row(vj)[0] += OrigLP.C.row(i)[0];
                else MatrixColIter(&LP.A) LP.C.row( j)[0] += OrigLP.C.row(i)[0] * LP.A.row(vj - vN)[j];
                if (vj >= vN)             LP.Z0           += OrigLP.C.row(i)[0] * LP.B.row(vj - vN)[0];
            }
            LP.objective = LP.Z0;
        }
        INFO("Optimizing ", integer_constrained ? "relaxation" : "objective");
        if (Optimize(&LP) != Result::Final) return;
        LP.UpdateDictionaryFromLastPivot();
        int total_cuts = 0;
        for (int iter=0; integer_constrained && iter<max_iters; iter++) {
            LP.PushConstraints(LP.basic.size());
            int integer_rows = 0;
            for (int i=0, m=LP.M/2; i<m; i++) {
                double v = LP.B.row(i)[0];
                v = -v + floor(v);
                if (Equal(v, 0, LP.epsilon) || Equal(fabs(v), 1, LP.epsilon)) { integer_rows++; continue; }
                LP.B.row(m+i-integer_rows)[0] = v;
                MatrixColIter(&LP.A) {
                    v = -LP.A.row(i)[j];
                    LP.A.row(m+i-integer_rows)[j] = v - floor(v);
                }
            }
            total_cuts += LP.M/2 - integer_rows;
            bool done = LP.M/2 == integer_rows;
            LP.PopConstraints(integer_rows);
            if (done) return;
            INFO("Adding cutting-planes, (iter=%d/%d)", iter+1, max_iters);
            LP.PrintRevised();
            INFO("Constructing dual");
            LinearProgram DualLP(LP, LinearProgram::Flag::Dualize);
            if (Optimize(&DualLP) == Result::Unbounded) { printf("INFEASIBLE\n"); return; }
            DualLP.UpdateDictionaryFromLastPivot();
            INFO("Reconstructing primal");
            LP = LinearProgram(DualLP, LinearProgram::Flag::Dualize);
            LP.PrintRevised();
            printf("%f added %d cuts (iter=%d/%d)\n", LP.objective, total_cuts, iter+1, max_iters);
        }
    }
};

}; // namespace LFL
#endif // __LFL_ML_LP_H__
