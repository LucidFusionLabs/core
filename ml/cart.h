/*
 * $Id: cart.h 1309 2014-10-10 19:20:55Z justin $
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

#ifndef __LFL_ML_CART_H__
#define __LFL_ML_CART_H__
namespace LFL {

/* classification and regression tree */
struct CART {
    struct Feature { int ind, count; string name; }; 

    struct FeatureSet {
        virtual int size() const = 0;
        virtual Feature *get(int ind) = 0;
    };

    struct Question {
        const char *name;
        virtual bool answer(const Feature *) const = 0;
    };

    struct QuestionList {
        const char *name;
        virtual int size() const = 0;
        virtual Question *get(int ind) = 0;
    };
    
    /* classification */
    struct Tree {
        static const int map_buckets=5, map_values=2;
        QuestionList *inventory;
        Matrix questions, namemap_data, leafnamemap_data;
        HashMatrix namemap, leafnamemap;
        vector<string> name;
        Tree() : namemap(&namemap_data, map_values), leafnamemap(&leafnamemap_data, map_values) {}
    };

    static const double *query(const Tree *model, const Feature *f) {
        string node="t";
        for (;;) {
            const double *he = model->namemap.Get(fnv32(node.c_str()));
            if (!he) break;

            int ind = he[1], qind = model->questions.row(ind)[0];
            const Question *q = model->inventory->get(qind);
            node += q->answer(f) ? "l" : "r";
        }

        return model->leafnamemap.Get(fnv32(node.c_str()));
    }

    static int read(const char *dir, const char *name, int lastiter, QuestionList *inventory, Tree *out) {
        out->inventory = inventory;
        Matrix *outp[] = { &out->questions, out->namemap.map, out->leafnamemap.map };
        vector<string> *outs[] = { &out->name };
        if (MatrixFile::ReadVersioned(dir, name, "questions",   &outp[0], 0, lastiter)<0) { ERROR(name, ".", lastiter, ".questions"  ); return -1; }
        if (MatrixFile::ReadVersioned(dir, name, "namemap",     &outp[1], 0, lastiter)<0) { ERROR(name, ".", lastiter, ".namemap"    ); return -1; }
        if (MatrixFile::ReadVersioned(dir, name, "leafnamemap", &outp[2], 0, lastiter)<0) { ERROR(name, ".", lastiter, ".leafnamemap"); return -1; }
        if (StringFile::ReadVersioned(dir, name, "name",        &outs[0], 0, lastiter)<0) { ERROR(name, ".", lastiter, ".name"       ); return -1; }
        return 0;
    }

    static void blank(QuestionList *inventory, Tree *out, double val) {
        out->inventory = inventory;
        out->namemap    .map->Open(4, Tree::map_buckets*Tree::map_values);
        out->leafnamemap.map->Open(4, Tree::map_buckets*Tree::map_values);
        double *he = out->leafnamemap.Set(fnv32("t"));
        he[1] = val;
    }

    /* regression */
    struct TreeBuilder {
        vector<string> name;
        vector<int> question;
        map<string, unsigned> leafname;
    };
    
    struct SplitRanker {
        virtual void parent(FeatureSet *pool, const vector<int> &parent) = 0;
        virtual double rank(FeatureSet *pool, const vector<int> &left, const vector<int> &right) = 0;
    };
    
    static void split(FeatureSet *pool, Question *q, const vector<int> &parent, vector<int> &left, vector<int> &right) {
        for (int i=0; i<parent.size(); i++) {
            Feature *f = pool->get(parent[i]);
            if (q->answer(f)) left.push_back(f->ind);
            else             right.push_back(f->ind);
        }
    }

    static int best_split(FeatureSet *pool, QuestionList *Q, SplitRanker *ranker, const vector<int> &parent, int minsize) {
        ranker->parent(pool, parent);

        double max; int maxind = -1;
        for (int i=0; i<Q->size(); i++) {
            Question *q = Q->get(i);
            
            vector<int> left, right;
            split(pool, q, parent, left, right);

            int leftsum=0, rightsum=0;
            for (int j=0; j<left.size();  j++)  leftsum += pool->get(left [j])->count;
            for (int j=0; j<right.size(); j++) rightsum += pool->get(right[j])->count;
            if (leftsum<minsize || rightsum<minsize) continue;

            double rank = ranker->rank(pool, left, right);
            if (maxind == -1 || rank > max) { max = rank; maxind = i; }
        }
        return maxind;
    }

    static void grow_tree(FeatureSet *pool, QuestionList *Q, SplitRanker *ranker, string name, const vector<int> &node, int stopnodesize, TreeBuilder &out) {
        if (node.size() < stopnodesize) { INFO(name, " done with ", node.size()); return add_leaf(pool, name, node, out);  }

        int best_question = best_split(pool, Q, ranker, node, stopnodesize);
        if (best_question < 0) { INFO(name, " done because best_question = ", best_question); return add_leaf(pool, name, node, out); }
        out.name.push_back(name);
        out.question.push_back(best_question);

        vector<int> left, right;
        Question *q = Q->get(best_question);

        split(pool, q, node, left, right);
        INFOf("%s %d best split by %s into %sl %d and %sr %d", name.c_str(), node.size(), q->name, name.c_str(), left.size(), name.c_str(), right.size());

        grow_tree(pool, Q, ranker, name + "l", left,  stopnodesize, out);
        grow_tree(pool, Q, ranker, name + "r", right, stopnodesize, out);
    }

    static void add_leaf(FeatureSet *pool, string name, const vector<int> &node, TreeBuilder &out) {
        int max, maxind=-1;
        for (int i=0; i<node.size(); i++) {
            Feature *f = pool->get(node[i]);
            if (maxind == -1 || f->count > max) { max = f->count; maxind = i; }
        }
        const char *leafname = pool->get(node[maxind])->name.c_str();
        out.leafname[name] = fnv32(leafname);
        INFO(name, " named leaf ", leafname);
    }

    static int write_tree(const TreeBuilder &tree, QuestionList *QL, const char *name, const char *dir, int iteration) {
        if (tree.name.size() != tree.question.size()) { ERROR("invalid size ", tree.name.size(), " ", tree.question.size()); return -1; }
        const char *flagtext = QL->name;

        int buckets=Tree::map_buckets, values=Tree::map_values;
        Matrix namemap_data    (NextPrime(tree.name.size()*4),     buckets*values);
        Matrix leafnamemap_data(NextPrime(tree.leafname.size()*4), buckets*values);
        HashMatrix namemap(&namemap_data, values), leafnamemap(&leafnamemap_data, values);

        LocalFile names    (string(dir) + MatrixFile::Filename(name, "name",      "string", iteration), "w");
        LocalFile questions(string(dir) + MatrixFile::Filename(name, "questions", "matrix", iteration), "w");

        MatrixFile::WriteHeader(&names,     BaseName(names.Filename()),     flagtext, tree.name.size(),     1);
        MatrixFile::WriteHeader(&questions, BaseName(questions.Filename()), flagtext, tree.question.size(), 1);

        for (int i=0; i<tree.question.size(); i++) {
            const char *n = tree.name[i].c_str();
            StringFile::WriteRow(&names, n);

            double qrow[1] = { (double)tree.question[i] };
            MatrixFile::WriteRow(&questions, qrow, 1);

            double *he = namemap.Set(fnv32(n));
            if (!he) FATAL("Matrix hash collision: ", n);
            he[1] = i;
        }

        for (map<string, unsigned>::const_iterator i = tree.leafname.begin(); i != tree.leafname.end(); i++) {
            const char *n = (*i).first.c_str();
            unsigned v = (*i).second;
            double *he = leafnamemap.Set(fnv32(n));
            if (!he) FATAL("Matrix hash collision: ", n);
            he[1] = v;
        }

        if (MatrixFile(namemap.map,     flagtext).WriteVersioned(dir, name,     "namemap", iteration)<0) { ERROR(name, " write namemap"    ); return -1; }
        if (MatrixFile(leafnamemap.map, flagtext).WriteVersioned(dir, name, "leafnamemap", iteration)<0) { ERROR(name, " write leafnamemap"); return -1; }

        return 0;
    }
};

struct PhoneticDecisionTree {
    struct Feature : public CART::Feature { int D; double *mean, *covar; };

    struct FeatureSet : public CART::FeatureSet {
        int N; Feature *S;
        
        FeatureSet(int n) : N(n), S(new Feature[N]) {}
        ~FeatureSet() { delete [] S; }

        int size() const { return N; }
        CART::Feature *get(int ind) { return (ind>=0 && ind<N) ? &S[ind] : 0; }
    };

    struct Question : public CART::Question {
        int is_phone;
        vector<int> member;

        bool answer(const CART::Feature *feat) const {
            Feature *f = (Feature *)feat;
            int prevPhone, nextPhone, phone=AcousticModel::parseName(f->name, 0, &prevPhone, &nextPhone);
            if (phone < 0) ERROR("AcousticModel::parseName failed for ", f->name);

            int question_refers = is_phone < 0 ? prevPhone : nextPhone;
            for (int i=0; i<member.size(); i++) if (member[i] == question_refers) return true;
            return false;
        }
    };

    struct QuestionList : public CART::QuestionList {
        int N; Question *L;

        QuestionList(int n) : N(n), L(new Question[N]) { name="PhoneticDecisionTree::QuestionList"; }
        ~QuestionList() { delete [] L; }

        int size() const { return N; }
        CART::Question *get(int ind) { return (ind>=0 && ind<N) ? &L[ind] : 0; }

        static QuestionList *create() {
            int count=0;
#           undef  XX
#           define XX(x) count++;
#           undef  YY
#           define YY(x)
#           include "speech/phonics.h"

            QuestionList *QL = new QuestionList(count);
            Question *q;
            int phone=-1, ind=0;
#           undef  XX
#           define XX(x) q = &QL->L[ind++]; q->name = #x; q->is_phone = phone;
#           undef  YY
#           define YY(x) q->member.push_back(Phoneme::x);
#           include "speech/phonics.h"

            phone=1; ind=0;
#           include "speech/phonics.h"

            return QL;
        }
    };

    struct SplitRanker : public CART::SplitRanker {
        double parentLikelihood;

        void parent(CART::FeatureSet *pool, const vector<int> &node) { parentLikelihood = likelihood(pool, node); }

        double rank(CART::FeatureSet *pool, const vector<int> &left, const vector<int> &right) {
            double leftLikelihood = likelihood(pool, left), rightLikelihood = likelihood(pool, right);
            return leftLikelihood + rightLikelihood - parentLikelihood;
        }

        static double likelihood(CART::FeatureSet *pool, const vector<int> &node) {
            int D = ((Feature *)pool->get(0))->D;
            double *mean = (double *)alloca(D*sizeof(double));
            double *covar = (double *)alloca(D*sizeof(double));
            double *x = (double *)alloca(D*sizeof(double));
            double denom = 0;

            /* determine node mean */
            memset(mean, 0, D*sizeof(double));
            for (int i=0; i<node.size(); i++) {
                Feature *f = (Feature *)pool->get(node[i]);

                Vector::Assign(x, f->mean, D);
                Vector::Mult(x, f->count, D);
                Vector::Add(mean, x, D);

                denom += f->count;
            }
            Vector::Div(mean, denom, D);

            /* determine node covariance */
            denom = 0;
            memset(covar, 0, D*sizeof(double));
            for (int i=0; i<node.size(); i++) {
                Feature *f = (Feature *)pool->get(node[i]);

                Vector::Assign(x, f->mean, D);
                Vector::Mult(x, x, D);
                Vector::Add(x, f->covar, D);
                Vector::Mult(x, f->count, D);
                Vector::Add(covar, x, D);

                denom += f->count;
            }
            Vector::Div(covar, denom, D);

            Vector::Assign(x, mean, D);
            Vector::Mult(x, x, D);

            Vector::Sub(covar, x, D);

            /* determine node likelihood */ 
            return -0.5 * ((1 + log(2*M_PI))*D + DiagDet(covar, D)) * denom;
        }
    };
};

}; // namespace LFL
#endif // __LFL_ML_CART_H__
